"""HuggingFace dataset build and reshard helpers (rules H and M)."""

from __future__ import annotations

import gc
import logging
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "VARIANT_NAMES",
    "build_hf_features",
    "parquet_dir_to_hf",
    "reshard_dataset",
]

# The four output variants produced by rules H, J, K, L.
VARIANT_NAMES = ("crops", "seg", "crops_density", "seg_density")

# Canonical columns that flow from extract_crops_plate through to HF datasets.
# perturbation/batch/treatment are derived per-dataset in Rule F.
_META_COLS = [
    "Metadata_JCP2022",
    "Metadata_InChIKey",
    "Metadata_PlateType",
    "perturbation",
    "batch",
    "treatment",
]


def build_hf_features(extra_meta_cols: list[str] | None = None) -> Any:
    """Build the explicit HF Features schema for crop datasets.

    ``extra_meta_cols`` defaults to the standard annotation columns.
    Pass an empty list to omit them.
    """
    from datasets import Features, Value

    meta = extra_meta_cols if extra_meta_cols is not None else _META_COLS
    base: dict[str, Any] = {
        "row_key": Value("string"),
        "source": Value("string"),
        "plate": Value("string"),
        "well": Value("string"),
        "tile": Value("string"),
        "id_local": Value("int64"),
        "nuc_area": Value("int64"),
        "cyto_area": Value("int64"),
        "nuc_cyto_ratio": Value("float64"),
        "n_cells_in_fov": Value("int64"),
        "n_cells_scaled": Value("float64"),
        "mask": Value("large_binary"),
        "cell": Value("large_binary"),
    }
    for col in meta:
        base[col] = Value("string")
    return Features(base)


def parquet_dir_to_hf(
    parquet_dir: Path | str,
    output_hf_dir: Path | str,
    features: Any,
    tmp_root: Path | str | None = None,
    intermediate_groups: int = 10,
    cleanup_tmp: bool = True,
    row_filter: Callable[[Any], Any] | None = None,
    row_transform: Callable[[Any], Any] | None = None,
) -> None:
    """Convert a directory of parquet shards into a HF Dataset on disk.

    Three-phase aggregation (shards → HF chunks → N intermediate groups →
    final concat) to cap peak RAM usage.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset, concatenate_datasets, load_from_disk

    parquet_dir = Path(parquet_dir)
    output_hf_dir = Path(output_hf_dir)

    shard_files = sorted(str(p) for p in parquet_dir.glob("**/*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No parquet shards under {parquet_dir}")

    logger.info("parquet_dir_to_hf: %d shards → %s", len(shard_files), output_hf_dir)

    tmp = Path(tmp_root) if tmp_root else output_hf_dir.parent / f"{output_hf_dir.name}_tmp"
    chunks_dir = tmp / "chunks"
    _ensure_clean(chunks_dir)

    def _types_mapper(pa_type: pa.DataType) -> pd.ArrowDtype:
        # Promote binary→large_binary to avoid 2 GB offset overflow when
        # boolean-filtering plates with ~16k cells × 135 KB each.
        if pa.types.is_binary(pa_type):
            return pd.ArrowDtype(pa.large_binary())
        return pd.ArrowDtype(pa_type)

    # Phase 1: each parquet shard → HF chunk on disk
    chunk_paths: list[str] = []
    for idx, shard_path in enumerate(shard_files):
        table = pq.read_table(shard_path)
        df = table.to_pandas(types_mapper=_types_mapper)
        if row_filter is not None:
            df = row_filter(df)
        if row_transform is not None:
            df = row_transform(df)
        hf_shard = Dataset.from_pandas(df, preserve_index=False, features=features)
        chunk_dir = chunks_dir / f"chunk_{idx:06d}"
        hf_shard.save_to_disk(str(chunk_dir))
        chunk_paths.append(str(chunk_dir))
        del hf_shard, df, table
        gc.collect()
        logger.debug("converted shard %d/%d", idx + 1, len(shard_files))

    # Phase 2: intermediate aggregation
    groups = _split_for_intermediate(chunk_paths, intermediate_groups)
    intermediate_dirs: list[str] = []
    for i, group in enumerate(groups):
        out = tmp / f"group_{i:03d}"
        _combine_chunks(group, str(out), features)
        intermediate_dirs.append(str(out))
        gc.collect()
        logger.debug("intermediate group %d/%d done", i + 1, len(groups))

    # Phase 3: final concat — load all intermediate groups then concatenate once
    _ensure_clean(output_hf_dir)
    if not intermediate_dirs:
        logger.warning("parquet_dir_to_hf: no data produced, writing empty dataset")
        from datasets import Dataset

        Dataset.from_dict({k: [] for k in features}, features=features).save_to_disk(
            str(output_hf_dir)
        )
    else:
        parts = [load_from_disk(d) for d in intermediate_dirs]
        final = concatenate_datasets(parts)
        final.save_to_disk(str(output_hf_dir))
        logger.info("parquet_dir_to_hf: wrote %d rows → %s", final.num_rows, output_hf_dir)

    if cleanup_tmp:
        shutil.rmtree(tmp, ignore_errors=True)


def reshard_dataset(
    input_hf_dir: Path | str,
    output_hf_dir: Path | str,
    rows_per_shard: int = 25_000,
    max_shards: int = 64,
) -> None:
    """Reshard a HF Dataset into ≤ ``max_shards`` large contiguous files.

    Each shard is materialised as a single Arrow file via a no-op
    ``.map(..., writer_batch_size=rows_per_file)``.
    """
    from datasets import Dataset, concatenate_datasets, load_from_disk

    input_hf_dir = Path(input_hf_dir)
    output_hf_dir = Path(output_hf_dir)
    tmp_parts = output_hf_dir.parent / f"{output_hf_dir.name}_tmp_parts"

    ds_in: Dataset = load_from_disk(str(input_hf_dir))
    n_rows = ds_in.num_rows
    logger.info("reshard: %d rows from %s", n_rows, input_hf_dir)

    if n_rows == 0:
        _ensure_clean(output_hf_dir)
        ds_in.save_to_disk(str(output_hf_dir))
        return

    est = math.ceil(n_rows / max(1, rows_per_shard))
    num_files = min(est, max_shards)
    rows_per_file = math.ceil(n_rows / num_files)

    logger.info("reshard: %d → %d files (~%d rows each)", n_rows, num_files, rows_per_file)

    _ensure_clean(tmp_parts)
    part_dirs: list[Path] = []

    start = 0
    i = 0
    while start < n_rows:
        end = min(n_rows, start + rows_per_file)
        view = ds_in.select(range(start, end))
        view = view.flatten_indices()
        view = view.map(
            lambda b: b,
            batched=True,
            batch_size=end - start,
            num_proc=1,  # identity lambda is not picklable; must stay 1
            load_from_cache_file=False,
            writer_batch_size=end - start,
            desc=f"reshard part {i:04d}",
        )
        part_dir = tmp_parts / f"part_{i:04d}"
        view.save_to_disk(str(part_dir))
        part_dirs.append(part_dir)
        del view
        gc.collect()
        start = end
        i += 1

    _ensure_clean(output_hf_dir)
    parts = [load_from_disk(str(p)) for p in part_dirs]
    final = concatenate_datasets(parts)
    final.save_to_disk(str(output_hf_dir))
    logger.info("reshard: done → %s (%d rows)", output_hf_dir, final.num_rows)

    shutil.rmtree(tmp_parts, ignore_errors=True)


# ── helpers ──────────────────────────────────────────────────────────────────


def _ensure_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _split_for_intermediate(paths: list[str], n: int) -> list[list[str]]:
    if not paths or n <= 1:
        return [paths]
    total = len(paths)
    base, rem = divmod(total, n)
    groups: list[list[str]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        groups.append(paths[start : start + size])
        start += size
    return [g for g in groups if g]


def _combine_chunks(chunk_paths: list[str], output_path: str, features: Any) -> None:
    from datasets import Dataset, concatenate_datasets, load_from_disk

    _ensure_clean(Path(output_path))
    if not chunk_paths:
        Dataset.from_dict({k: [] for k in features}, features=features).save_to_disk(output_path)
        return
    concatenate_datasets([load_from_disk(p) for p in chunk_paths]).save_to_disk(output_path)
