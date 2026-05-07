"""RxRx3-core per-FOV JP2 fetch from HuggingFace parquet shards + stack into Zarr v3.

Images ship as 35 inline-bytes parquet shards on HuggingFace
(``recursionpharma/rxrx3-core``).  Each row is one channel image: the
``__key__`` column encodes ``{experiment}/Plate{plate}/{address}_s1_{ch}``
(1-indexed) and the ``jp2`` column holds raw JP2 bytes.

Concurrency model: shards are loaded one at a time (sequential, large network
fetch).  Within each shard a :class:`~concurrent.futures.ThreadPoolExecutor`
decodes FOVs in parallel — JP2 decode with Pillow releases the GIL via the C
extension, so thread-level parallelism is effective here.  The ``key→bytes``
dict built from the shard is read-only and shared across threads without any
additional locking.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lamin_utils import logger

from cp_bg_bench.io.zarr_io import open_plate_group, plate_store_path, write_fov_array

__all__ = [
    "MAX_DECODE_WORKERS",
    "PLATE_GROUP_COLUMNS",
    "decode_jp2",
    "download_batch_to_zarr",
    "load_shard",
]

PLATE_GROUP_COLUMNS: tuple[str, str, str] = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
)

# JP2 decode is CPU-bound (Pillow C extension); 8 threads saturates typical
# compute node cores without excessive memory pressure from held shard tables.
MAX_DECODE_WORKERS: int = 8

# HuggingFace webdataset-as-parquet: image bytes are stored under the column
# named after the file extension. RxRx3-core uses JPEG 2000 files → "jp2".
_JP2_COLUMN = "jp2"

_N_TOTAL_SHARDS = 35


def _read_hf_token() -> str | None:
    """Read HuggingFace token from the standard cache file, if present."""
    from pathlib import Path

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        token = token_file.read_text().strip()
        return token or None
    return None


def load_shard(hf_repo: str, shard_idx: int) -> dict[str, bytes]:
    """Open one parquet shard from HuggingFace and return a ``{key: jp2_bytes}`` dict.

    Only the ``__key__`` and ``jp2`` columns are read — all other columns
    (metadata, etc.) are skipped to minimise transfer.
    """
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem  # lazy: optional io dep

    shard_path = f"datasets/{hf_repo}/data/train-{shard_idx:05d}-of-{_N_TOTAL_SHARDS:05d}.parquet"
    _token = os.environ.get("HF_TOKEN") or _read_hf_token()
    fs = HfFileSystem(token=_token)
    with fs.open(shard_path, "rb") as fh:
        table = pq.read_table(fh, columns=["__key__", _JP2_COLUMN])

    keys = table["__key__"].to_pylist()
    blobs = table[_JP2_COLUMN].to_pylist()
    # HuggingFace Image feature stores bytes as {"bytes": ..., "path": ...} structs.
    raw: list[bytes] = [b["bytes"] if isinstance(b, dict) else b for b in blobs]
    return dict(zip(keys, raw, strict=True))


def decode_jp2(jp2_bytes: bytes) -> np.ndarray:
    """Decode raw JP2 bytes to a ``(H, W)`` uint8 array via Pillow."""
    from PIL import Image  # lazy: pillow is an optional io-feature dep

    return np.asarray(Image.open(BytesIO(jp2_bytes)).convert("L"))


def _fetch_and_write_fov(
    row: dict[str, Any],
    key_to_bytes: dict[str, bytes],
    plate_group: Any,
    n_channels: int,
) -> str:
    """Decode all channels for one FOV, stack to ``(C, H, W)`` uint16, write to Zarr.

    ``key_to_bytes`` is a read-only shared dict built from the pre-loaded shard
    table; concurrent access from multiple threads is safe for Python dicts.

    Returns the written ``fov_id``.
    """
    fov_id = row["id"]
    key_prefix = row["parquet_key_prefix"]

    planes: list[np.ndarray] = []
    for ch in range(1, n_channels + 1):
        key = f"{key_prefix}_{ch}"
        jp2_bytes = key_to_bytes.get(key)
        if jp2_bytes is None:
            raise KeyError(
                f"JP2 key {key!r} not found in shard. "
                "The shard index may be stale — delete cache and rerun resolve_metadata."
            )
        planes.append(decode_jp2(jp2_bytes))

    shapes = {p.shape for p in planes}
    if len(shapes) > 1:
        raise ValueError(
            f"Channel shape mismatch for FOV {fov_id!r}: got shapes {[p.shape for p in planes]}"
        )

    stack = np.stack(planes, axis=0).astype(np.uint16, copy=False)
    write_fov_array(plate_group, fov_id, stack)
    return fov_id


def _validate_meta_batch(meta_batch: pd.DataFrame, n_channels: int) -> None:
    if meta_batch.empty:
        raise ValueError("meta_batch is empty — nothing to download")
    required = {"id", "parquet_shard", "parquet_key_prefix", *PLATE_GROUP_COLUMNS}
    missing = required - set(meta_batch.columns)
    if missing:
        raise KeyError(
            f"meta_batch missing columns: {sorted(missing)}. "
            "Rule A's parquet may be stale; "
            "rerun `snakemake --forceall resolve_metadata` to regenerate."
        )
    if meta_batch["id"].duplicated().any():
        duped = meta_batch.loc[meta_batch["id"].duplicated(), "id"].tolist()
        raise ValueError(f"duplicate FOV ids in meta_batch: {duped[:5]}")
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")


def download_batch_to_zarr(
    meta_batch: pd.DataFrame,
    output_root: Path | str,
    hf_repo: str,
    channel_names: list[str],
    n_threads: int = 4,
) -> dict[str, int]:
    """Fetch every FOV in ``meta_batch`` from HuggingFace shards into Zarr v3 stores.

    Processing order: shards are loaded one at a time (sequential network fetch).
    Within each shard, FOV decode+write is parallelised across
    ``min(n_threads, MAX_DECODE_WORKERS)`` threads.

    The ``parquet_shard`` column in ``meta_batch`` is the pre-computed shard
    index from :func:`~cp_bg_bench.io.rxrx3_core.resolve_metadata`.

    Returns a ``{store_path: n_fovs_written}`` summary.
    Idempotent: existing FOV arrays with matching shape are no-ops.
    """
    n_channels = len(channel_names)
    _validate_meta_batch(meta_batch, n_channels)

    output_root = Path(output_root)
    effective_workers = max(1, min(int(n_threads), MAX_DECODE_WORKERS))
    plate_cols = list(PLATE_GROUP_COLUMNS)

    summary: dict[str, int] = {}

    for shard_idx, shard_rows in meta_batch.groupby("parquet_shard", sort=True):
        shard_idx = int(shard_idx)
        logger.info(
            f"rxrx3_core: loading shard {shard_idx} "
            f"({len(shard_rows)} FOVs across "
            f"{shard_rows.groupby(plate_cols).ngroups} plates)"
        )
        key_to_bytes = load_shard(hf_repo, shard_idx)

        for plate_key, plate_rows in shard_rows.groupby(plate_cols, sort=False):
            source, batch, plate = plate_key
            store_path = plate_store_path(output_root, source, batch, plate)
            plate_group = open_plate_group(store_path)

            rows = plate_rows.to_dict(orient="records")
            n_written = 0

            pool = ThreadPoolExecutor(max_workers=effective_workers)
            try:
                futures = [
                    pool.submit(
                        _fetch_and_write_fov,
                        row,
                        key_to_bytes,
                        plate_group,
                        n_channels,
                    )
                    for row in rows
                ]
                for fut in as_completed(futures):
                    fov_id = fut.result()
                    n_written += 1
                    logger.debug(f"wrote FOV {fov_id} to {store_path}")
            except Exception:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)

            summary[str(store_path)] = summary.get(str(store_path), 0) + n_written
            logger.info(
                f"plate store {store_path.name}: wrote {n_written} FOVs from shard {shard_idx}"
            )

    return summary
