"""RxRx3-core metadata resolution and parquet-shard index helpers.

RxRx3-core is a HUVEC CRISPR KO Cell Painting dataset from Recursion
Pharmaceuticals (HuggingFace: ``recursionpharma/rxrx3-core``).

Images ship as 35 inline-bytes parquet shards. Shards 16–34 contain all
CRISPR perturbation data; shards 0–15 contain compound data and are not
used by this adapter. The ``__key__`` column in each shard encodes the
experiment, plate, well address, and channel index:
    ``{experiment_name}/Plate{plate}/{address}_s1_{channel_idx}``

Column mapping (metadata CSV → pipeline schema)
------------------------------------------------
    experiment_name  → Metadata_Source
    plate (int)      → Metadata_Plate      (cast to str)
    address          → Metadata_Well       (used for id construction)
    gene             → Metadata_InChIKey   (gene symbol; "EMPTY_control" for controls)
    treatment        → Metadata_InChI      (guide-level e.g. "TP53_guide_1")
    perturbation_type → Metadata_PlateType
    (fixed "rxrx3_core") → Metadata_Batch
    (fixed "1")          → Metadata_Site
    id = "{experiment_name}__rxrx3_core__{plate}__{address}__1"
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.parquet as pq
from lamin_utils import logger

if TYPE_CHECKING:
    from cp_bg_bench.config import Rxrx3CoreConfig, Rxrx3CoreSample

__all__ = [
    "assign_snakemake_batches",
    "resolve_metadata",
]

_RXRX3_CORE_BATCH = "rxrx3_core"

# Shards 15–34 contain CRISPR data in rxrx3-core v1 (verified Feb 2025).
# Shard 15 is included conservatively; compounds occupy shards 0–14.
# If Recursion uploads new experiments and the boundary shifts, delete the
# shard index cache (~/.cache/cp-bg-bench/... or the run-specific cache dir)
# and update this range accordingly.
_CRISPR_SHARD_SCAN_RANGE = range(15, 35)
_N_TOTAL_SHARDS = 35

_COL_MAP: dict[str, str] = {
    "experiment_name": "Metadata_Source",
    "plate": "Metadata_Plate",
    "address": "Metadata_Well",
    "gene": "Metadata_InChIKey",
    "treatment": "Metadata_InChI",
    "perturbation_type": "Metadata_PlateType",
}

_SAFE_METADATA_COLUMNS: tuple[str, ...] = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
)
_SAFE_METADATA_VALUE = re.compile(r"^[A-Za-z0-9._-]+$")


def _read_hf_token() -> str | None:
    """Read HuggingFace token from the standard cache file, if present."""
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        token = token_file.read_text().strip()
        return token or None
    return None


_SCALAR_OUTPUT_COLUMNS: tuple[str, ...] = (
    "id",
    "snakemake_batch",
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_PlateType",
    "Metadata_InChIKey",
    "Metadata_InChI",
    "parquet_shard",
    "parquet_key_prefix",
)


def _validate_safe_metadata(fov: pd.DataFrame) -> None:
    for col in _SAFE_METADATA_COLUMNS:
        if col not in fov.columns:
            continue
        as_str = fov[col].astype(str)
        bad_mask = ~as_str.str.fullmatch(_SAFE_METADATA_VALUE)
        if bad_mask.any():
            bad = as_str[bad_mask].unique()[:5].tolist()
            raise ValueError(
                f"{col} contains unsafe value(s) for filesystem paths / composite "
                f"ids: {bad}. Allowed characters are [A-Za-z0-9._-]."
            )


def _download_metadata_csv(hf_repo: str, cache_dir: Path) -> pd.DataFrame:
    """Download ``metadata_rxrx3_core.csv`` from HuggingFace; cache locally.

    The CSV is copied to ``cache_dir`` so subsequent runs skip the download.
    To force a refresh (e.g. after Recursion updates the upstream CSV), delete
    ``{cache_dir}/metadata_rxrx3_core.csv`` and rerun resolve_metadata.
    """
    import os

    from huggingface_hub import hf_hub_download  # lazy: optional io dep

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_csv = cache_dir / "metadata_rxrx3_core.csv"
    _token = os.environ.get("HF_TOKEN") or _read_hf_token()

    if not local_csv.is_file():
        logger.info(f"downloading rxrx3_core metadata from {hf_repo}")
        path = hf_hub_download(
            hf_repo, "metadata_rxrx3_core.csv", repo_type="dataset", token=_token
        )
        local_csv.write_bytes(Path(path).read_bytes())
    else:
        logger.debug(f"rxrx3_core metadata CSV cache hit: {local_csv}")

    return pd.read_csv(local_csv, low_memory=False)


def _build_shard_index(hf_repo: str, cache_dir: Path) -> dict[str, int]:
    """Build a ``parquet_key_prefix → shard_idx`` lookup by scanning shard keys.

    Reads only the ``__key__`` column from shards
    :data:`_CRISPR_SHARD_SCAN_RANGE` — column projection means only the key
    data (~0.5 MB per shard) is transferred, not the JP2 payload.
    Result is cached to ``{cache_dir}/shard_index.json`` so subsequent
    ``resolve_metadata`` calls skip the scan.
    """
    import os

    from huggingface_hub import HfFileSystem  # lazy: optional io dep

    cache_path = cache_dir / "shard_index.json"
    if cache_path.is_file():
        logger.debug(f"rxrx3_core shard index cache hit: {cache_path}")
        return json.loads(cache_path.read_text())

    logger.info(
        f"building rxrx3_core shard index from {len(_CRISPR_SHARD_SCAN_RANGE)} shards "
        f"(reading __key__ column only)"
    )
    # Prefer HF_TOKEN env var; fall back to huggingface_hub token file for
    # authenticated access (higher rate limits, faster downloads).
    _token: str | None = os.environ.get("HF_TOKEN") or _read_hf_token()
    fs = HfFileSystem(token=_token)
    prefix_to_shard: dict[str, int] = {}

    for shard_idx in _CRISPR_SHARD_SCAN_RANGE:
        shard_path = (
            f"datasets/{hf_repo}/data/train-{shard_idx:05d}-of-{_N_TOTAL_SHARDS:05d}.parquet"
        )
        for attempt in range(3):
            try:
                with fs.open(shard_path, "rb") as fh:
                    table = pq.read_table(fh, columns=["__key__"])
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2**attempt
                logger.warning(
                    f"shard {shard_idx} fetch failed (attempt {attempt + 1}/3), "
                    f"retrying in {wait}s: {exc}"
                )
                time.sleep(wait)

        for key in table["__key__"].to_pylist():
            if not key.startswith("gene-"):
                continue
            # Strip channel suffix: "gene-032/Plate7/D27_s1_3" → "gene-032/Plate7/D27_s1"
            prefix = key.rsplit("_", 1)[0]
            prefix_to_shard[prefix] = shard_idx

    logger.info(f"rxrx3_core shard index: {len(prefix_to_shard):,} key prefixes mapped")
    # Atomic write: write to .tmp then rename so a concurrent writer never
    # leaves a partial JSON at the canonical cache path.
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prefix_to_shard))
    tmp.replace(cache_path)
    return prefix_to_shard


def _select_samples(meta: pd.DataFrame, samples: list[Rxrx3CoreSample]) -> pd.DataFrame:
    """Filter to CRISPR rows, then apply the AND-within / OR-across whitelist."""
    crispr = meta[meta["perturbation_type"] == "CRISPR"].copy()

    selected_rows: list[pd.DataFrame] = []
    seen_any_nonempty = False

    for sample in samples:
        mask = pd.Series(True, index=crispr.index)
        if sample.experiment is not None:
            mask &= crispr["experiment_name"] == sample.experiment
        if sample.plate is not None:
            mask &= crispr["plate"] == sample.plate
        if sample.address is not None:
            mask &= crispr["address"] == sample.address
        if sample.gene is not None:
            mask &= crispr["gene"] == sample.gene

        matched = crispr.loc[mask]
        if matched.empty:
            logger.warning(
                f"rxrx3_core whitelist sample matched 0 rows: "
                f"experiment={sample.experiment!r}, plate={sample.plate!r}"
            )
        else:
            selected_rows.append(matched)
            seen_any_nonempty = True

    if not seen_any_nonempty:
        raise ValueError(
            "rxrx3_core whitelist produced an empty selection; check the samples block"
        )

    combined = pd.concat(selected_rows, axis=0)
    return combined.drop_duplicates(subset=["experiment_name", "plate", "address"]).reset_index(
        drop=True
    )


def _build_fov_table(
    meta: pd.DataFrame,
    rxrx3_cfg: Rxrx3CoreConfig,
    shard_index: dict[str, int],
) -> pd.DataFrame:
    """Rename columns, add derived fields, and assign parquet shard + key prefix."""
    fov = meta.rename(columns=_COL_MAP)
    fov["Metadata_Batch"] = _RXRX3_CORE_BATCH
    fov["Metadata_Site"] = "1"

    # Cast identifier columns to str for id composition and path safety checks.
    for col in ("Metadata_Source", "Metadata_Plate", "Metadata_Well"):
        if col in fov.columns:
            fov[col] = fov[col].astype(str)

    _validate_safe_metadata(fov)

    fov["id"] = (
        fov["Metadata_Source"]
        + "__"
        + _RXRX3_CORE_BATCH
        + "__"
        + fov["Metadata_Plate"]
        + "__"
        + fov["Metadata_Well"]
        + "__1"
    )

    # Parquet key prefix: "{experiment_name}/Plate{plate}/{address}_s1"
    fov["parquet_key_prefix"] = (
        fov["Metadata_Source"]
        + "/Plate"
        + fov["Metadata_Plate"]
        + "/"
        + fov["Metadata_Well"]
        + "_s1"
    )

    # Assign shard index from pre-built lookup.
    fov["parquet_shard"] = fov["parquet_key_prefix"].map(shard_index)
    n_missing = fov["parquet_shard"].isna().sum()
    if n_missing:
        missing_prefixes = (
            fov.loc[fov["parquet_shard"].isna(), "parquet_key_prefix"].unique()[:5].tolist()
        )
        raise ValueError(
            f"{n_missing} FOVs not found in any parquet shard. "
            f"Sample missing prefixes: {missing_prefixes}. "
            "Delete the shard index cache and rerun resolve_metadata."
        )
    fov["parquet_shard"] = fov["parquet_shard"].astype(int)

    valid_shards = set(_CRISPR_SHARD_SCAN_RANGE)
    bad_shards = fov.loc[~fov["parquet_shard"].isin(valid_shards), "parquet_shard"].unique()
    if len(bad_shards):
        raise ValueError(
            f"parquet_shard values outside expected range {_CRISPR_SHARD_SCAN_RANGE}: "
            f"{sorted(bad_shards.tolist())}. "
            "The shard index may be stale — delete the cache and rerun resolve_metadata."
        )

    return fov


def assign_snakemake_batches(fov: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Assign ``snakemake_batch`` ids — same contract as the rxrx1 equivalent."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    fov = fov.copy()
    within_source = fov.groupby("Metadata_Source", sort=False).cumcount()
    bucket_idx = within_source // batch_size
    fov["snakemake_batch"] = (
        fov["Metadata_Source"].astype(str) + "_snakemake_batch_" + bucket_idx.astype(str)
    )
    return fov


def resolve_metadata(
    rxrx3_cfg: Rxrx3CoreConfig,
    cache_dir: Path,
    batch_size: int,
) -> pd.DataFrame:
    """End-to-end driver: download metadata CSV → filter → build paths → bucket.

    Returns the per-FOV DataFrame with scalar columns + ``parquet_shard`` +
    ``parquet_key_prefix``; the Snakemake driver writes it atomically.
    The shard index is cached so repeat runs skip the shard scan.
    """
    cache_dir = Path(cache_dir)
    meta_cache = cache_dir / "rxrx3_core_metadata"
    shard_cache = cache_dir / "rxrx3_core_shard_index"
    shard_cache.mkdir(parents=True, exist_ok=True)

    raw = _download_metadata_csv(rxrx3_cfg.hf_repo, meta_cache)
    shard_index = _build_shard_index(rxrx3_cfg.hf_repo, shard_cache)
    selected = _select_samples(raw, rxrx3_cfg.samples)
    fov = _build_fov_table(selected, rxrx3_cfg, shard_index)
    bucketed = assign_snakemake_batches(fov, batch_size)

    columns = list(_SCALAR_OUTPUT_COLUMNS)
    missing = [c for c in columns if c not in bucketed.columns]
    if missing:
        raise KeyError(f"resolved rxrx3_core metadata missing columns: {missing}")
    return bucketed[columns].reset_index(drop=True)
