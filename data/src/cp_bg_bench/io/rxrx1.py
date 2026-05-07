"""Rxrx1 metadata resolution and image path helpers.

Rxrx1 is the first publicly released Recursion Cell Painting dataset
(https://www.rxrx.ai/rxrx1). Images ship as a bulk GCS zip; metadata ships
as a separate smaller zip containing a single ``metadata.csv``.

Public data layout
------------------
Metadata zip (``cfg.metadata_url``):
    Archive contains ``rxrx1/metadata.csv`` with columns:
    site_id, well_id, cell_type, dataset, experiment, plate, well,
    site, well_type, sirna, sirna_id.

Images zip (``cfg.images_zip_url``):
    47 GB archive. Path within zip:
    ``rxrx1/images/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png``
    where ``channel`` is 1-indexed (w1 = Hoechst, …, w6 = WGA).
    Individual files fetched via HTTP range requests (``remotezip``).

Column mapping (metadata CSV → pipeline schema)
------------------------------------------------
    experiment  → Metadata_Source
    plate       → Metadata_Plate
    well        → Metadata_Well   (used during construction; dropped in output)
    site        → Metadata_Site
    well_type   → Metadata_PlateType
    sirna       → Metadata_InChIKey  (siRNA ThermoFisher ID)
    sirna_id    → Metadata_InChI     (numeric siRNA index)
    (fixed)     → Metadata_Batch = "rxrx1"
    id          = "{experiment}__rxrx1__{plate}__{well}__{site}"
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from lamin_utils import logger

if TYPE_CHECKING:
    from cp_bg_bench.config import Rxrx1Config, Rxrx1Sample

__all__ = [
    "assign_snakemake_batches",
    "resolve_metadata",
]

# Fixed batch label — Rxrx1 ships as a monolithic release with no batch split.
_RXRX1_BATCH = "rxrx1"

# CSV member path within the metadata zip.
_METADATA_CSV_MEMBER = "rxrx1/metadata.csv"

# Path prefix for all image members within the images zip.
_ZIP_IMAGE_PREFIX = "rxrx1/images/"

# Map metadata CSV columns → shared pipeline schema.
_RXRX1_TO_PIPELINE: dict[str, str] = {
    "experiment": "Metadata_Source",
    "plate": "Metadata_Plate",
    "well": "Metadata_Well",
    "site": "Metadata_Site",
    "well_type": "Metadata_PlateType",
    "sirna": "Metadata_InChIKey",
    "sirna_id": "Metadata_InChI",
}

# Identifier columns that must be path-safe (same constraint as JUMP).
_SAFE_METADATA_COLUMNS: tuple[str, ...] = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
)
_SAFE_METADATA_VALUE = re.compile(r"^[A-Za-z0-9._-]+$")

# Scalar columns in the final parquet — mirrors JUMP's _SCALAR_OUTPUT_COLUMNS.
_SCALAR_OUTPUT_COLUMNS: tuple[str, ...] = (
    "id",
    "snakemake_batch",
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_PlateType",
    "Metadata_InChIKey",
    "Metadata_InChI",
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


def _download_metadata_csv(metadata_url: str, cache_dir: Path) -> pd.DataFrame:
    """Download the metadata zip and extract ``metadata.csv``; cache both locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_zip = cache_dir / "rxrx1-metadata.zip"
    local_csv = cache_dir / "metadata.csv"

    if not local_csv.is_file():
        if not local_zip.is_file():
            logger.info(f"downloading rxrx1 metadata from {metadata_url}")
            tmp = local_zip.with_suffix(".zip.tmp")
            with urllib.request.urlopen(metadata_url, timeout=30) as resp:
                tmp.write_bytes(resp.read())
            tmp.replace(local_zip)
        logger.info("extracting rxrx1/metadata.csv from zip")
        with zipfile.ZipFile(local_zip) as z, z.open(_METADATA_CSV_MEMBER) as src:
            local_csv.write_bytes(src.read())
    else:
        logger.debug(f"rxrx1 metadata CSV cache hit: {local_csv}")

    return pd.read_csv(local_csv)


def _select_samples(meta: pd.DataFrame, samples: list[Rxrx1Sample]) -> pd.DataFrame:
    """Apply the whitelist: AND within a sample entry, OR across entries."""
    selected_rows: list[pd.DataFrame] = []
    seen_any_nonempty = False

    for sample in samples:
        mask = pd.Series(True, index=meta.index)
        if sample.experiment is not None:
            mask &= meta["experiment"] == sample.experiment
        if sample.plate is not None:
            mask &= meta["plate"].astype(str) == str(sample.plate)
        if sample.well is not None:
            mask &= meta["well"] == sample.well
        if sample.sirna is not None:
            mask &= meta["sirna"] == sample.sirna

        matched = meta.loc[mask]
        if matched.empty:
            logger.warning(
                f"rxrx1 whitelist sample matched 0 rows: "
                f"experiment={sample.experiment!r}, plate={sample.plate!r}"
            )
        else:
            selected_rows.append(matched)
            seen_any_nonempty = True

    if not seen_any_nonempty:
        raise ValueError("rxrx1 whitelist produced an empty selection; check the samples block")

    combined = pd.concat(selected_rows, axis=0)
    return combined.drop_duplicates(subset=["experiment", "plate", "well", "site"]).reset_index(
        drop=True
    )


def _build_fov_table(meta: pd.DataFrame, rxrx1_cfg: Rxrx1Config) -> pd.DataFrame:
    """Rename columns, add derived fields, and build per-channel zip path columns."""
    fov = meta.rename(columns=_RXRX1_TO_PIPELINE)
    fov["Metadata_Batch"] = _RXRX1_BATCH

    # Cast identifier columns to str for id composition and path safety checks.
    for col in ("Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_Site"):
        if col in fov.columns:
            fov[col] = fov[col].astype(str)

    _validate_safe_metadata(fov)

    fov["id"] = (
        fov["Metadata_Source"]
        + "__"
        + fov["Metadata_Batch"]
        + "__"
        + fov["Metadata_Plate"]
        + "__"
        + fov["Metadata_Well"]
        + "__"
        + fov["Metadata_Site"]
    )

    # Build zip paths: rxrx1/images/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png
    # Vectorised string concatenation — no per-row Python overhead.
    prefix = (
        _ZIP_IMAGE_PREFIX
        + fov["Metadata_Source"]
        + "/Plate"
        + fov["Metadata_Plate"]
        + "/"
        + fov["Metadata_Well"]
        + "_s"
        + fov["Metadata_Site"]
        + "_w"
    )
    for channel_idx, key in enumerate(rxrx1_cfg.channel_zip_keys, start=1):
        fov[key] = prefix + str(channel_idx) + ".png"

    return fov


def assign_snakemake_batches(fov: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Assign ``snakemake_batch`` ids — same contract as the JUMP equivalent."""
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
    rxrx1_cfg: Rxrx1Config,
    cache_dir: Path,
    batch_size: int,
) -> pd.DataFrame:
    """End-to-end driver: download metadata zip → filter → build paths → bucket.

    Returns the final per-FOV DataFrame (scalar columns + per-channel zip-path
    columns); the Snakemake driver writes it atomically.
    """
    cache_dir = Path(cache_dir)
    metadata_dir = cache_dir / "rxrx1_metadata"

    raw = _download_metadata_csv(rxrx1_cfg.metadata_url, metadata_dir)
    selected = _select_samples(raw, rxrx1_cfg.samples)
    fov = _build_fov_table(selected, rxrx1_cfg)
    bucketed = assign_snakemake_batches(fov, batch_size)

    channel_zip_keys = rxrx1_cfg.channel_zip_keys
    columns = [*_SCALAR_OUTPUT_COLUMNS, *channel_zip_keys]
    missing = [c for c in columns if c not in bucketed.columns]
    if missing:
        raise KeyError(f"resolved rxrx1 metadata missing columns: {missing}")
    return bucketed[columns].reset_index(drop=True)
