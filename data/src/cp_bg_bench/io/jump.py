"""JUMP-CP (cpg0016) metadata resolution.

Rule A in :mod:`snakemake/rules/resolve_metadata.smk` composes the functions
defined here to go from a :class:`~cp_bg_bench.config.JumpConfig` whitelist
to a per-FOV parquet of S3 image URLs + compound annotation. The two
external surfaces are:

- https://github.com/jump-cellpainting/datasets/raw/main/metadata/*.csv.gz
  (plate / well / compound / orf)
- s3://cellpainting-gallery/cpg0016-jump/<source>/workspace/load_data_csv/
  <batch>/<plate>/load_data_with_illum.parquet  (per-plate FOV listing)

Ported from ``jump-cpg0016-segmentation/snakemake/scripts/dl.py`` (``JumpMeta``,
``JumpDl``, ``generate_batch_ids``), with the home-dir defaults dropped and
pandarallel / multiprocessing replaced by vectorised pandas on a serial
boto3 client. At the scales we hit (≤ tens of plates per run) serial S3 is
fine; parallelism belongs at the rule level via Snakemake, not inside a
single rule body.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from lamin_utils import logger

if TYPE_CHECKING:
    from cp_bg_bench.config import JumpConfig, JumpSample

__all__ = [
    "ANNOTATION_TABLES",
    "LOAD_DATA_TEMPLATE",
    "JumpS3Error",
    "assign_snakemake_batches",
    "download_metadata_tables",
    "expand_to_fovs",
    "fetch_load_data_per_plate",
    "load_metadata",
    "make_anonymous_s3_client",
    "parse_s3_url",
    "required_output_columns",
    "resolve_metadata",
    "s3_get_bytes",
    "select_samples",
]


class JumpS3Error(OSError):
    """Raised for any failed JUMP S3 fetch (botocore error or non-200 status)."""


# botocore exception types we translate to JumpS3Error. Imported lazily so
# `cp_bg_bench.io.jump` stays importable without the optional io feature env.
# If botocore isn't installed (test envs that never touch S3), the tuple is
# empty — programmer errors like TypeError then surface instead of being
# silently repackaged as JumpS3Error.
try:
    from botocore.exceptions import BotoCoreError as _BotoCoreError
    from botocore.exceptions import ClientError as _ClientError

    _S3_ERROR_TYPES: tuple[type[BaseException], ...] = (_BotoCoreError, _ClientError)
except ImportError:
    _S3_ERROR_TYPES = ()


# JUMP-CP stores per-plate FOV listings as CSV at this path. The reference
# pipeline predates a parquet migration that never fully landed — every
# source we've observed ships `.csv` here, so we parse CSV unconditionally.
LOAD_DATA_TEMPLATE = (
    "s3://cellpainting-gallery/cpg0016-jump/"
    "{source}/workspace/load_data_csv/{batch}/{plate}/load_data_with_illum.csv"
)

# Map JumpSample pydantic field names → column names in the merged metadata.
_SAMPLE_FIELD_TO_COLUMN: dict[str, str] = {
    "metadata_source": "Metadata_Source",
    "metadata_batch": "Metadata_Batch",
    "metadata_plate": "Metadata_Plate",
    "metadata_well": "Metadata_Well",
    "metadata_inchikey": "Metadata_InChIKey",
    "metadata_jcp2022": "Metadata_JCP2022",
}

# Columns that uniquely identify a well after plate⨝well.
_WELL_KEY: tuple[str, ...] = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
)

# Metadata identifiers flow into filesystem paths (per-plate Zarr store
# directory, FOV array name) and into the composite ``id`` column used
# as a key across the pipeline. Restrict them to characters that are
# both path-safe and round-trip cleanly through the ``__``-delimited id
# format. JUMP metadata already conforms; this guard fail-closes against
# future data sources (rxrx1, user-supplied configs) that might ship
# values like ``../foo`` or ``a/b``.
_SAFE_METADATA_VALUE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_METADATA_COLUMNS: tuple[str, ...] = (*_WELL_KEY, "Metadata_Site")


def _validate_safe_metadata(fov: pd.DataFrame) -> None:
    """Raise if any identifier column contains filesystem-unsafe characters.

    Empty strings and values with ``/``, whitespace, NUL, or other
    specials are rejected — they'd either escape the output directory
    tree or collide with the ``__`` id separator.
    """
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


# Annotation tables merged into per-well metadata (if present in config).
# ``compound`` is the canonical annotation for JUMP-CP compound plates;
# ``orf`` / ``crispr`` cover the ORF and CRISPR reagent plates respectively.
# Preference order matters when the same Metadata_JCP2022 appears in multiple
# tables — the first wins.
ANNOTATION_TABLES: tuple[str, ...] = ("compound", "orf", "crispr")

# Scalar non-channel columns that must appear in the final parquet.
# ``Metadata_Batch`` / ``Metadata_Plate`` are carried through so rule B
# can group FOVs into per-plate Zarr stores without re-parsing ``id``.
_SCALAR_OUTPUT_COLUMNS: tuple[str, ...] = (
    "id",
    "snakemake_batch",
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_PlateType",
    "Metadata_InChIKey",
    "Metadata_InChI",
    "Metadata_SMILES",
)


def required_output_columns(channel_s3_keys: Iterable[str]) -> list[str]:
    """Expected column order of the final ``selected_metadata.parquet``."""
    return [*_SCALAR_OUTPUT_COLUMNS, *channel_s3_keys]


def make_anonymous_s3_client() -> Any:
    """Anonymous S3 client for the public ``cellpainting-gallery`` bucket.

    Imports ``boto3`` lazily so :mod:`cp_bg_bench.io.jump` can be imported
    without the optional ``io`` feature env being active (e.g. during
    unit tests that mock S3 entirely).
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def parse_s3_url(url: str) -> tuple[str, str]:
    """Split ``s3://bucket/key...`` into ``(bucket, key)``. Validates scheme."""
    if not url.startswith("s3://"):
        raise ValueError(f"not an s3 url: {url!r}")
    rest = url[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 url: {url!r}")
    return bucket, key


def s3_get_bytes(client: Any, url: str) -> bytes:
    """Anonymous ``GetObject`` against an ``s3://...`` URL, returning raw bytes.

    Centralises the "parse url → get_object → check status → close body"
    dance so the CSV fetcher (rule A) and the TIFF fetcher (rule B)
    both share one error-wrapping code path. Botocore errors and
    non-200 statuses raise :class:`JumpS3Error`; programmer errors
    (e.g., a misspelled kwarg) are not caught — they surface directly
    so they don't masquerade as network failures.
    """
    bucket, key = parse_s3_url(url)
    logger.debug(f"fetching s3://{bucket}/{key}")
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except _S3_ERROR_TYPES as exc:
        raise JumpS3Error(f"failed to fetch s3://{bucket}/{key}: {exc}") from exc

    status = response["ResponseMetadata"]["HTTPStatusCode"]
    if status != 200:
        raise JumpS3Error(f"unexpected HTTP {status} for s3://{bucket}/{key}")

    with closing(response["Body"]) as body:
        return body.read()


def download_metadata_tables(urls: dict[str, str], dest_dir: Path) -> dict[str, Path]:
    """Download the JUMP global metadata CSV files into ``dest_dir``.

    One file per ``(name, url)`` pair. Existing files are reused — callers
    wanting a fresh pull should remove the cache dir first.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in urls.items():
        local = dest_dir / Path(url).name
        if local.is_file():
            logger.debug(f"metadata cache hit: {local}")
        else:
            logger.info(f"downloading {name} metadata from {url}")
            tmp = local.with_suffix(local.suffix + ".tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(local)
        paths[name] = local
    return paths


def load_metadata(table_paths: dict[str, Path]) -> pd.DataFrame:
    """Merge plate + well with compound / orf / crispr annotations.

    Plate and well join structurally; the annotation tables are concatenated
    on ``Metadata_JCP2022`` and left-joined. This keeps ORF and CRISPR wells
    in the result (with NaN-filled compound columns) instead of the
    inner-join behaviour that silently dropped them before.
    """
    for required in ("plate", "well", "compound"):
        if required not in table_paths:
            raise ValueError(f"missing metadata table {required!r}")

    plate = pd.read_csv(table_paths["plate"])
    well = pd.read_csv(table_paths["well"])
    merged = plate.merge(well, on=["Metadata_Source", "Metadata_Plate"])

    annotation_frames: list[pd.DataFrame] = [
        pd.read_csv(table_paths[name]) for name in ANNOTATION_TABLES if name in table_paths
    ]
    if annotation_frames:
        if len(annotation_frames) == 1:
            annotations = annotation_frames[0]
        else:
            # ANNOTATION_TABLES order decides the winner on overlapping JCP2022s.
            annotations = pd.concat(annotation_frames, axis=0, ignore_index=True)
            annotations = annotations.drop_duplicates(subset=["Metadata_JCP2022"], keep="first")
        merged = merged.merge(annotations, on="Metadata_JCP2022", how="left")

    return merged.reset_index(drop=True)


def select_samples(meta: pd.DataFrame, samples: Iterable[JumpSample]) -> pd.DataFrame:
    """Apply the whitelist: AND within a ``JumpSample``, OR across entries.

    Any entry whose fields are all ``None`` selects the full table (logged at
    info level — that's a valid intent, not a typo). Entries matching no rows
    emit a warning so real typos are surfaced.

    The returned rows are deduplicated on the well key
    ``(Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well)``,
    which is the natural FOV-uniqueness contract — the annotation columns
    can drift without changing well identity.
    """
    selected_rows: list[pd.DataFrame] = []
    seen_any_nonempty = False

    for sample in samples:
        filters: dict[str, Any] = {}
        for field, column in _SAMPLE_FIELD_TO_COLUMN.items():
            value = getattr(sample, field)
            if value is not None:
                filters[column] = value

        if not filters:
            logger.info("whitelist sample has no filter fields; selecting all rows")
            selected_rows.append(meta)
            seen_any_nonempty = True
            continue

        mask = pd.Series(True, index=meta.index)
        for column, value in filters.items():
            if column not in meta.columns:
                raise KeyError(f"column {column!r} not in merged metadata")
            mask &= meta[column] == value

        matched = meta.loc[mask]
        if matched.empty:
            logger.warning(f"whitelist sample matched 0 rows: {filters}")
        else:
            selected_rows.append(matched)
            seen_any_nonempty = True

    if not seen_any_nonempty:
        raise ValueError("whitelist produced an empty selection; check the samples block")

    combined = pd.concat(selected_rows, axis=0)
    return combined.drop_duplicates(subset=list(_WELL_KEY)).reset_index(drop=True)


def _unique_plate_triples(meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]
    return meta[cols].drop_duplicates().reset_index(drop=True)


def _fetch_load_data_csv(client: Any, url: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.is_file():
        logger.debug(f"load_data cache hit: {cache_path}")
        return pd.read_csv(cache_path)

    body = s3_get_bytes(client, url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(cache_path)
    return pd.read_csv(cache_path)


def fetch_load_data_per_plate(
    meta: pd.DataFrame,
    cache_dir: Path,
    s3_client: Any | None = None,
) -> pd.DataFrame:
    """Fetch per-plate ``load_data_with_illum.csv`` from S3 and concatenate.

    ``cache_dir`` holds a local copy of each plate's CSV so reruns don't
    re-download. Client is optional: callers can inject a mock for tests.
    """
    client = s3_client if s3_client is not None else make_anonymous_s3_client()
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    triples = _unique_plate_triples(meta)
    for row in triples.itertuples(index=False):
        source = row.Metadata_Source
        batch = row.Metadata_Batch
        plate = row.Metadata_Plate
        url = LOAD_DATA_TEMPLATE.format(source=source, batch=batch, plate=plate)
        cache_path = cache_dir / f"{source}__{batch}__{plate}.csv"
        frames.append(_fetch_load_data_csv(client, url, cache_path))

    if not frames:
        raise ValueError("no plates to fetch load_data for")
    return pd.concat(frames, axis=0, ignore_index=True)


def expand_to_fovs(
    selected: pd.DataFrame,
    load_data: pd.DataFrame,
    channel_s3_keys: list[str],
) -> pd.DataFrame:
    """Inner-join per-well metadata with per-FOV ``load_data``.

    After the join the DataFrame is per-FOV (one row per Metadata_Site).
    Adds:

    - ``id`` = ``"{source}__{batch}__{plate}__{well}__{site}"`` — stable key
      used throughout the rest of the pipeline.
    - One ``s3_Orig{channel}`` column per entry in ``channel_s3_keys``,
      copied from the corresponding ``URL_Orig{channel}`` column that JUMP
      ships in ``load_data_with_illum.csv``.
    """
    join_cols = list(_WELL_KEY)
    # JUMP plate IDs are all-digit strings in plate.csv.gz but are read as
    # int64 from load_data_with_illum.csv — coerce both sides to str so the
    # merge doesn't blow up with a dtype-mismatch ValueError.
    selected = selected.astype({c: "string" for c in join_cols})
    load_data = load_data.astype({c: "string" for c in join_cols})
    fov = selected.merge(load_data, how="inner", on=join_cols)

    if fov.empty:
        raise ValueError("expand_to_fovs produced 0 rows — no FOVs matched the whitelist")

    missing_site = "Metadata_Site" not in fov.columns
    if missing_site:
        raise KeyError("load_data has no Metadata_Site column — cannot expand to FOVs")

    _validate_safe_metadata(fov)

    # Ensure Site is treated as a string so ids round-trip cleanly.
    site = fov["Metadata_Site"].astype(str)
    fov["id"] = (
        fov["Metadata_Source"].astype(str)
        + "__"
        + fov["Metadata_Batch"].astype(str)
        + "__"
        + fov["Metadata_Plate"].astype(str)
        + "__"
        + fov["Metadata_Well"].astype(str)
        + "__"
        + site
    )

    for key in channel_s3_keys:
        if not key.startswith("s3_"):
            raise ValueError(f"channel_s3_keys entries must start with 's3_', got {key!r}")
        channel = key.removeprefix("s3_")  # e.g. OrigDNA
        url_col = f"URL_{channel}"
        if url_col not in fov.columns:
            raise KeyError(f"load_data missing {url_col!r} — cannot build {key!r}")
        fov[key] = fov[url_col].astype(str)

    return fov


def assign_snakemake_batches(fov: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Assign each FOV a ``snakemake_batch`` id, reset per ``Metadata_Source``.

    IDs are formatted ``{source}_snakemake_batch_{n}`` where ``n`` starts at
    0 and increments every ``batch_size`` rows within each source. Row order
    within the bucket is preserved (stable).
    """
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
    jump_cfg: JumpConfig,
    cache_dir: Path,
    batch_size: int,
    s3_client: Any | None = None,
) -> pd.DataFrame:
    """End-to-end driver: download → merge → filter → expand → bucket.

    Returns the final DataFrame (columns = ``REQUIRED_OUTPUT_COLUMNS``); the
    Snakemake driver is responsible for writing it atomically.
    """
    cache_dir = Path(cache_dir)
    metadata_dir = cache_dir / "jump_metadata"
    load_data_dir = cache_dir / "load_data"

    table_paths = download_metadata_tables(jump_cfg.metadata_tables, metadata_dir)
    merged = load_metadata(table_paths)
    selected = select_samples(merged, jump_cfg.samples)
    load_data = fetch_load_data_per_plate(selected, load_data_dir, s3_client=s3_client)
    fov = expand_to_fovs(selected, load_data, jump_cfg.channel_s3_keys)
    bucketed = assign_snakemake_batches(fov, batch_size)

    columns = required_output_columns(jump_cfg.channel_s3_keys)
    missing = [c for c in columns if c not in bucketed.columns]
    if missing:
        raise KeyError(f"resolved metadata missing columns: {missing}")
    return bucketed[columns].reset_index(drop=True)
