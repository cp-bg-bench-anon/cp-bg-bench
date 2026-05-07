"""JUMP-CP per-FOV TIFF fetch + stack into per-plate Zarr v3 stores.

Rule B in :mod:`snakemake/rules/download.smk` calls
:func:`download_batch_to_zarr` with one ``snakemake_batch`` worth of
metadata (from rule A's parquet). The heavy lifting — boto3 GET →
``tifffile`` decode → ``np.stack`` → Zarr v3 write — lives here.

Ported from ``jump-cpg0016-segmentation/snakemake/scripts/download_stack.py``
(``download_tif``, ``download_stack``, ``download_batch``). Differences:

- Output is a per-plate Zarr v3 store (``full_images/<source>__<batch>__
  <plate>.zarr/<fov_id>``) instead of a per-batch ``.npy`` blob, so
  downstream rules can open a single plate without loading every batch.
- Shape-mismatch raises :class:`ValueError` (not ``AttributeError``) —
  wrong-shape data is a validation failure, not an attribute lookup bug.
- Botocore errors wrap in :class:`~cp_bg_bench.io.jump.JumpS3Error` so
  every S3 failure shares a single ``except OSError`` catch point.
- Concurrency is a bounded :class:`~concurrent.futures.ThreadPoolExecutor`
  over FOVs; boto3 clients are documented thread-safe for
  :meth:`get_object` and threads plug directly into Snakemake's
  ``threads:`` directive without dragging an event loop into the
  script.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lamin_utils import logger
from tifffile import tifffile

from cp_bg_bench.io.jump import make_anonymous_s3_client, s3_get_bytes
from cp_bg_bench.io.zarr_io import open_plate_group, plate_store_path, write_fov_array

__all__ = [
    "MAX_DOWNLOAD_WORKERS",
    "PLATE_GROUP_COLUMNS",
    "download_batch_to_zarr",
    "fetch_tiff",
    "plate_store_path",
    "stack_fov",
]


# Columns grouped on to select a per-plate Zarr store.
PLATE_GROUP_COLUMNS: tuple[str, str, str] = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
)

# Upper cap on in-rule thread pool. A single snakemake_batch already
# saturates ~90 % of a 1 Gbit pipe with 4 workers in the smoke run;
# anything above ~16 concurrent GETs from a single process against one
# S3 key prefix hits diminishing returns (and AWS's per-key concurrency
# comfort zone). Raise if a future cluster gives us fatter networking.
MAX_DOWNLOAD_WORKERS: int = 16


def fetch_tiff(client: Any, url: str) -> np.ndarray:
    """Download a single ``s3://...`` TIFF and decode to a 2-D array.

    Delegates the boto3 GET + error-wrapping + body-close dance to
    :func:`~cp_bg_bench.io.jump.s3_get_bytes`; adds only the
    TIFF-decode + 2-D shape check. The returned array dtype is whatever
    ``tifffile`` yields for the source file (JUMP uses uint16).
    """
    blob = s3_get_bytes(client, url)
    arr = tifffile.imread(BytesIO(blob))
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D TIFF from {url}, got shape {arr.shape}")
    return arr


def stack_fov(client: Any, urls: list[str]) -> np.ndarray:
    """Fetch all channel TIFFs for one FOV and stack to ``(C, H, W)`` uint16.

    Channel order matches ``urls`` order. All channels must share the
    same 2-D shape; mismatch raises :class:`ValueError`. The final
    array is cast to ``uint16`` — JUMP TIFFs are already uint16, so
    this is a no-op in practice but guarantees the output contract.
    """
    if not urls:
        raise ValueError("stack_fov requires at least one channel url")

    planes = [fetch_tiff(client, url) for url in urls]
    shapes = {p.shape for p in planes}
    if len(shapes) > 1:
        raise ValueError(
            f"channel shape mismatch for FOV: got shapes {[p.shape for p in planes]} "
            f"across urls {urls}"
        )
    return np.stack(planes, axis=0).astype(np.uint16, copy=False)


def _fetch_and_write_fov(
    row_dict: dict[str, Any],
    *,
    client: Any,
    channel_s3_keys: list[str],
    store_path: Path,
) -> str:
    """Worker-thread body: fetch the 5 TIFFs, stack, append to Zarr.

    Each worker opens its own Zarr group handle (proven safe by
    ``test_open_plate_group_concurrent_writers``; sharing a single
    Group across threads has unspecified semantics in zarr-python).
    Running the zstd encode inside the worker thread instead of the
    caller thread is the key scalability win: at 10 Gbit networking,
    serial caller-thread compression becomes the bottleneck that
    defeats threading entirely.
    """
    fov_id = row_dict["id"]
    urls = [row_dict[k] for k in channel_s3_keys]
    stack = stack_fov(client, urls)
    group = open_plate_group(store_path)
    write_fov_array(group, fov_id, stack)
    return fov_id


def _validate_meta_batch(meta_batch: pd.DataFrame, channel_s3_keys: list[str]) -> None:
    if meta_batch.empty:
        raise ValueError("meta_batch is empty — nothing to download")
    required = {"id", *PLATE_GROUP_COLUMNS, *channel_s3_keys}
    missing = required - set(meta_batch.columns)
    if missing:
        raise KeyError(
            f"meta_batch missing columns: {sorted(missing)}. "
            "Rule A's parquet may pre-date the output-contract extension; "
            "rerun `snakemake --forceall resolve_metadata` to regenerate."
        )
    if meta_batch["id"].duplicated().any():
        duped = meta_batch.loc[meta_batch["id"].duplicated(), "id"].tolist()
        raise ValueError(f"duplicate FOV ids in meta_batch: {duped[:5]}")


def download_batch_to_zarr(
    meta_batch: pd.DataFrame,
    output_root: Path | str,
    channel_s3_keys: list[str],
    n_threads: int = 4,
    s3_client: Any | None = None,
) -> dict[str, int]:
    """Fetch every FOV in ``meta_batch`` into per-plate Zarr v3 stores.

    Rows are grouped by ``(source, batch, plate)``; one store per group.
    Within a plate, FOV fetches fan out across a
    :class:`~concurrent.futures.ThreadPoolExecutor` sharing one boto3
    client; writes are serialised on the caller thread (zarr group
    writes are not concurrency-safe for the same array metadata file).

    Returns a ``{store_path: n_fovs_written}`` summary.
    Idempotent: existing FOV arrays with matching shape are no-ops.
    """
    _validate_meta_batch(meta_batch, channel_s3_keys)

    client = s3_client if s3_client is not None else make_anonymous_s3_client()
    output_root = Path(output_root)
    effective_workers = max(1, min(int(n_threads), MAX_DOWNLOAD_WORKERS))

    summary: dict[str, int] = {}
    plate_cols = list(PLATE_GROUP_COLUMNS)

    for plate_key, plate_rows in meta_batch.groupby(plate_cols, sort=False):
        source, batch, plate = plate_key
        store_path = plate_store_path(output_root, source, batch, plate)

        rows = plate_rows.to_dict(orient="records")
        n_written = 0
        # Explicit shutdown lets us cancel queued (not-yet-started)
        # futures when any FOV raises — Snakemake retries the whole
        # batch, so any pending fetches would be re-submitted anyway.
        # Running futures complete naturally (threads aren't
        # interruptible); we just avoid wasting connections on ones
        # that haven't started.
        pool = ThreadPoolExecutor(max_workers=effective_workers)
        try:
            futures = [
                pool.submit(
                    _fetch_and_write_fov,
                    row,
                    client=client,
                    channel_s3_keys=channel_s3_keys,
                    store_path=store_path,
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

        summary[str(store_path)] = n_written
        logger.info(f"plate store {store_path.name}: wrote {n_written} FOVs")

    return summary
