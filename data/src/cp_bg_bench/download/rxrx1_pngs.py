"""Rxrx1 per-FOV PNG fetch (HTTP range requests) + stack into per-plate Zarr v3.

Images ship as a single 47 GB GCS zip. This module uses :mod:`remotezip`
to fetch individual files via HTTP ``Range:`` requests — only the ~50 KB
per PNG is transferred, not the full archive.

Concurrency model: one :class:`~concurrent.futures.ThreadPoolExecutor` per
plate. FOVs are partitioned into ``effective_workers`` chunks; each worker
opens exactly one :class:`remotezip.RemoteZip` for its entire chunk, so
the ~5 MB central-directory download is paid O(workers) times, not O(FOVs).
After the initial central-directory fetch, each per-channel ``read()`` is a
single targeted range GET.
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
    "MAX_DOWNLOAD_WORKERS",
    "PLATE_GROUP_COLUMNS",
    "download_batch_to_zarr",
    "fetch_png",
    "stack_fov",
]

PLATE_GROUP_COLUMNS: tuple[str, str, str] = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
)

# Cap matching the TIFF downloader; GCS connections saturate around 16 workers.
MAX_DOWNLOAD_WORKERS: int = 16


def fetch_png(remote_zip: Any, path_within_zip: str) -> np.ndarray:
    """Read a single PNG from an open RemoteZip and decode to a 2-D uint8 array.

    Both ``remotezip`` and ``Pillow`` are imported lazily so this module stays
    importable without the ``io`` feature env active.
    """
    from PIL import Image  # lazy: pillow is an optional io-feature dep

    blob = remote_zip.read(path_within_zip)
    return np.asarray(Image.open(BytesIO(blob)).convert("L"))


def stack_fov(remote_zip: Any, zip_paths: list[str]) -> np.ndarray:
    """Fetch all channel PNGs for one FOV and stack to ``(C, H, W)`` uint16.

    Channel order matches ``zip_paths`` order. All channels must share the
    same 2-D shape; mismatch raises :class:`ValueError`.
    """
    if not zip_paths:
        raise ValueError("stack_fov requires at least one channel path")
    planes = [fetch_png(remote_zip, p) for p in zip_paths]
    shapes = {p.shape for p in planes}
    if len(shapes) > 1:
        raise ValueError(
            f"channel shape mismatch for FOV: got shapes {[p.shape for p in planes]} "
            f"across paths {zip_paths}"
        )
    return np.stack(planes, axis=0).astype(np.uint16, copy=False)


def _process_chunk(
    rows: list[dict[str, Any]],
    *,
    images_zip_url: str,
    channel_zip_keys: list[str],
    store_path: Path,
) -> list[str]:
    """Worker-thread body: one RemoteZip per call → O(1) central-dir cost per worker.

    Each worker thread receives a pre-partitioned slice of the plate's FOVs and
    processes them sequentially under a single :class:`remotezip.RemoteZip`
    context — fetching the central directory exactly once per worker lifetime.
    """
    from remotezip import RemoteZip  # lazy: remotezip is an optional io-feature dep

    group = open_plate_group(store_path)
    written: list[str] = []
    with RemoteZip(images_zip_url) as rz:
        for row in rows:
            fov_id = row["id"]
            zip_paths = [row[k] for k in channel_zip_keys]
            stack = stack_fov(rz, zip_paths)
            write_fov_array(group, fov_id, stack)
            written.append(fov_id)
    return written


def _validate_meta_batch(meta_batch: pd.DataFrame, channel_zip_keys: list[str]) -> None:
    if meta_batch.empty:
        raise ValueError("meta_batch is empty — nothing to download")
    required = {"id", *PLATE_GROUP_COLUMNS, *channel_zip_keys}
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


def download_batch_to_zarr(
    meta_batch: pd.DataFrame,
    output_root: Path | str,
    images_zip_url: str,
    channel_zip_keys: list[str],
    n_threads: int = 4,
) -> dict[str, int]:
    """Fetch every FOV in ``meta_batch`` into per-plate Zarr v3 stores.

    Rows are grouped by ``(Metadata_Source, Metadata_Batch, Metadata_Plate)``
    — one store per group. Each worker thread opens its own
    :class:`remotezip.RemoteZip` (independent HTTP session + central-directory
    cache) and fetches its FOV's channel PNGs via HTTP range requests.

    Returns a ``{store_path: n_fovs_written}`` summary.
    Idempotent: existing FOV arrays with matching shape are no-ops.
    """
    _validate_meta_batch(meta_batch, channel_zip_keys)

    output_root = Path(output_root)
    effective_workers = max(1, min(int(n_threads), MAX_DOWNLOAD_WORKERS))

    summary: dict[str, int] = {}
    plate_cols = list(PLATE_GROUP_COLUMNS)

    for plate_key, plate_rows in meta_batch.groupby(plate_cols, sort=False):
        source, batch, plate = plate_key
        store_path = plate_store_path(output_root, source, batch, plate)

        rows = plate_rows.to_dict(orient="records")
        n_written = 0
        n_chunks = min(effective_workers, len(rows))
        chunks = [rows[i::n_chunks] for i in range(n_chunks)]

        pool = ThreadPoolExecutor(max_workers=effective_workers)
        try:
            futures = [
                pool.submit(
                    _process_chunk,
                    chunk,
                    images_zip_url=images_zip_url,
                    channel_zip_keys=channel_zip_keys,
                    store_path=store_path,
                )
                for chunk in chunks
            ]
            for fut in as_completed(futures):
                written_ids = fut.result()
                n_written += len(written_ids)
                for fov_id in written_ids:
                    logger.debug(f"wrote FOV {fov_id} to {store_path}")
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        summary[str(store_path)] = n_written
        logger.info(f"plate store {store_path.name}: wrote {n_written} FOVs")

    return summary
