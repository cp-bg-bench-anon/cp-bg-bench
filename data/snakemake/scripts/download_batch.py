"""Rule B driver: fetch one ``snakemake_batch`` of FOVs to per-plate Zarr v3.

Runs under the Snakemake ``script:`` directive (``snakemake`` is
injected). Reads rule A's parquet, filters to this batch's rows, calls
the data-source-appropriate downloader, then touches the sentinel so
Snakemake marks the job complete.
"""

from pathlib import Path

import pandas as pd
from lamin_utils import logger

from cp_bg_bench.config import load as load_config
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821  # injected by the Snakemake runtime

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)
logger.info(f"runtime allocation: {allocation}")

cfg = load_config(snakemake_obj.input.global_config)
batch_id = snakemake_obj.wildcards.batch
parquet_path = Path(snakemake_obj.input.parquet)

df = pd.read_parquet(parquet_path)
meta_batch = df.loc[df["snakemake_batch"] == batch_id].reset_index(drop=True)
if meta_batch.empty:
    raise ValueError(
        f"snakemake_batch {batch_id!r} not found in {parquet_path} — "
        f"rule A's parquet must be stale; try `snakemake --forceall resolve_metadata`"
    )

logger.info(f"download_batch {batch_id}: {len(meta_batch)} FOVs")

ds = cfg.global_.data_source

if ds == "jump":
    from cp_bg_bench.download.tiffs import download_batch_to_zarr

    summary = download_batch_to_zarr(
        meta_batch=meta_batch,
        output_root=cfg.output_root,
        channel_s3_keys=cfg.data_source.channel_s3_keys,
        n_threads=snakemake_obj.threads,
    )
elif ds == "rxrx1":
    from cp_bg_bench.download.rxrx1_pngs import download_batch_to_zarr

    summary = download_batch_to_zarr(
        meta_batch=meta_batch,
        output_root=cfg.output_root,
        images_zip_url=cfg.data_source.images_zip_url,
        channel_zip_keys=cfg.data_source.channel_zip_keys,
        n_threads=snakemake_obj.threads,
    )
elif ds == "rxrx3_core":
    from cp_bg_bench.download.rxrx3_core_jp2 import download_batch_to_zarr

    summary = download_batch_to_zarr(
        meta_batch=meta_batch,
        output_root=cfg.output_root,
        hf_repo=cfg.data_source.hf_repo,
        channel_names=cfg.data_source.channel_names,
        n_threads=snakemake_obj.threads,
    )
else:
    raise ValueError(f"unknown data_source: {ds!r}")

total = sum(summary.values())
logger.info(f"download_batch {batch_id}: wrote {total} FOVs across {len(summary)} plates")

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(f"batch={batch_id} fovs={total} plates={len(summary)}\n")
