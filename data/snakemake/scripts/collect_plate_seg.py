"""Rule D2 driver: merge per-batch seg parquets into the plate-level parquet."""

from pathlib import Path

import pandas as pd
from lamin_utils import logger

from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

plate_key: str = snakemake_obj.wildcards.plate_key

batch_parquets = sorted(
    Path(s).with_suffix(".parquet") for s in snakemake_obj.input.batch_sentinels
)

cell_df = pd.concat([pd.read_parquet(p) for p in batch_parquets], ignore_index=True)

parquet_path = Path(snakemake_obj.output.parquet)
parquet_path.parent.mkdir(parents=True, exist_ok=True)
tmp = parquet_path.with_suffix(".tmp.parquet")
cell_df.to_parquet(tmp, index=False)
tmp.rename(parquet_path)

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(f"plate_key={plate_key}\nn_cells={len(cell_df)}\n")
logger.info(f"collect_plate_seg: {plate_key} — {len(cell_df)} total cells from {len(batch_parquets)} batches")
