"""Rule C driver: calibrate Cellpose-SAM diameter estimates per source.

Reads rule A's parquet, samples FOVs per source, runs Cellpose-SAM's
built-in SizeModel on the nucleus and cytosol channel stacks, and writes
a three-file calibration report to ``<RESULTS>/calibration/``.
"""

from pathlib import Path

import pandas as pd
from lamin_utils import logger

from cp_bg_bench.calibrate.diameters import run_calibration
from cp_bg_bench.config import load as load_config
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821  # injected by the Snakemake runtime

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)
logger.info(f"runtime allocation: {allocation}")

cfg = load_config(snakemake_obj.input.global_config)
meta_df = pd.read_parquet(Path(snakemake_obj.input.parquet))

use_gpu = len(allocation.visible_gpu_ids) > 0
logger.info(f"calibrate: GPU={"yes" if use_gpu else "no (CPU}")")

config_hash, output_dir = run_calibration(
    meta_df=meta_df,
    output_root=cfg.output_root,
    seg_config=cfg.data_source.segmentation,
    cal_config=cfg.global_.calibrate,
    data_source_stem=cfg.data_source_stem,
    use_gpu=use_gpu,
)

logger.info(f"calibration complete: {output_dir}/{config_hash}.*")

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(f"config_hash={config_hash}\n")
