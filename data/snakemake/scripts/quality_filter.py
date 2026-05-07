"""Rule I driver: quality filter by morphology quantile thresholds."""

import shutil
from pathlib import Path

import yaml
from datasets import load_from_disk
from lamin_utils import logger

from cp_bg_bench.config import load as load_config
from cp_bg_bench.datasets.quality_filter import compute_thresholds, filter_hf_dataset
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)
qf_cfg = cfg.global_.quality_filter

input_dir = cfg.output_root / "datasets" / "crops_unfiltered_hf"
output_dir = cfg.output_root / "datasets" / "crops_hf"
report_path = Path(snakemake_obj.output.report)
report_path.parent.mkdir(parents=True, exist_ok=True)

logger.info(f"quality_filter: loading {input_dir}")
ds_in = load_from_disk(str(input_dir))
n_before = ds_in.num_rows
logger.info(f"quality_filter: {n_before} rows before filter")

if not qf_cfg.enabled:
    logger.info("quality_filter: disabled — copying crops_unfiltered_hf → crops_hf")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(str(input_dir), str(output_dir))
    thresholds: dict = {}
    n_after = n_before
else:
    # Compute thresholds from relevant numeric columns only (zero-copy path)
    fields_present = [f for f in qf_cfg.fields if f in ds_in.column_names]
    stats_df = ds_in.select_columns(fields_present).to_pandas()
    thresholds = compute_thresholds(stats_df, qf_cfg.fields, qf_cfg.quantiles)
    del stats_df
    logger.info(f"quality_filter: thresholds = {thresholds}")

    ds_out = filter_hf_dataset(
        ds_in,
        thresholds,
        batch_size=10_000,
        num_proc=snakemake_obj.threads,
    )
    n_after = ds_out.num_rows
    logger.info(f"quality_filter: {n_after} rows after filter (dropped {n_before - n_after})")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ds_out.save_to_disk(str(output_dir))

report = {
    "enabled": qf_cfg.enabled,
    "fields": list(qf_cfg.fields),
    "quantiles": list(qf_cfg.quantiles),
    "thresholds": {k: list(v) for k, v in thresholds.items()},
    "n_before": n_before,
    "n_after": n_after,
    "n_dropped": n_before - n_after,
}
report_path.write_text(yaml.safe_dump(report, sort_keys=True))

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.write_text(f"n_after={n_after}\n")
logger.info(f"quality_filter: wrote report to {report_path}")
