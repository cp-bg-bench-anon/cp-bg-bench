"""Rule E driver: cell selection across all plates."""

import json
from pathlib import Path

from lamin_utils import logger

from cp_bg_bench.config import load as load_config
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe
from cp_bg_bench.selection import select_cells

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)
logger.info(f"runtime allocation: {allocation}")

cfg = load_config(snakemake_obj.input.global_config)

sel_cfg = cfg.global_.selection
selected_df, scaling_stats = select_cells(
    seg_parquet_dir=cfg.output_root / "segmentation",
    meta_parquet=snakemake_obj.input.meta_parquet,
    strategy=sel_cfg.strategy,
    seed=sel_cfg.random_seed,
    cells_per_well=sel_cfg.cells_per_well,
    max_cells=sel_cfg.max_cells,
    target_cells=sel_cfg.target_cells,
    control_labels=sel_cfg.control_labels,
)

parquet_path = Path(snakemake_obj.output.parquet)
parquet_path.parent.mkdir(parents=True, exist_ok=True)
tmp_parquet = parquet_path.with_suffix(".tmp.parquet")
selected_df.to_parquet(tmp_parquet, index=False)
tmp_parquet.rename(parquet_path)

# Persist scaling stats alongside the parquet for audit
stats_path = parquet_path.parent / "scaling_stats.json"
stats_path.write_text(json.dumps(scaling_stats, indent=2))

logger.info(f"select_cells: wrote {len(selected_df)} rows to {parquet_path}")

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(
    f"n_cells={len(selected_df)}\n"
    f"strategy={sel_cfg.strategy}\n"
    f"n_min={scaling_stats['n_min']}\n"
    f"n_max={scaling_stats['n_max']}\n"
)
