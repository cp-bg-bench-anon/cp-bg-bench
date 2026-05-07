"""Rule M driver: reshard one of the four dataset variants."""

from pathlib import Path

from lamin_utils import logger

from cp_bg_bench.config import load as load_config
from cp_bg_bench.datasets.hf import reshard_dataset
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)

variant: str = snakemake_obj.wildcards.variant
sharding_cfg = cfg.global_.sharding

input_dir = cfg.output_root / "datasets" / f"{variant}_hf"
output_dir = cfg.output_root / "datasets" / f"{variant}_resharded"

logger.info(f"reshard_variant: {input_dir} → {output_dir}")

reshard_dataset(
    input_hf_dir=input_dir,
    output_hf_dir=output_dir,
    rows_per_shard=sharding_cfg.hf_rows_per_shard,
    max_shards=sharding_cfg.hf_max_shards,
)

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(f"variant={variant}\noutput={output_dir}\n")
logger.info(f"reshard_variant: done → {output_dir}")
