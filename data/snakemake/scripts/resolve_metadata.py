"""Rule A driver: download metadata, filter, expand to FOVs, write parquet.

Runs under the Snakemake ``script:`` directive, which injects a module-level
``snakemake`` object exposing ``.input``, ``.output``, ``.log``, etc. Note:
Snakemake wraps this file with a preamble at execution time, so no
``from __future__`` import is permitted here.
"""

from pathlib import Path

from lamin_utils import logger

from cp_bg_bench.config import load as load_config
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821  # injected by the Snakemake runtime

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)
logger.info(f"runtime allocation: {allocation}")

cfg = load_config(snakemake_obj.input.global_config)
logger.info(
    f"loaded config from {cfg.source_path} "
    f"(data_source_stem={cfg.data_source_stem}, fingerprint={cfg.fingerprint()[:12]})"
)

cache_dir = cfg.output_root / "meta" / "cache"
ds = cfg.global_.data_source

if ds == "jump":
    from cp_bg_bench.io.jump import resolve_metadata

    df = resolve_metadata(
        jump_cfg=cfg.data_source,
        cache_dir=cache_dir,
        batch_size=cfg.global_.sharding.snakemake_batch_size,
    )
elif ds == "rxrx1":
    from cp_bg_bench.io.rxrx1 import resolve_metadata

    df = resolve_metadata(
        rxrx1_cfg=cfg.data_source,
        cache_dir=cache_dir,
        batch_size=cfg.global_.sharding.snakemake_batch_size,
    )
elif ds == "rxrx3_core":
    from cp_bg_bench.io.rxrx3_core import resolve_metadata

    df = resolve_metadata(
        rxrx3_cfg=cfg.data_source,
        cache_dir=cache_dir,
        batch_size=cfg.global_.sharding.snakemake_batch_size,
    )
else:
    raise ValueError(f"unknown data_source: {ds!r}")

max_fovs = cfg.global_.max_fovs_per_plate
if max_fovs is not None:
    df = (
        df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"], sort=False)
        .head(max_fovs)
        .reset_index(drop=True)
    )
    logger.info(f"resolve_metadata: capped to {max_fovs} FOVs/plate → {len(df):,} rows")

max_fovs_well = cfg.global_.max_fovs_per_well
if max_fovs_well is not None:
    # Metadata_Well is not persisted as a scalar column — parse from the composite id
    # which has the format {source}__{batch}__{plate}__{well}__{site}.
    df = df.copy()
    df["_well"] = df["id"].str.split("__", n=4).str[3]
    df = (
        df.groupby(
            ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "_well"],
            sort=False,
        )
        .head(max_fovs_well)
        .reset_index(drop=True)
        .drop(columns=["_well"])
    )
    logger.info(f"resolve_metadata: capped to {max_fovs_well} FOVs/well → {len(df):,} rows")

out = Path(snakemake_obj.output.parquet)
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(out.suffix + ".tmp")
df.to_parquet(tmp, index=False)
tmp.replace(out)

logger.info(f"wrote {out} ({len(df):,} rows)")
