"""Rule K driver: derive crops_density_hf by adding corner density patches."""

import shutil
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from lamin_utils import logger

from cp_bg_bench.config import JumpConfig, Rxrx1Config, Rxrx3CoreConfig
from cp_bg_bench.config import load as load_config
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe
from cp_bg_bench.transforms.density_patch import draw_corner_patches_batch

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)

input_dir = cfg.output_root / "datasets" / "crops_hf"
output_dir = cfg.output_root / "datasets" / "crops_density_hf"

logger.info(f"derive_crops_density: {input_dir} → {output_dir}")
ds_in = load_from_disk(str(input_dir))

image_size = cfg.global_.crops.image_size
patch_cfg = cfg.global_.patching
# NOTE: extend this block when adding a new data-source adapter.
if isinstance(cfg.data_source, JumpConfig):
    n_image_channels = len(cfg.data_source.channel_s3_keys)
elif isinstance(cfg.data_source, (Rxrx1Config, Rxrx3CoreConfig)):
    n_image_channels = len(cfg.data_source.channel_names)
else:
    raise ValueError(f"Unknown data source type: {type(cfg.data_source)}")
cell_shape = (n_image_channels, image_size, image_size)


def _apply_patches_batch(batch: dict) -> dict:
    patched = draw_corner_patches_batch(
        cell_list=batch["cell"],
        intensities=batch["n_cells_scaled"],
        cell_shape=cell_shape,
        patch_size=patch_cfg.patch_size_resized,
        pad=patch_cfg.pad_resized,
    )
    return {"cell": patched}


feats = ds_in.features
ds_out = ds_in.map(
    _apply_patches_batch,
    batched=True,
    batch_size=10_000,
    num_proc=snakemake_obj.threads,
    features=feats,
    desc="derive_crops_density: corner patches",
)

if output_dir.exists():
    shutil.rmtree(output_dir)
ds_out.save_to_disk(str(output_dir))

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.write_text(f"n_rows={ds_out.num_rows}\n")
logger.info(f"derive_crops_density: done → {output_dir} ({ds_out.num_rows} rows)")
