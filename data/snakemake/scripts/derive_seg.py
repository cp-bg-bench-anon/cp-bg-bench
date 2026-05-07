"""Rule J driver: derive seg_hf by applying masks to fluorescent channels."""

import shutil
from pathlib import Path

from datasets import load_from_disk
from lamin_utils import logger

from cp_bg_bench.config import JumpConfig, Rxrx1Config, Rxrx3CoreConfig
from cp_bg_bench.config import load as load_config
from cp_bg_bench.crops.extract import N_MASK_CHANNELS
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe
from cp_bg_bench.transforms.masking import _apply_masks_batch as _mask_batch

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)

input_dir = cfg.output_root / "datasets" / "crops_hf"
output_dir = cfg.output_root / "datasets" / "seg_hf"

logger.info(f"derive_seg: {input_dir} → {output_dir}")
ds_in = load_from_disk(str(input_dir))

image_size = cfg.global_.crops.image_size
# NOTE: extend this block when adding a new data-source adapter.
if isinstance(cfg.data_source, JumpConfig):
    n_image_channels = len(cfg.data_source.channel_s3_keys)
elif isinstance(cfg.data_source, (Rxrx1Config, Rxrx3CoreConfig)):
    n_image_channels = len(cfg.data_source.channel_names)
else:
    raise ValueError(f"Unknown data source type: {type(cfg.data_source)}")

cell_shape = (n_image_channels, image_size, image_size)
seg_shape = (N_MASK_CHANNELS, image_size, image_size)


def _batch_fn(batch: dict) -> dict:
    return {"cell": _mask_batch(batch["cell"], batch["mask"], cell_shape, seg_shape)}


feats = ds_in.features
ds_out = ds_in.map(
    _batch_fn,
    batched=True,
    batch_size=10_000,
    num_proc=snakemake_obj.threads,
    features=feats,
    desc="derive_seg: applying masks",
)

if output_dir.exists():
    shutil.rmtree(output_dir)
ds_out.save_to_disk(str(output_dir))

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.write_text(f"n_rows={ds_out.num_rows}\n")
logger.info(f"derive_seg: done → {output_dir} ({ds_out.num_rows} rows)")
