"""Rule D driver: segment one batch of FOVs within a plate."""

from pathlib import Path

import pandas as pd
from lamin_utils import logger

from cp_bg_bench.config import load as load_config
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe
from cp_bg_bench.segmentation.cpsam import segment_plate

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)

plate_key: str = snakemake_obj.wildcards.plate_key
seg_batch_idx: int = int(snakemake_obj.wildcards.seg_batch_idx)
source, batch, plate = plate_key.split("__", 2)

batch_size = cfg.global_.compute.gpu.seg_fov_batch_size

meta_df = pd.read_parquet(
    snakemake_obj.input.metadata,
    filters=[
        ("Metadata_Source", "=", source),
        ("Metadata_Batch", "=", batch),
        ("Metadata_Plate", "=", plate),
    ],
    columns=["id"],
)
all_fov_ids = sorted(meta_df["id"].tolist())
start = seg_batch_idx * batch_size
batch_fov_ids = all_fov_ids[start : start + batch_size]

if not batch_fov_ids:
    raise ValueError(
        f"seg_batch_idx={seg_batch_idx} out of range for plate_key={plate_key} "
        f"({len(all_fov_ids)} FOVs, batch_size={batch_size})"
    )

use_gpu = len(allocation.visible_gpu_ids) > 0
logger.info(
    f"segment_fov_batch: plate={plate_key} batch={seg_batch_idx} "
    f"FOVs={batch_fov_ids[0]}..{batch_fov_ids[-1]} ({len(batch_fov_ids)}) GPU={use_gpu}"
)

cell_df = segment_plate(
    source=source,
    batch=batch,
    plate=plate,
    output_root=cfg.output_root,
    seg_config=cfg.data_source.segmentation,
    use_gpu=use_gpu,
    fov_ids=batch_fov_ids,
)

parquet_path = Path(snakemake_obj.output.parquet)
parquet_path.parent.mkdir(parents=True, exist_ok=True)
tmp_parquet = parquet_path.with_suffix(".tmp.parquet")
cell_df.to_parquet(tmp_parquet, index=False)
tmp_parquet.rename(parquet_path)

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(f"plate_key={plate_key}\nseg_batch_idx={seg_batch_idx}\nn_cells={len(cell_df)}\n")
logger.info(f"segment_fov_batch: wrote {len(cell_df)} cells to {parquet_path}")
