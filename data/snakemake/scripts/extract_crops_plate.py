"""Rule F driver: per-plate crop extraction with in-memory resize."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lamin_utils import logger

from cp_bg_bench.config import JumpConfig, Rxrx1Config, Rxrx3CoreConfig
from cp_bg_bench.config import load as load_config
from cp_bg_bench.crops.extract import N_MASK_CHANNELS, extract_plate_crops
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe
from cp_bg_bench.transforms.resize import resize_batch

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)

plate_key: str = snakemake_obj.wildcards.plate_key
source, pipeline_batch, plate = plate_key.split("__", 2)

logger.info(f"extract_crops_plate: plate_key={plate_key}")

# Load only the rows for this plate via pyarrow pushdown filter
plate_df = pd.read_parquet(
    cfg.output_root / "selection" / "selected_cells.parquet",
    filters=[
        ("source", "=", source),
        ("batch", "=", pipeline_batch),
        ("plate", "=", plate),
    ],
).reset_index(drop=True)

logger.info(f"extract_crops_plate: {len(plate_df)} cells selected for plate_key={plate_key}")

# Derive canonical columns: perturbation, batch (split key), treatment.
# Detection via Metadata_Batch == pipeline_batch from the plate_key.
plate_df = plate_df.copy()

if pipeline_batch == "rxrx1":
    # Map siRNA ThermoFisher IDs (Metadata_InChIKey) → gene symbols.
    # Static CSV lives alongside other repo meta files.
    _meta_dir = Path(__file__).parent.parent.parent / "data" / "meta"
    sirna_csv = _meta_dir / "rxrx1_sirna_to_gene.csv"
    if not sirna_csv.is_file():
        raise FileNotFoundError(f"rxrx1_sirna_to_gene.csv not found at {sirna_csv}")
    sirna_df = pd.read_csv(sirna_csv, dtype={"sirna_id": str, "gene_symbol": str})

    _sirna_to_gene: dict[str, str] = {}
    for _, _r in sirna_df.iterrows():
        _sid = str(_r["sirna_id"])
        _gene = str(_r["gene_symbol"]) if pd.notna(_r["gene_symbol"]) else ""
        if _gene == "NEGATIVE_CONTROL":
            _sirna_to_gene[_sid] = "EMPTY"
        elif _gene:
            _sirna_to_gene[_sid] = _gene
        # UNRESOLVED (empty gene_symbol) → not added → rows dropped below

    def _resolve(sirna_id: str) -> str | None:
        if sirna_id == "EMPTY":
            return "EMPTY"
        return _sirna_to_gene.get(sirna_id, None)

    plate_df["perturbation"] = plate_df["Metadata_InChIKey"].map(_resolve)

    n_unresolved = plate_df["perturbation"].isna().sum()
    if n_unresolved:
        logger.warning(
            f"extract_crops_plate: dropping {n_unresolved} UNRESOLVED siRNA rows "
            f"for plate_key={plate_key}"
        )
        plate_df = plate_df[plate_df["perturbation"].notna()].reset_index(drop=True)

    plate_df["treatment"] = plate_df["Metadata_InChIKey"]
    # RxRx1 imaging batch = experiment (HEPG2-01..11), stored in `source`.
    # Plates 1-4 are replicates within an experiment, not independent batches.
    plate_df["batch"] = plate_df["source"]

elif pipeline_batch == "rxrx3_core":
    # Metadata_InChIKey = gene symbol; normalise control label.
    plate_df["perturbation"] = plate_df["Metadata_InChIKey"].replace("EMPTY_control", "EMPTY")
    plate_df["treatment"] = plate_df["Metadata_InChI"]
    plate_df["batch"] = f"plate_{plate}"

else:
    # JUMP-CP: Metadata_InChIKey is the canonical compound identifier.
    plate_df["perturbation"] = plate_df["Metadata_InChIKey"]
    plate_df["treatment"] = ""
    plate_df["batch"] = plate_df["source"]  # Metadata_Source for train/val split

crops_cfg = cfg.global_.crops
table = extract_plate_crops(
    plate_key=plate_key,
    selected_df=plate_df,
    output_root=cfg.output_root,
    patch_size=crops_cfg.patch_size,
    norm_scheme=crops_cfg.normalization.scheme,
    norm_low=crops_cfg.normalization.low,
    norm_high=crops_cfg.normalization.high,
)

# Resize cell and mask separately after the cell/mask split.
if table.num_rows > 0 and crops_cfg.patch_size != crops_cfg.image_size:
    if isinstance(cfg.data_source, JumpConfig):
        n_image_channels = len(cfg.data_source.channel_s3_keys)
    elif isinstance(cfg.data_source, (Rxrx1Config, Rxrx3CoreConfig)):
        n_image_channels = len(cfg.data_source.channel_names)
    else:
        raise ValueError(f"Unknown data source type: {type(cfg.data_source)}")

    out_hw = (crops_cfg.image_size, crops_cfg.image_size)

    # Fluorescence channels: bilinear (n_mask_channels=0 → all channels bilinear).
    resized_cells = resize_batch(
        table["cell"].to_pylist(),
        in_shape=(n_image_channels, crops_cfg.patch_size, crops_cfg.patch_size),
        out_hw=out_hw,
        n_mask_channels=0,
    )
    cell_idx = table.schema.get_field_index("cell")
    table = table.set_column(cell_idx, "cell", pa.array(resized_cells, type=pa.binary()))

    # Segmentation masks: nearest-neighbour (all channels are masks).
    resized_masks = resize_batch(
        table["mask"].to_pylist(),
        in_shape=(N_MASK_CHANNELS, crops_cfg.patch_size, crops_cfg.patch_size),
        out_hw=out_hw,
        n_mask_channels=N_MASK_CHANNELS,
    )
    mask_idx = table.schema.get_field_index("mask")
    table = table.set_column(mask_idx, "mask", pa.array(resized_masks, type=pa.binary()))

parquet_path = Path(snakemake_obj.output.parquet)
parquet_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = parquet_path.with_suffix(".tmp.parquet")
pq.write_table(table, str(tmp_path), compression="zstd", use_dictionary=True)
tmp_path.rename(parquet_path)

logger.info(f"extract_crops_plate: wrote {table.num_rows} rows to {parquet_path}")

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(f"plate_key={plate_key}\nn_crops={table.num_rows}\n")
