"""Rule H driver: convert resized parquets to HuggingFace Dataset."""

from pathlib import Path

from lamin_utils import logger

from cp_bg_bench.config import JumpConfig, Rxrx1Config, Rxrx3CoreConfig
from cp_bg_bench.config import load as load_config
from cp_bg_bench.datasets.hf import _META_COLS, build_hf_features, parquet_dir_to_hf

_CONTROL_LABEL = "CONTROL"
_RXRX3_EMPTY_GENE = "EMPTY_control"
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)

input_dir = cfg.output_root / "crops_unfiltered"
output_dir = cfg.output_root / "datasets" / "crops_unfiltered_hf"

logger.info(f"build_unfiltered_hf: {input_dir} → {output_dir}")

features = build_hf_features(extra_meta_cols=[*_META_COLS, "Metadata_Perturbation"])


def _make_sirna_row_filter(valid_ids: set[str]):
    """Drop treatment rows whose siRNA has no gene embedding; keep all controls."""

    def _filter(df):
        is_treatment = df["Metadata_PlateType"] == "treatment"
        keep = ~is_treatment | df["Metadata_InChIKey"].isin(valid_ids)
        n_dropped = int((~keep).sum())
        if n_dropped:
            logger.warning(
                f"build_unfiltered_hf: dropped {n_dropped:,} rows targeting siRNAs "
                "without gene embeddings"
            )
        return df[keep].reset_index(drop=True)

    return _filter


row_filter = None
filter_files = list(snakemake_obj.input.perturbation_filter)
if filter_files:
    valid_ids = set(Path(filter_files[0]).read_text().splitlines()) - {""}
    logger.info(f"build_unfiltered_hf: siRNA allowlist loaded ({len(valid_ids):,} valid IDs)")
    row_filter = _make_sirna_row_filter(valid_ids)

_data_source = cfg.data_source


def _add_perturbation_col(df):
    """Derive Metadata_Perturbation column.

    - JUMP:       SMILES (DMSO controls carry their own SMILES, CS(C)=O)
    - RxRx1:      siRNA ThermoFisher ID, or _CONTROL_LABEL for non-treatment wells
    - RxRx3Core:  HGNC gene symbol, or _CONTROL_LABEL for EMPTY_control wells
    """
    if isinstance(_data_source, JumpConfig):
        df["Metadata_Perturbation"] = df["Metadata_SMILES"]
    elif isinstance(_data_source, Rxrx1Config):
        # well_type → Metadata_PlateType; anything other than "treatment" is a control
        is_treatment = df["Metadata_PlateType"] == "treatment"
        df["Metadata_Perturbation"] = df["Metadata_InChIKey"].where(is_treatment, _CONTROL_LABEL)
    elif isinstance(_data_source, Rxrx3CoreConfig):
        # perturbation_type is always "CRISPR"; controls are identified by gene symbol
        is_control = df["Metadata_InChIKey"] == _RXRX3_EMPTY_GENE
        df["Metadata_Perturbation"] = df["Metadata_InChIKey"].where(~is_control, _CONTROL_LABEL)
    else:
        raise ValueError(f"Unknown data source type: {type(_data_source)}")
    return df


parquet_dir_to_hf(
    parquet_dir=input_dir,
    output_hf_dir=output_dir,
    features=features,
    tmp_root=cfg.output_root / "datasets" / "_tmp_hf_build",
    row_filter=row_filter,
    row_transform=_add_perturbation_col,
)

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(f"output={output_dir}\n")
logger.info(f"build_unfiltered_hf: done → {output_dir}")
