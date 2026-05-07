from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = (Path(__file__).resolve().parent.parent / "config").as_posix()


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base="1.2", config_dir=CONFIG_DIR):
        return compose(config_name="default", overrides=overrides)


def test_compose_default():
    """Default Hydra config composes without errors and key fields resolve."""
    cfg = _compose([])
    assert cfg.name == "pretrain"
    assert cfg.training.logger.wandb.project == "cp-bg-bench-model"
    assert cfg.model.litmodule._target_ == "cp_bg_bench_model.models.models.PretrainModule"
    assert cfg.datamodule._target_ == "cp_bg_bench_model.datamodule.ImageMoleculeDataModule"
    assert "/cp-bg-bench" in cfg.paths.root


@pytest.mark.parametrize("variant", ["dinov3", "openphenom", "densenet", "subcell"])
def test_compose_model_variants(variant: str):
    """Each model variant overlays cleanly onto default."""
    cfg = _compose([f"model={variant}"])
    assert cfg.model.litmodule.image_encoder_name == variant


def test_compose_gene_esm2_rxrx1():
    """gene_esm2 model + rxrx1 datamodule compose correctly."""
    cfg = _compose(["model=gene_esm2", "datamodule=rxrx1"])
    assert cfg.model.litmodule.perturbation_encoder_name == "gene_lookup"
    assert cfg.model.litmodule.loss == "CLIP"
    assert cfg.model.litmodule.perturbation_head == "mlp"
    assert cfg.model.litmodule.in_channels == 6
    assert "esm2_1280.parquet" in cfg.model.litmodule.fingerprint_path
    assert cfg.datamodule.perturbation_source_col == "Metadata_InChIKey"
    assert cfg.datamodule.split_by_column == "Metadata_InChIKey"


def test_compose_data_variants():
    """data/ group switches paths.data subdirectory."""
    for variant, suffix in [
        ("crops", "training/crops"),
        ("seg", "training/segmented"),
        ("crops_density", "training/crops_density"),
        ("seg_density", "training/seg_density"),
    ]:
        cfg = _compose([f"data={variant}"])
        resolved_path = OmegaConf.to_container(cfg.paths, resolve=True)["data"]
        assert resolved_path.endswith(suffix), (variant, resolved_path)


@pytest.mark.parametrize(
    "datamodule, expected_names",
    [
        ("jump", ["DNA", "AGP", "ER", "Mito", "RNA"]),
        ("rxrx1", ["DNA", "ER", "AGP_actin", "RNA", "Mito", "AGP_membrane"]),
        ("rxrx3c", ["DNA", "ER", "AGP_actin", "RNA", "Mito", "AGP_membrane"]),
    ],
)
def test_compose_channel_names_resolve(datamodule: str, expected_names: list[str]):
    """channel_names in each datamodule resolves and propagates to litmodule."""
    cfg = _compose([f"datamodule={datamodule}", "model=subcell"])
    # Resolve only the sub-configs we care about to avoid triggering
    # Hydra-runtime interpolations (e.g. ${hydra:runtime.output_dir}) elsewhere.
    dm_names = OmegaConf.to_container(cfg.datamodule, resolve=True)["channel_names"]
    assert dm_names == expected_names, f"datamodule.channel_names wrong: {dm_names}"
    lm_names = OmegaConf.to_container(cfg.model.litmodule, resolve=True)["image_encoder_channel_names"]
    assert lm_names == expected_names, f"litmodule.image_encoder_channel_names wrong: {lm_names}"
