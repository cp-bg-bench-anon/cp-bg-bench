"""Tests for :mod:`cp_bg_bench.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cp_bg_bench.config import (
    GlobalConfig,
    JumpConfig,
    PipelineConfig,
    Rxrx1Config,
    Rxrx3CoreConfig,
    SelectionConfig,
    load,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

GLOBAL_CONFIGS = ["config.yml", "config_smoke.yml"]
DATA_SOURCE_CONFIGS = ["jump.yml", "smoke.yml"]
RXRX3_CORE_GLOBAL_CONFIGS = ["config_smoke_rxrx3_core.yml", "config_training_rxrx3_core.yml"]


@pytest.mark.parametrize("name", GLOBAL_CONFIGS)
def test_global_configs_roundtrip(name: str) -> None:
    """Every shipped global YAML validates against the pydantic schema."""
    cfg = load(CONFIG_DIR / name)
    assert isinstance(cfg, PipelineConfig)
    assert isinstance(cfg.global_, GlobalConfig)
    assert isinstance(cfg.data_source, JumpConfig)
    assert cfg.global_.data_source == "jump"


@pytest.mark.parametrize("name", RXRX3_CORE_GLOBAL_CONFIGS)
def test_rxrx3_core_global_configs_roundtrip(name: str) -> None:
    """Every shipped rxrx3-core global YAML validates against the pydantic schema."""
    cfg = load(CONFIG_DIR / name)
    assert isinstance(cfg, PipelineConfig)
    assert isinstance(cfg.global_, GlobalConfig)
    assert isinstance(cfg.data_source, Rxrx3CoreConfig)
    assert cfg.global_.data_source == "rxrx3_core"


@pytest.mark.parametrize("name", DATA_SOURCE_CONFIGS)
def test_data_source_configs_roundtrip(name: str) -> None:
    """Data-source YAMLs validate against :class:`JumpConfig`."""
    raw = yaml.safe_load((CONFIG_DIR / name).read_text())
    JumpConfig.model_validate(raw)


def test_output_root_keyed_on_data_source_stem() -> None:
    """``output_root`` resolves to ``paths.root / data_source_stem``."""
    prod = load(CONFIG_DIR / "config.yml")
    smoke = load(CONFIG_DIR / "config_smoke.yml")
    assert prod.data_source_stem == "jump"
    assert smoke.data_source_stem == "smoke"
    assert prod.output_root == prod.paths.root / "jump"
    assert smoke.output_root == smoke.paths.root / "smoke"
    # Production and smoke must not share an output subtree.
    assert prod.output_root != smoke.output_root


def test_load_resolves_relative_data_source(tmp_path: Path) -> None:
    """``data_source_config`` is resolved relative to the global config file."""
    jump_yaml = yaml.safe_load((CONFIG_DIR / "jump.yml").read_text())
    (tmp_path / "ds.yml").write_text(yaml.safe_dump(jump_yaml))

    global_yaml = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    global_yaml["data_source_config"] = "ds.yml"
    global_yaml["paths"]["root"] = str(tmp_path / "out")
    (tmp_path / "cfg.yml").write_text(yaml.safe_dump(global_yaml))

    cfg = load(tmp_path / "cfg.yml")
    assert cfg.data_source_path == (tmp_path / "ds.yml").resolve()
    assert cfg.data_source_stem == "ds"


def test_fingerprint_stable() -> None:
    """Same config bytes → same fingerprint; differing configs differ."""
    cfg_a = load(CONFIG_DIR / "config.yml")
    cfg_b = load(CONFIG_DIR / "config.yml")
    assert cfg_a.fingerprint() == cfg_b.fingerprint()
    assert cfg_a.fingerprint() != load(CONFIG_DIR / "config_smoke.yml").fingerprint()


def test_extra_keys_rejected() -> None:
    """Typos in keys surface as validation errors, not silent drops."""
    raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    raw["selection"]["bogus_key"] = 1
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(raw)


def test_quantiles_must_be_ordered() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    raw["quality_filter"]["quantiles"] = [0.9, 0.1]
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(raw)


def test_normalization_bounds_must_be_ordered() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    raw["crops"]["normalization"]["low"] = 0.9
    raw["crops"]["normalization"]["high"] = 0.1
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(raw)


def test_unknown_strategy_rejected() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    raw["selection"]["strategy"] = "not_a_strategy"
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(raw)


def test_max_fovs_per_well_accepted() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    raw["max_fovs_per_well"] = 2
    cfg = GlobalConfig.model_validate(raw)
    assert cfg.max_fovs_per_well == 2


def test_max_fovs_per_well_defaults_none() -> None:
    cfg = GlobalConfig.model_validate(yaml.safe_load((CONFIG_DIR / "config.yml").read_text()))
    assert cfg.max_fovs_per_well is None


def test_max_fovs_per_well_and_per_plate_independent() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    raw["max_fovs_per_plate"] = 10
    raw["max_fovs_per_well"] = 2
    cfg = GlobalConfig.model_validate(raw)
    assert cfg.max_fovs_per_plate == 10
    assert cfg.max_fovs_per_well == 2


def test_uniform_per_well_requires_cells_per_well() -> None:
    with pytest.raises(ValidationError, match="cells_per_well"):
        SelectionConfig.model_validate({"strategy": "uniform_per_well", "random_seed": 0})


def test_uniform_total_requires_max_cells() -> None:
    with pytest.raises(ValidationError, match="max_cells"):
        SelectionConfig.model_validate({"strategy": "uniform_total", "random_seed": 0})


def test_uniform_total_valid_with_max_cells() -> None:
    cfg = SelectionConfig.model_validate(
        {"strategy": "uniform_total", "random_seed": 0, "max_cells": 500}
    )
    assert cfg.max_cells == 500
    assert cfg.cells_per_well is None


def test_all_strategy_needs_no_extra_params() -> None:
    cfg = SelectionConfig.model_validate({"strategy": "all", "random_seed": 0})
    assert cfg.strategy == "all"
    assert cfg.cells_per_well is None
    assert cfg.max_cells is None


def test_missing_data_source_file_errors(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    raw["data_source_config"] = "does_not_exist.yml"
    (tmp_path / "cfg.yml").write_text(yaml.safe_dump(raw))
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "cfg.yml")


def test_jump_requires_core_metadata_tables() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "jump.yml").read_text())
    raw["metadata_tables"].pop("plate")
    with pytest.raises(ValidationError):
        JumpConfig.model_validate(raw)


def test_missing_config_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.yml")


# ── Rxrx1 stubs ──────────────────────────────────────────────────────────────


def test_rxrx1_config_roundtrip() -> None:
    """Shipped rxrx1.yml validates against Rxrx1Config."""
    raw = yaml.safe_load((CONFIG_DIR / "rxrx1.yml").read_text())
    cfg = Rxrx1Config.model_validate(raw)
    assert len(cfg.channel_names) == 6
    assert len(cfg.samples) >= 1
    assert cfg.metadata_url.startswith("https://")
    assert cfg.images_zip_url.startswith("https://")
    assert cfg.channel_zip_keys == [f"zip_{n}" for n in cfg.channel_names]


def test_rxrx1_smoke_config_roundtrip() -> None:
    """Shipped rxrx1_smoke.yml validates against Rxrx1Config."""
    raw = yaml.safe_load((CONFIG_DIR / "rxrx1_smoke.yml").read_text())
    cfg = Rxrx1Config.model_validate(raw)
    assert cfg.samples[0].experiment == "HEPG2-08"
    assert cfg.samples[0].plate == "1"


def test_load_with_rxrx1_data_source(tmp_path: Path) -> None:
    """load() returns Rxrx1Config when data_source=rxrx1."""
    (tmp_path / "rxrx1.yml").write_text((CONFIG_DIR / "rxrx1.yml").read_text())

    global_raw = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    global_raw["data_source"] = "rxrx1"
    global_raw["data_source_config"] = "rxrx1.yml"
    global_raw["paths"]["root"] = str(tmp_path / "out")
    (tmp_path / "cfg.yml").write_text(yaml.safe_dump(global_raw))

    cfg = load(tmp_path / "cfg.yml")
    assert isinstance(cfg.data_source, Rxrx1Config)
    assert cfg.global_.data_source == "rxrx1"
    assert cfg.data_source_stem == "rxrx1"


# ── Rxrx3Core stubs ───────────────────────────────────────────────────────────


def test_rxrx3_core_smoke_config_roundtrip() -> None:
    """Shipped rxrx3_core_smoke.yml validates against Rxrx3CoreConfig."""
    raw = yaml.safe_load((CONFIG_DIR / "rxrx3_core_smoke.yml").read_text())
    cfg = Rxrx3CoreConfig.model_validate(raw)
    assert len(cfg.channel_names) == 6
    assert cfg.samples[0].experiment == "gene-001"
    assert cfg.hf_repo.startswith("recursionpharma/")


def test_rxrx3_core_training_config_roundtrip() -> None:
    """Shipped rxrx3_training.yml validates against Rxrx3CoreConfig."""
    raw = yaml.safe_load((CONFIG_DIR / "rxrx3_training.yml").read_text())
    cfg = Rxrx3CoreConfig.model_validate(raw)
    assert len(cfg.samples) == 176
    assert cfg.samples[0].experiment == "gene-001"
    assert cfg.samples[-1].experiment == "gene-176"


def test_rxrx3_core_samples_empty_raises() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "rxrx3_core_smoke.yml").read_text())
    raw["samples"] = []
    with pytest.raises(ValidationError, match="samples must be non-empty"):
        Rxrx3CoreConfig.model_validate(raw)


def test_rxrx3_core_channel_names_empty_raises() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "rxrx3_core_smoke.yml").read_text())
    raw["channel_names"] = []
    with pytest.raises(ValidationError, match="channel_names must be non-empty"):
        Rxrx3CoreConfig.model_validate(raw)


def test_load_with_rxrx3_core_data_source(tmp_path: Path) -> None:
    """load() returns Rxrx3CoreConfig when data_source=rxrx3_core."""
    (tmp_path / "rxrx3_core_smoke.yml").write_text(
        (CONFIG_DIR / "rxrx3_core_smoke.yml").read_text()
    )

    global_raw = yaml.safe_load((CONFIG_DIR / "config_smoke_rxrx3_core.yml").read_text())
    global_raw["data_source_config"] = "rxrx3_core_smoke.yml"
    global_raw["paths"]["root"] = str(tmp_path / "out")
    (tmp_path / "cfg.yml").write_text(yaml.safe_dump(global_raw))

    cfg = load(tmp_path / "cfg.yml")
    assert isinstance(cfg.data_source, Rxrx3CoreConfig)
    assert cfg.global_.data_source == "rxrx3_core"
    assert cfg.data_source_stem == "rxrx3_core_smoke"
