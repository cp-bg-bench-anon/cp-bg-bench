"""Roundtrip test for the encoder export: config sidecar + dual-format load.

Exports a real encoder (built via the ``build_encoder_from_config`` factory)
through :func:`export_image_encoder_with_head`, loads it back with
:func:`load_image_encoder_with_head`, and verifies that the reloaded encoder
produces bit-identical output on the same inputs.

Also unit-tests :func:`_encoder_cfg_to_dict` / :func:`_encoder_cfg_from_dict`
for every registered ``ImageEncoderConfig`` subclass so that adding a new
encoder without updating the registry triggers an obvious test failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cp_bg_bench_model.encoders.image_encoders import (
    DinoV3ViTConfig,
    OpenPhenomConfig,
    SubCellConfig,
    TimmCNNConfig,
    build_encoder_from_config,
)
from cp_bg_bench_model.lora import LoraConfig
from cp_bg_bench_model.models._export import (
    ImageEncoderWithHead,
    _encoder_cfg_from_dict,
    _encoder_cfg_to_dict,
    export_image_encoder_with_head,
    load_image_encoder_with_head,
)


# ── Config-dict roundtrip (fast, no encoder build) ──────────────────────────

_ROUNDTRIP_CONFIGS = [
    pytest.param(
        TimmCNNConfig(name="timm_cnn", timm_name="resnet18", pretrained=False),
        id="timm_cnn_minimal",
    ),
    pytest.param(
        TimmCNNConfig(
            name="timm_cnn",
            timm_name="resnet18",
            pretrained=False,
            lora=LoraConfig(enabled=True, r=4, targets=["conv"]),
        ),
        id="timm_cnn_with_lora",
    ),
    pytest.param(
        DinoV3ViTConfig(name="dinov3", pretrained=False),
        id="dinov3",
    ),
    pytest.param(
        OpenPhenomConfig(name="openphenom"),
        id="openphenom",
    ),
    pytest.param(
        SubCellConfig(
            channel_names=("DNA", "AGP", "ER", "Mito", "RNA"),
            protein_bundle_priority=("AGP", "RNA"),
        ),
        id="subcell_with_tuples",
    ),
]


@pytest.mark.parametrize("cfg", _ROUNDTRIP_CONFIGS)
def test_config_dict_roundtrip(cfg):
    serialized = _encoder_cfg_to_dict(cfg)
    restored = _encoder_cfg_from_dict(json.loads(json.dumps(serialized)))
    assert type(restored) is type(cfg)
    # asdict-equality is robust: tuples are preserved through our
    # type-driven list→tuple conversion, so the dataclasses compare equal.
    assert asdict(restored) == asdict(cfg)


def test_unknown_config_class_rejected():
    with pytest.raises(ValueError, match="Unknown encoder config class"):
        _encoder_cfg_from_dict({"encoder_config_class": "NotARealConfig", "encoder_config": {}})


# ── Full export → load roundtrip via the real factory ───────────────────────


class _LitWrapper(nn.Module):
    """Minimal stand-in for PretrainModule: exposes ``.model.image_encoder``."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.model = nn.Module()
        self.model.image_encoder = encoder


def _make_encoder() -> nn.Module:
    """Build a small encoder offline via the real factory."""
    torch.manual_seed(0)
    cfg = TimmCNNConfig(
        name="timm_cnn",
        embed_dim=64,
        image_size=64,
        in_channels=5,
        timm_name="resnet18",
        pretrained=False,
        freeze_backbone_when_no_lora=False,
    )
    return build_encoder_from_config(cfg).eval()


def test_export_writes_all_three_artifacts(tmp_path: Path):
    lit = _LitWrapper(_make_encoder())
    paths = export_image_encoder_with_head(
        lit_model=lit, out_dir=tmp_path, file_stem="imgenc_test"
    )
    assert Path(paths["module"]).is_file()       # .pt  (legacy)
    assert Path(paths["state_dict"]).is_file()   # .pth (new)
    assert Path(paths["config"]).is_file()       # .json (new)


def test_load_prefers_config_sidecar_and_matches_source(tmp_path: Path):
    encoder = _make_encoder()
    lit = _LitWrapper(encoder)
    export_image_encoder_with_head(lit_model=lit, out_dir=tmp_path, file_stem="imgenc_test")

    reference = ImageEncoderWithHead(encoder).eval()
    x = torch.randn(2, 5, 64, 64)
    with torch.inference_mode():
        expected = reference(x)

    loaded = load_image_encoder_with_head(tmp_path / "imgenc_test.pth")
    with torch.inference_mode():
        got = loaded(x)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_load_falls_back_to_legacy_pickle_without_sidecar(tmp_path: Path):
    """Old checkpoints have only .pt (+ .pth); no .json sidecar.

    The loader must still accept them, because re-running training to
    produce sidecars for every already-trained model is not an option.
    """
    encoder = _make_encoder()
    lit = _LitWrapper(encoder)
    export_image_encoder_with_head(lit_model=lit, out_dir=tmp_path, file_stem="legacy")
    # Simulate a pre-sidecar export: remove both new-format files.
    (tmp_path / "legacy.json").unlink()
    (tmp_path / "legacy.pth").unlink()

    reference = ImageEncoderWithHead(encoder).eval()
    x = torch.randn(2, 5, 64, 64)
    with torch.inference_mode():
        expected = reference(x)

    loaded = load_image_encoder_with_head(tmp_path / "legacy.pt")
    with torch.inference_mode():
        got = loaded(x)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_load_accepts_any_artifact_path_or_stem(tmp_path: Path):
    lit = _LitWrapper(_make_encoder())
    export_image_encoder_with_head(lit_model=lit, out_dir=tmp_path, file_stem="x")
    stem = tmp_path / "x"

    for path in (stem.with_suffix(".pth"), stem.with_suffix(".json"), stem.with_suffix(".pt"), stem):
        assert isinstance(load_image_encoder_with_head(path), ImageEncoderWithHead)


def test_load_raises_when_nothing_present(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No encoder export found"):
        load_image_encoder_with_head(tmp_path / "missing")
