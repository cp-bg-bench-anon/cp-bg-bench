"""Tests for SubCellConfig, SubCellEncoder, and registry integration (no network access)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from cp_bg_bench_model.encoders._subcell_vit import GatedAttentionPooler, ViTPoolClassifier
from cp_bg_bench_model.encoders.image_encoders import (
    ImageEncoderRegistry,
    SubCellConfig,
    SubCellEncoder,
    _indices_from_names,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _dummy_trunk(in_ch: int, out_dim: int = 1536) -> ViTPoolClassifier:
    """Build a ViTPoolClassifier with tiny random weights — no checkpoint download."""
    model_cfg = {
        "vit_model": {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "image_size": 16,
            "patch_size": 8,
            "num_channels": in_ch,
            "qkv_bias": True,
        },
        "pool_model": {"dim": 32, "int_dim": 16, "num_heads": 2},
        "num_classes": 4,
    }
    return ViTPoolClassifier(model_cfg)


def _mock_load_trunk(cfg: SubCellConfig) -> tuple:
    """Return a tiny dummy trunk instead of downloading from S3."""
    n_ch = 3 if cfg.variant == "er_dna_protein" else 4
    trunk = _dummy_trunk(in_ch=n_ch, out_dim=64)
    # out_dim = pool_model.out_dim = 32 * 2 = 64
    return trunk, 64  # base_dim=64 for the tiny dummy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_subcell_config_defaults():
    cfg = SubCellConfig(name="subcell")
    assert cfg.variant == "er_dna_protein"
    assert cfg.channel_mode == "rotation"
    assert cfg.channel_names is None
    assert cfg.reference_channel_indices is None
    assert cfg.extra_channel_indices is None
    assert cfg.native_channels is True


def test_subcell_config_immutable():
    cfg = SubCellConfig(name="subcell")
    with pytest.raises((AttributeError, TypeError)):
        cfg.variant = "all_channels"  # type: ignore[misc]


def test_subcell_config_replace():
    cfg = SubCellConfig(name="subcell")
    cfg2 = replace(cfg, variant="all_channels", channel_mode="projection")
    assert cfg2.variant == "all_channels"
    assert cfg2.channel_mode == "projection"
    assert cfg.variant == "er_dna_protein"  # original unchanged


# ---------------------------------------------------------------------------
# SubCellEncoder — rotation mode
# ---------------------------------------------------------------------------


@pytest.fixture()
def rotation_encoder():
    cfg = SubCellConfig(
        name="subcell",
        embed_dim=32,
        in_channels=5,
        variant="er_dna_protein",
        channel_mode="rotation",
        reference_channel_indices=(1, 0),
        extra_channel_indices=(4, 3, 2),
    )
    with patch(
        "cp_bg_bench_model.encoders._subcell_loader.load_subcell_trunk",
        side_effect=lambda **kw: _mock_load_trunk(cfg),
    ):
        enc = SubCellEncoder(cfg)
    return enc


def test_rotation_out_features(rotation_encoder):
    enc = rotation_encoder
    # out_features = base_dim * N_extras = 64 * 3 = 192
    assert enc.out_features == 64 * 3


def test_rotation_forward_shape(rotation_encoder):
    enc = rotation_encoder.eval()
    x = torch.rand(2, 5, 16, 16)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 32), f"expected (2, 32), got {tuple(out.shape)}"
    assert torch.isfinite(out).all()


def test_rotation_no_channel_bundle(rotation_encoder):
    assert rotation_encoder.channel_bundle is None


def test_rotation_trainable_groups(rotation_encoder):
    groups = rotation_encoder.trainable_param_groups()
    assert "head" in groups and "lora" in groups
    assert len(groups["head"]) > 0  # head params should be trainable


# ---------------------------------------------------------------------------
# SubCellEncoder — projection mode
# ---------------------------------------------------------------------------


@pytest.fixture()
def projection_encoder():
    cfg = SubCellConfig(
        name="subcell",
        embed_dim=32,
        in_channels=5,
        variant="all_channels",
        channel_mode="projection",
        passthrough_channel_indices=(4, 1, 0),
        bundle_channel_indices=(2, 3),
    )
    with patch(
        "cp_bg_bench_model.encoders._subcell_loader.load_subcell_trunk",
        side_effect=lambda **kw: _mock_load_trunk(cfg),
    ):
        enc = SubCellEncoder(cfg)
    return enc


def test_projection_out_features(projection_encoder):
    assert projection_encoder.out_features == 64  # base_dim only


def test_projection_forward_shape(projection_encoder):
    enc = projection_encoder.eval()
    x = torch.rand(2, 5, 16, 16)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_projection_channel_bundle_in_trainable_groups(projection_encoder):
    groups = projection_encoder.trainable_param_groups()
    bundle_params = list(projection_encoder.channel_bundle.parameters())
    head_params = groups["head"]
    # channel_bundle params must appear in head group
    assert any(any(p is bp for bp in bundle_params) for p in head_params)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_subcell():
    assert ImageEncoderRegistry.has("subcell")


def test_registry_list_names():
    assert "subcell" in ImageEncoderRegistry.list_names()


def test_registry_get_config():
    cfg = ImageEncoderRegistry.get_config("subcell")
    assert isinstance(cfg, SubCellConfig)
    assert cfg.name == "subcell"
    # Index fields must be None — channel_names is the required resolution path.
    assert cfg.reference_channel_indices is None
    assert cfg.extra_channel_indices is None
    assert cfg.passthrough_channel_indices is None
    assert cfg.bundle_channel_indices is None


def test_registry_default_requires_channel_names():
    """Building a SubCellEncoder from the bare registry default must raise, not silently mis-map."""
    cfg = ImageEncoderRegistry.get_config("subcell")
    with patch(
        "cp_bg_bench_model.encoders._subcell_loader.load_subcell_trunk",
        side_effect=lambda **kw: _mock_load_trunk(cfg),
    ):
        with pytest.raises(ValueError, match="channel_names"):
            SubCellEncoder(cfg)


def test_registry_get_config_override():
    cfg = ImageEncoderRegistry.get_config("subcell", embed_dim=64, image_size=224)
    assert cfg.embed_dim == 64
    assert cfg.image_size == 224


# ---------------------------------------------------------------------------
# GatedAttentionPooler
# ---------------------------------------------------------------------------


def test_gated_attention_pooler_output_dim():
    pool = GatedAttentionPooler(dim=768, int_dim=512, num_heads=2)
    assert pool.out_dim == 768 * 2  # 1536

    x = torch.randn(2, 100, 768)
    out, attn = pool(x)
    assert out.shape == (2, 1536)
    assert attn.shape == (2, 2, 100)


# ---------------------------------------------------------------------------
# Min-max normalisation
# ---------------------------------------------------------------------------


def test_minmax_norm_output_range():
    x = torch.rand(3, 5, 32, 32) * 255
    x_norm = SubCellEncoder._minmax_norm(x)
    assert x_norm.min() >= 0.0 - 1e-6
    assert x_norm.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# _indices_from_names — index derivation from channel names
# ---------------------------------------------------------------------------

JUMP_NAMES = ("DNA", "AGP", "ER", "Mito", "RNA")
RXRX1_NAMES = ("DNA", "ER", "AGP_actin", "RNA", "Mito", "AGP_membrane")


def test_indices_from_names_jump_rotation():
    ref, extras, passthrough, bundle = _indices_from_names(JUMP_NAMES, "rotation", ())
    assert ref == (2, 0), f"ER=2, DNA=0; got {ref}"
    assert set(extras) == {1, 3, 4}  # AGP, Mito, RNA
    assert extras == (1, 3, 4)  # source order
    assert passthrough is None
    assert bundle is None


def test_indices_from_names_jump_projection():
    ref, extras, passthrough, bundle = _indices_from_names(JUMP_NAMES, "projection", ())
    assert passthrough == (3, 2, 0)  # Mito→MT, ER, DNA
    assert set(bundle) == {1, 4}  # AGP, RNA (everything not in MT/ER/DNA)


def test_indices_from_names_rxrx1_rotation():
    ref, extras, passthrough, bundle = _indices_from_names(RXRX1_NAMES, "rotation", ())
    assert ref == (1, 0)  # ER=1, DNA=0
    assert extras == (2, 3, 4, 5)  # AGP_actin, RNA, Mito, AGP_membrane in source order


def test_indices_from_names_rxrx1_projection():
    ref, extras, passthrough, bundle = _indices_from_names(RXRX1_NAMES, "projection", ())
    assert passthrough == (4, 1, 0)  # Mito→MT, ER, DNA
    assert set(bundle) == {2, 3, 5}  # AGP_actin, RNA, AGP_membrane


def test_indices_from_names_missing_required():
    with pytest.raises(ValueError, match="ER"):
        _indices_from_names(("DNA", "AGP", "Mito"), "rotation", ())


def test_indices_from_names_missing_mito_projection():
    with pytest.raises(ValueError, match="Mito"):
        _indices_from_names(("DNA", "ER", "AGP"), "projection", ())


def test_indices_from_names_protein_bundle_priority():
    # Only route RNA and Mito through protein slot, in that order
    ref, extras, _, _ = _indices_from_names(JUMP_NAMES, "rotation", ("RNA", "Mito"))
    assert extras == (4, 3)  # RNA=4, Mito=3


# ---------------------------------------------------------------------------
# SubCellEncoder — name-based construction (rotation, JUMP layout)
# ---------------------------------------------------------------------------


@pytest.fixture()
def jump_rotation_encoder():
    cfg = SubCellConfig(
        name="subcell",
        embed_dim=32,
        in_channels=5,
        variant="er_dna_protein",
        channel_mode="rotation",
        channel_names=JUMP_NAMES,
    )
    with patch(
        "cp_bg_bench_model.encoders._subcell_loader.load_subcell_trunk",
        side_effect=lambda **kw: _mock_load_trunk(cfg),
    ):
        enc = SubCellEncoder(cfg)
    return enc


def test_jump_rotation_resolved_indices(jump_rotation_encoder):
    enc = jump_rotation_encoder
    assert enc._ref_idx == (2, 0)
    assert enc._extra_idx == (1, 3, 4)


def test_jump_rotation_out_features(jump_rotation_encoder):
    # 3 extra channels × base_dim 64
    assert jump_rotation_encoder.out_features == 64 * 3


def test_jump_rotation_forward_shape(jump_rotation_encoder):
    enc = jump_rotation_encoder.eval()
    x = torch.rand(2, 5, 16, 16)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SubCellEncoder — name-based construction (projection, RxRx1 layout)
# ---------------------------------------------------------------------------


@pytest.fixture()
def rxrx1_projection_encoder():
    cfg = SubCellConfig(
        name="subcell",
        embed_dim=32,
        in_channels=6,
        variant="all_channels",
        channel_mode="projection",
        channel_names=RXRX1_NAMES,
    )
    with patch(
        "cp_bg_bench_model.encoders._subcell_loader.load_subcell_trunk",
        side_effect=lambda **kw: _mock_load_trunk(cfg),
    ):
        enc = SubCellEncoder(cfg)
    return enc


def test_rxrx1_projection_resolved_indices(rxrx1_projection_encoder):
    enc = rxrx1_projection_encoder
    assert enc._passthrough_idx == (4, 1, 0)
    assert set(enc._bundle_idx) == {2, 3, 5}


def test_rxrx1_projection_forward_shape(rxrx1_projection_encoder):
    enc = rxrx1_projection_encoder.eval()
    x = torch.rand(2, 6, 16, 16)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Error: no channel_names and no explicit indices
# ---------------------------------------------------------------------------


def test_encoder_no_channel_info_raises():
    cfg = SubCellConfig(
        name="subcell",
        embed_dim=32,
        in_channels=5,
        variant="er_dna_protein",
        channel_mode="rotation",
        # neither channel_names nor explicit indices
    )
    with patch(
        "cp_bg_bench_model.encoders._subcell_loader.load_subcell_trunk",
        side_effect=lambda **kw: _mock_load_trunk(cfg),
    ):
        with pytest.raises(ValueError, match="channel_names"):
            SubCellEncoder(cfg)
