from __future__ import annotations

import torch

from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.encoders.image_encoders import DinoV3ViTConfig, TimmCNNConfig
from cp_bg_bench_model.models import CLIPModel


def _build_model(in_channels: int) -> CLIPModel:
    return CLIPModel(
        embed_dim=128,
        image_encoder_name="densenet",
        perturbation_encoder_name="precomputed",
        loss="CLIP",
        image_size=128,
        in_channels=in_channels,
        precomputed_in_dim=64,
        freeze_backbone_when_no_lora=False,
    )


def _synthetic_batch(in_channels: int, batch_size: int = 4) -> dict:
    img = torch.randint(0, 256, (batch_size, in_channels, 128, 128), dtype=torch.uint8)
    mol = torch.randn(batch_size, 64, dtype=torch.float32)
    return {DatasetEnum.IMG: img, DatasetEnum.PERTURBATION: mol}


def _check_forward(in_channels: int):
    torch.manual_seed(0)
    model = _build_model(in_channels=in_channels).eval()
    batch = _synthetic_batch(in_channels=in_channels, batch_size=4)
    with torch.no_grad():
        img_embed, mol_embed = model(batch)
    assert img_embed.shape == (4, 128)
    assert mol_embed.shape == (4, 128)
    assert torch.isfinite(img_embed).all()
    assert torch.isfinite(mol_embed).all()


def test_forward_5ch_synthetic():
    _check_forward(5)


def test_forward_6ch_synthetic():
    """Locks n-channel parametrisation: same model accepts non-default channel count."""
    _check_forward(6)


def test_native_channels_flag_skips_adapter():
    """native_channels=True must result in adapter=None on DinoV3ViTEncoder and TimmCNNEncoder."""
    from cp_bg_bench_model.encoders.image_encoders import ImageEncoderRegistry

    for enc_name in ("dinov3", "densenet"):
        enc = ImageEncoderRegistry.build_from_name(
            enc_name, embed_dim=64, image_size=32, in_channels=3, native_channels=True
        )
        assert enc.adapter is None, f"{enc_name}: adapter should be None when native_channels=True"

    # Default (native_channels=False) must still build the adapter
    enc_default = ImageEncoderRegistry.build_from_name("densenet", embed_dim=64, image_size=32, in_channels=5)
    assert enc_default.adapter is not None, "adapter must exist when native_channels=False"
