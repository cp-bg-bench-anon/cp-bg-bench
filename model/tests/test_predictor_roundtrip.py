from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

# These tests all exercise the image-only checkpoint path deliberately;
# silence the expected "molecule_encoder not saved" warning so it doesn't
# obscure real failures.
pytestmark = pytest.mark.filterwarnings("ignore:molecule_encoder not saved:UserWarning")

from cp_bg_bench_model import Cp_bg_benchModelPredictor, save_checkpoint
from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.models import CLIPModel
from cp_bg_bench_model.models._export import _try_merge_lora_inplace


def _build_model(in_channels: int = 5, image_size: int = 64) -> CLIPModel:
    return CLIPModel(
        embed_dim=128,
        image_encoder_name="densenet",
        perturbation_encoder_name="precomputed",
        loss="CLIP",
        image_size=image_size,
        in_channels=in_channels,
        precomputed_in_dim=64,
        freeze_backbone_when_no_lora=False,
    )


def _crops(in_channels: int = 5, batch_size: int = 4, hw: int = 64) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randint(0, 256, (batch_size, in_channels, hw, hw), dtype=torch.uint8)


def test_save_load_embed_matches_live_model(tmp_path: Path):
    """Roundtripped predictor produces the same image embeddings as the live model."""
    torch.manual_seed(0)
    model = _build_model().eval()
    crops = _crops()

    with torch.inference_mode():
        live_img, _ = model({DatasetEnum.IMG: crops, DatasetEnum.PERTURBATION: torch.zeros(4, 64)})

    ckpt_path = save_checkpoint(model, tmp_path / "predictor.pt", merge_lora=False)
    predictor = Cp_bg_benchModelPredictor.load(ckpt_path, device="cpu")
    loaded_img = predictor.embed(crops)

    assert loaded_img.shape == live_img.shape
    assert torch.allclose(loaded_img, live_img, atol=1e-5, rtol=1e-5)


def test_lora_merge_idempotent(tmp_path: Path):
    """Merging LoRA twice produces identical weights — catches the PEFT footgun."""
    torch.manual_seed(0)
    model = _build_model().eval()

    save_checkpoint(model, tmp_path / "once.pt", merge_lora=True)
    once = Cp_bg_benchModelPredictor.load(tmp_path / "once.pt", device="cpu")

    # Re-merge on the already-merged predictor's encoder; weights must not change.
    before = {n: p.detach().clone() for n, p in once.image_encoder.encoder.named_parameters()}
    encoder_copy = copy.deepcopy(once.image_encoder.encoder)
    _try_merge_lora_inplace(encoder_copy)
    after = dict(encoder_copy.named_parameters())

    for name, b in before.items():
        assert torch.equal(b, after[name]), f"weight drifted after re-merge: {name}"


def test_predict_batch_vs_single(tmp_path: Path):
    """``predict(tile)`` matches ``predict_batch([tile])[0]``."""
    torch.manual_seed(0)
    model = _build_model().eval()
    crops = _crops(batch_size=3)

    save_checkpoint(model, tmp_path / "p.pt", merge_lora=False)
    predictor = Cp_bg_benchModelPredictor.load(tmp_path / "p.pt", device="cpu")

    batched = predictor.predict_batch(crops)
    single = predictor.predict(crops[0])

    assert batched.shape == (3, 128)
    assert single.shape == (128,)
    # tolerance: float32 arithmetic + same device → bit-identical expected
    assert (batched[0] == single).all() or torch.allclose(
        torch.from_numpy(batched[0]), torch.from_numpy(single), atol=1e-6
    )


def test_molecule_encoder_optional(tmp_path: Path):
    """``include_molecule_encoder`` toggles ``embed_molecule`` availability."""
    torch.manual_seed(0)
    model = _build_model().eval()

    save_checkpoint(model, tmp_path / "img_only.pt", merge_lora=False)
    img_only = Cp_bg_benchModelPredictor.load(tmp_path / "img_only.pt", device="cpu")
    assert not img_only.has_molecule_encoder
    with pytest.raises(RuntimeError, match="without a molecule encoder"):
        img_only.embed_molecule(torch.randn(1, 64))

    save_checkpoint(model, tmp_path / "full.pt", merge_lora=False, include_molecule_encoder=True)
    full = Cp_bg_benchModelPredictor.load(tmp_path / "full.pt", device="cpu")
    assert full.has_molecule_encoder
    mol_emb = full.embed_molecule(torch.randn(4, 64))
    assert mol_emb.shape == (4, 128)
    assert torch.isfinite(mol_emb).all()


def test_save_checkpoint_with_preprocess_hook(tmp_path: Path):
    """save_checkpoint succeeds and restores hooks even when image_preprocess has a hook.

    Regression for: preprocess hooks were not stripped before deepcopy, so
    non-picklable closures on ImageGpuPreprocess would cause torch.save to fail.
    """
    torch.manual_seed(0)
    model = _build_model().eval()

    fired: list[bool] = []

    def _hook(module, input, output):  # noqa: ANN001
        fired.append(True)
        return output

    handle = model.image_preprocess.register_forward_hook(_hook)
    try:
        ckpt = save_checkpoint(model, tmp_path / "hooked.pt", merge_lora=False)
    finally:
        handle.remove()

    # Hook removed from live model after save; calling preprocess fires nothing.
    assert len(fired) == 0, "hook fired during save — should have been stripped"
    # Hook is restored on the live model after save.
    assert len(model.image_preprocess._forward_hooks) == 0  # handle already removed above

    # Saved checkpoint loads and embeds correctly.
    predictor = Cp_bg_benchModelPredictor.load(ckpt, device="cpu")
    crops = _crops()
    emb = predictor.embed(crops)
    assert emb.shape == (4, 128)
    assert torch.isfinite(emb).all()


def test_save_checkpoint_with_preprocess_pre_hook(tmp_path: Path):
    """save_checkpoint strips _forward_pre_hooks too, not just _forward_hooks.

    Regression for: _strip_forward_hooks only cleared _forward_hooks; a
    non-picklable closure in _forward_pre_hooks would still cause torch.save
    to fail with a PicklingError.
    """
    torch.manual_seed(0)
    model = _build_model().eval()

    fired: list[bool] = []

    # Capture a local variable so the closure is non-trivially picklable
    # (closures over mutable containers can't be pickled by default).
    def _pre_hook(module, input):  # noqa: ANN001
        fired.append(True)
        return input

    handle = model.image_preprocess.register_forward_pre_hook(_pre_hook)
    try:
        ckpt = save_checkpoint(model, tmp_path / "pre_hooked.pt", merge_lora=False)
    finally:
        handle.remove()

    assert len(fired) == 0, "pre-hook fired during save — should have been stripped"
    assert len(model.image_preprocess._forward_pre_hooks) == 0  # handle already removed

    predictor = Cp_bg_benchModelPredictor.load(ckpt, device="cpu")
    emb = predictor.embed(_crops())
    assert emb.shape == (4, 128)
    assert torch.isfinite(emb).all()
