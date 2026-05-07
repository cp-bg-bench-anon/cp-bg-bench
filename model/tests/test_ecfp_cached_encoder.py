"""Tests for ECFPCachedEncoder: config, lookup, error handling, CLIPModel integration, 5-step training, inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf

from cp_bg_bench_model import save_checkpoint
from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.datamodule import ImageMoleculeDataModule

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from cp_bg_bench_model.encoders.molecule_encoders import (
    ECFPCachedConfig,
    ECFPCachedEncoder,
    MoleculeEncoderRegistry,
)
from cp_bg_bench_model.models import CLIPModel
from cp_bg_bench_model.models.models import PretrainModule

_EMBED_DIM = 64
_SMILES = ["CCO", "CC(=O)O", "c1ccccc1"]
_HW = 32
_IN_CHANNELS = 5


# ── helpers ──────────────────────────────────────────────────────────────────


def _write_parquet(path: Path, smiles: list[str]) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Metadata_SMILES": smiles,
            "ecfp4_2048": [rng.random(2048).astype(np.float32).tolist() for _ in smiles],
        }
    )
    df.to_parquet(path, index=False)
    return path


def _write_hf_dataset(path: Path, smiles: list[str]) -> None:
    rng = np.random.default_rng(1)
    buffers = [rng.integers(0, 256, size=_IN_CHANNELS * _HW * _HW, dtype=np.uint8).tobytes() for _ in smiles]
    Dataset.from_dict({"cell": buffers, DatasetEnum.PERTURBATION: smiles}).save_to_disk(str(path))


def _make_cfg() -> object:
    return OmegaConf.create(
        {
            "model": {
                "optimizer": {"lr_head": 3e-4, "lr_lora": 1e-4, "lr_temp": 1e-5},
            },
            "logging": {"train_loss_every_n_steps": 1, "knn_every_n_steps": 1000},
        }
    )


# ── config ────────────────────────────────────────────────────────────────────


def test_ecfp_cached_config_fields():
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=128, fingerprint_path="/some/path.parquet")
    assert cfg.fingerprint_path == "/some/path.parquet"
    assert cfg.embed_dim == 128
    assert cfg.projection_head == "mlp"


def test_ecfp_cached_config_empty_path_raises(tmp_path: Path):
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=64, fingerprint_path="")
    with pytest.raises(ValueError, match="fingerprint_path"):
        ECFPCachedEncoder(cfg)


def test_ecfp_cached_config_missing_file_raises(tmp_path: Path):
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=64, fingerprint_path=str(tmp_path / "nope.parquet"))
    with pytest.raises(FileNotFoundError):
        ECFPCachedEncoder(cfg)


# ── lookup ────────────────────────────────────────────────────────────────────


def test_ecfp_cached_encoder_lookup_shape(tmp_path: Path):
    fp_path = _write_parquet(tmp_path / "fp.parquet", _SMILES)
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=_EMBED_DIM, fingerprint_path=str(fp_path))
    enc = ECFPCachedEncoder(cfg)
    enc.eval()
    with torch.no_grad():
        out = enc(_SMILES)
    assert out.shape == (len(_SMILES), _EMBED_DIM)
    assert torch.isfinite(out).all()


def test_ecfp_cached_encoder_custom_key_column(tmp_path: Path):
    """key_column overrides the default 'Metadata_SMILES' column name."""
    path = tmp_path / "fp_gene.parquet"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "gene_id": _SMILES,
            "ecfp4_2048": [rng.random(2048).astype(np.float32).tolist() for _ in _SMILES],
        }
    )
    df.to_parquet(path, index=False)
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=_EMBED_DIM, fingerprint_path=str(path), key_column="gene_id")
    enc = ECFPCachedEncoder(cfg)
    enc.eval()
    with torch.no_grad():
        out = enc(_SMILES)
    assert out.shape == (len(_SMILES), _EMBED_DIM)
    assert torch.isfinite(out).all()


def test_ecfp_cached_encoder_missing_smiles_raises(tmp_path: Path):
    fp_path = _write_parquet(tmp_path / "fp.parquet", _SMILES)
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=_EMBED_DIM, fingerprint_path=str(fp_path))
    enc = ECFPCachedEncoder(cfg)
    with pytest.raises(KeyError, match="not in fingerprint lookup"):
        enc(["UNKNOWN_SMILES_XYZ"])


def test_load_lookup_raises_on_bad_shape(tmp_path: Path):
    path = tmp_path / "bad_shape.parquet"
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Metadata_SMILES": _SMILES,
            "ecfp4_2048": [rng.random(512).astype(np.float32).tolist() for _ in _SMILES],
        }
    )
    df.to_parquet(path, index=False)
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=_EMBED_DIM, fingerprint_path=str(path))
    with pytest.raises(ValueError, match="wrong shape"):
        ECFPCachedEncoder(cfg)


def test_load_lookup_raises_on_nan(tmp_path: Path):
    path = tmp_path / "nan.parquet"
    rng = np.random.default_rng(0)
    rows = [rng.random(2048).astype(np.float32).tolist() for _ in _SMILES]
    rows[0][0] = float("nan")
    df = pd.DataFrame({"Metadata_SMILES": _SMILES, "ecfp4_2048": rows})
    df.to_parquet(path, index=False)
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=_EMBED_DIM, fingerprint_path=str(path))
    with pytest.raises(ValueError, match="NaN/Inf"):
        ECFPCachedEncoder(cfg)


def test_encode_smiles_missing_key_message(tmp_path: Path):
    fp_path = _write_parquet(tmp_path / "fp.parquet", _SMILES)
    cfg = ECFPCachedConfig(name="ecfp_cached", embed_dim=_EMBED_DIM, fingerprint_path=str(fp_path))
    enc = ECFPCachedEncoder(cfg)
    with pytest.raises(KeyError, match="MISSING_MOL"):
        enc(["CCO", "MISSING_MOL"])


def test_ecfp_cached_registry_lookup(tmp_path: Path):
    fp_path = _write_parquet(tmp_path / "fp.parquet", _SMILES[:1])
    enc = MoleculeEncoderRegistry.build_from_name("ecfp_cached", embed_dim=64, fingerprint_path=str(fp_path))
    assert isinstance(enc, ECFPCachedEncoder)
    assert enc.embed_dim == 64


# ── CLIPModel integration ─────────────────────────────────────────────────────


def test_ecfp_cached_in_clip_model_forward(tmp_path: Path):
    fp_path = _write_parquet(tmp_path / "fp.parquet", _SMILES)
    torch.manual_seed(0)
    model = CLIPModel(
        embed_dim=_EMBED_DIM,
        image_encoder_name="densenet",
        perturbation_encoder_name="ecfp_cached",
        loss="CLIP",
        image_size=_HW,
        in_channels=_IN_CHANNELS,
        freeze_backbone_when_no_lora=False,
        fingerprint_path=str(fp_path),
    ).eval()

    B = len(_SMILES)
    img = torch.randint(0, 256, (B, _IN_CHANNELS, _HW, _HW), dtype=torch.uint8)
    batch = {DatasetEnum.IMG: img, DatasetEnum.PERTURBATION: _SMILES}
    with torch.no_grad():
        img_emb, mol_emb = model(batch)
    assert img_emb.shape == (B, _EMBED_DIM)
    assert mol_emb.shape == (B, _EMBED_DIM)
    assert torch.isfinite(img_emb).all()
    assert torch.isfinite(mol_emb).all()


def test_data_image_size_resize(tmp_path: Path):
    """DataModule inserts T.Resize when data_image_size != image_size."""
    data_hw = _HW       # 32 — stored size in HF dataset blobs
    model_hw = _HW // 2  # 16 — size the model expects

    smiles = ["CCO"] * 4
    ds_path = tmp_path / "ds"
    _write_hf_dataset(ds_path, smiles)  # images stored at _HW × _HW

    dm = ImageMoleculeDataModule(
        dataset_path=str(ds_path),
        image_encoder_name="densenet",
        perturbation_encoder_name="ecfp_cached",
        batch_size=4,
        num_workers=0,
        in_channels=_IN_CHANNELS,
        image_size=model_hw,
        data_image_size=data_hw,
        split_by_column=DatasetEnum.PERTURBATION.value,
        split_stratify_by=None,
        stratify_by_column=None,
        test_frac=0.0,
        val_frac=0.0,
        perturbations_per_batch=None,
        augment=False,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    imgs = batch[DatasetEnum.IMG]
    assert imgs.shape[-2:] == (model_hw, model_hw), f"Expected ({model_hw}, {model_hw}), got {tuple(imgs.shape[-2:])}"


# ── 5-step Lightning training ─────────────────────────────────────────────────


def test_ecfp_cached_5step_train(tmp_path: Path):
    """Full training loop: synthetic dataset + parquet → 5 steps, finite loss."""
    import pytorch_lightning as pl

    smiles = ["CCO"] * 4 + ["CC(=O)O"] * 4  # 8 rows, 2 compounds
    fp_path = _write_parquet(tmp_path / "fp.parquet", list(set(smiles)))
    ds_path = tmp_path / "ds"
    _write_hf_dataset(ds_path, smiles)

    dm = ImageMoleculeDataModule(
        dataset_path=str(ds_path),
        image_encoder_name="densenet",
        perturbation_encoder_name="ecfp_cached",
        batch_size=4,
        num_workers=0,
        in_channels=_IN_CHANNELS,
        image_size=_HW,
        split_by_column=DatasetEnum.PERTURBATION.value,
        split_stratify_by=None,
        stratify_by_column=None,
        test_frac=0.0,
        val_frac=0.0,
        perturbations_per_batch=None,
        augment=False,
    )

    module = PretrainModule(
        embed_dim=_EMBED_DIM,
        image_encoder_name="densenet",
        perturbation_encoder_name="ecfp_cached",
        loss="CLIP",
        image_size=_HW,
        in_channels=_IN_CHANNELS,
        temperature=0.1,
        lr=1e-4,
        weight_decay=1e-4,
        cfg=_make_cfg(),
        strategy="auto",
        freeze_backbone_when_no_lora=False,
        fingerprint_path=str(fp_path),
    )

    trainer = pl.Trainer(
        max_steps=5,
        accelerator="cpu",
        gradient_clip_val=5.0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(module, dm)

    assert "train_loss_step" in trainer.logged_metrics, "train_loss_step was never logged"
    loss_val = float(trainer.logged_metrics["train_loss_step"])
    assert np.isfinite(loss_val), f"Loss exploded: {loss_val}"


# ── inference: train → checkpoint → embed ────────────────────────────────────


def test_ecfp_cached_inference_after_5step_train(tmp_path: Path):
    """Train 5 steps → save_checkpoint → embed_dataset → finite unit-normed float16."""
    import pytorch_lightning as pl
    from embed_dataset import embed_dataset  # noqa: E402 (scripts/ not a package)

    smiles = ["CCO"] * 4 + ["CC(=O)O"] * 4
    fp_path = _write_parquet(tmp_path / "fp.parquet", list(set(smiles)))
    ds_path = tmp_path / "ds"
    _write_hf_dataset(ds_path, smiles)

    module = PretrainModule(
        embed_dim=_EMBED_DIM,
        image_encoder_name="densenet",
        perturbation_encoder_name="ecfp_cached",
        loss="CLIP",
        image_size=_HW,
        in_channels=_IN_CHANNELS,
        temperature=0.1,
        lr=1e-4,
        weight_decay=1e-4,
        cfg=_make_cfg(),
        strategy="auto",
        freeze_backbone_when_no_lora=False,
        fingerprint_path=str(fp_path),
    )
    dm = ImageMoleculeDataModule(
        dataset_path=str(ds_path),
        image_encoder_name="densenet",
        perturbation_encoder_name="ecfp_cached",
        batch_size=4,
        num_workers=0,
        in_channels=_IN_CHANNELS,
        image_size=_HW,
        split_by_column=DatasetEnum.PERTURBATION.value,
        split_stratify_by=None,
        stratify_by_column=None,
        test_frac=0.0,
        val_frac=0.0,
        perturbations_per_batch=None,
        augment=False,
    )

    pl.Trainer(
        max_steps=5,
        accelerator="cpu",
        gradient_clip_val=5.0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    ).fit(module, dm)

    ckpt = save_checkpoint(module.model, tmp_path / "predictor.pt", merge_lora=False)

    out = embed_dataset(
        checkpoint=ckpt,
        dataset_path=ds_path,
        output=tmp_path / "embeddings.parquet",
        channels=_IN_CHANNELS,
        image_size=_HW,
        batch_size=8,
        device="cpu",
    )

    df = pd.read_parquet(out)
    assert len(df) == len(smiles)
    arr = np.stack(df["embedding"].to_list())
    assert arr.shape == (len(smiles), _EMBED_DIM)
    assert arr.dtype == np.float16
    assert np.isfinite(arr).all()
    norms = np.linalg.norm(arr.astype(np.float32), axis=1)
    np.testing.assert_allclose(norms, np.ones(len(smiles)), atol=1e-2)

    # metadata baked into checkpoint
    from cp_bg_bench_model import Cp_bg_benchModelPredictor

    predictor = Cp_bg_benchModelPredictor.load(ckpt)
    assert predictor.metadata["in_channels"] == _IN_CHANNELS
    assert predictor.metadata["image_size"] == _HW
    assert predictor.metadata["embed_dim"] == _EMBED_DIM
    assert "saved_at" in predictor.metadata
