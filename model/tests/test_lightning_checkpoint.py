"""Tests that ModelCheckpoint writes .ckpt files and that they are loadable.

Covers:
- best_ckpt callback writes last.ckpt and metric-keyed checkpoints when val fires
- A saved last.ckpt can be used to resume training without error
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.datamodule import ImageMoleculeDataModule
from cp_bg_bench_model.models.models import PretrainModule

_EMBED_DIM = 32
_HW = 32
_IN_CHANNELS = 5
# Two distinct SMILES → two compound classes in val
_SMILES = ["CCO"] * 4 + ["CC(=O)O"] * 4


def _write_parquet(path: Path, smiles: list[str]) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Metadata_SMILES": smiles,
            "ecfp4_2048": [rng.random(2048).astype(np.float32).tolist() for _ in set(smiles)],
        }
    )
    df.to_parquet(path, index=False)
    return path


def _write_dataset(path: Path, smiles: list[str]) -> None:
    rng = np.random.default_rng(1)
    buffers = [
        rng.integers(0, 256, size=_IN_CHANNELS * _HW * _HW, dtype=np.uint8).tobytes()
        for _ in smiles
    ]
    Dataset.from_dict({"cell": buffers, DatasetEnum.PERTURBATION: smiles}).save_to_disk(str(path))


def _make_cfg() -> object:
    return OmegaConf.create(
        {
            "model": {"optimizer": {"lr_head": 3e-4, "lr_lora": 1e-4, "lr_temp": 1e-5}},
            "logging": {"train_loss_every_n_steps": 1, "knn_every_n_steps": 10000},
        }
    )


def _make_module(fp_path: Path) -> PretrainModule:
    return PretrainModule(
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
        fingerprint_path=str(fp_path),
        freeze_backbone_when_no_lora=False,
    )


def _make_dm(ds_path: Path) -> ImageMoleculeDataModule:
    return ImageMoleculeDataModule(
        dataset_path=str(ds_path),
        image_encoder_name="densenet",
        perturbation_encoder_name="ecfp_cached",
        batch_size=4,
        num_workers=0,
        in_channels=_IN_CHANNELS,
        image_size=_HW,
        # Split by perturbation so each class contributes to both train and val.
        split_by_column=DatasetEnum.PERTURBATION.value,
        split_stratify_by=None,
        stratify_by_column=None,
        test_frac=0.0,
        val_frac=0.5,
        perturbations_per_batch=None,
        augment=False,
    )


def _make_ckpt_callback(dirpath: Path) -> ModelCheckpoint:
    return ModelCheckpoint(
        monitor="val_R@1_I2P_macro",
        mode="max",
        save_last=True,
        save_top_k=3,
        dirpath=str(dirpath),
        filename="best-{step}",
        auto_insert_metric_name=False,
    )


@pytest.fixture()
def dataset(tmp_path: Path):
    fp_path = _write_parquet(tmp_path / "fp.parquet", list(dict.fromkeys(_SMILES)))
    ds_path = tmp_path / "ds"
    _write_dataset(ds_path, _SMILES)
    return tmp_path, fp_path, ds_path


def test_best_ckpt_written_after_val(dataset, tmp_path):
    """ModelCheckpoint writes last.ckpt and ≥1 metric checkpoint when val fires."""
    import pytorch_lightning as pl

    _, fp_path, ds_path = dataset
    ckpt_dir = tmp_path / "checkpoints"

    # check_val_every_n_epoch=None: val_check_interval becomes an absolute step count
    # (mirrors how the training config uses eval_every_n_steps).
    pl.Trainer(
        max_steps=4,
        accelerator="cpu",
        val_check_interval=1,
        check_val_every_n_epoch=None,
        callbacks=[_make_ckpt_callback(ckpt_dir)],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    ).fit(_make_module(fp_path), _make_dm(ds_path))

    assert (ckpt_dir / "last.ckpt").exists(), "last.ckpt not written"
    metric_ckpts = [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt"]
    assert len(metric_ckpts) >= 1, f"No metric checkpoint written; found: {list(ckpt_dir.iterdir())}"


def test_lightning_ckpt_resume(dataset, tmp_path):
    """A saved last.ckpt can be used to resume training without error and loss stays finite."""
    import pytorch_lightning as pl

    _, fp_path, ds_path = dataset
    ckpt_dir = tmp_path / "checkpoints"

    pl.Trainer(
        max_steps=4,
        accelerator="cpu",
        val_check_interval=1,
        check_val_every_n_epoch=None,
        callbacks=[_make_ckpt_callback(ckpt_dir)],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    ).fit(_make_module(fp_path), _make_dm(ds_path))

    last_ckpt = ckpt_dir / "last.ckpt"
    assert last_ckpt.exists()

    # Resume: fresh module + same DM, pick up from last.ckpt
    trainer2 = pl.Trainer(
        max_steps=8,
        accelerator="cpu",
        val_check_interval=1,
        check_val_every_n_epoch=None,
        callbacks=[_make_ckpt_callback(tmp_path / "checkpoints2")],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer2.fit(_make_module(fp_path), _make_dm(ds_path), ckpt_path=str(last_ckpt))

    loss = float(trainer2.logged_metrics.get("train_loss_step", float("nan")))
    assert np.isfinite(loss), f"Loss is not finite after resume: {loss}"
