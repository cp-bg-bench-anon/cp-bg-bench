"""Regression tests for epoch-end cadence.

Proves that after a short epoch-based run:
- `train_R@...` metrics are logged once at end of train epoch.
- `val_R@...` metrics are logged once at end of val epoch.
- The train-side accumulation buffers are cleared between epochs (no
  unbounded growth across epochs).
- The cosine LR schedule is built from `trainer.estimated_stepping_batches`
  (so total_steps tracks dataset size × max_epochs, not a config constant).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from datasets import Dataset
from omegaconf import OmegaConf

from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.datamodule import ImageMoleculeDataModule
from cp_bg_bench_model.models.models import PretrainModule

_EMBED_DIM = 32
_HW = 32
_IN_CHANNELS = 5
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
            "model": {
                "optimizer": {"lr_head": 3e-4, "lr_lora": 1e-4, "lr_temp": 1e-5},
                "lr_schedule": {"warmup_frac": 0.1, "num_cycles": 0.5},
            },
            "logging": {"train_loss_every_n_steps": 1},
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
        split_by_column=DatasetEnum.PERTURBATION.value,
        split_stratify_by=None,
        stratify_by_column=None,
        test_frac=0.0,
        val_frac=0.5,
        perturbations_per_batch=None,
        augment=False,
    )


@pytest.fixture(scope="module")
def fitted_trainer(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("epoch_end_metrics")
    fp_path = _write_parquet(tmp_path / "fp.parquet", list(dict.fromkeys(_SMILES)))
    ds_path = tmp_path / "ds"
    _write_dataset(ds_path, _SMILES)

    module = _make_module(fp_path)
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(module, _make_dm(ds_path))
    return module, trainer


def test_train_epoch_end_logs_train_retrieval(fitted_trainer):
    """`on_train_epoch_end` logs full-set retrieval with `train_` prefix."""
    module, trainer = fitted_trainer
    logged = set(trainer.logged_metrics)
    # At least one train retrieval metric must be present; check the headline one.
    assert "train_R@1_I2P_macro" in logged, sorted(logged)
    assert "train_R@5_I2P_macro" in logged
    assert "train_R@10_I2P_macro" in logged
    val_ref = float(trainer.logged_metrics["train_R@1_I2P_macro"])
    assert np.isfinite(val_ref) and 0.0 <= val_ref <= 1.0


def test_val_epoch_end_logs_val_retrieval(fitted_trainer):
    """`on_validation_epoch_end` still fires with `val_` prefix."""
    _, trainer = fitted_trainer
    logged = set(trainer.logged_metrics)
    assert "val_R@1_I2P_macro" in logged, sorted(logged)
    assert "val_R@5_I2P_macro" in logged
    assert "val_R@10_I2P_macro" in logged


def test_train_buffers_cleared_between_epochs(fitted_trainer):
    """After fit, the accumulated per-epoch buffers are empty (not leaked)."""
    module, _ = fitted_trainer
    assert module._train_img_embs == []
    assert module._train_pert_embs == []
    assert module._train_compound_ids == []


def test_lr_schedule_uses_estimated_stepping_batches(fitted_trainer):
    """The cosine schedule total = trainer.estimated_stepping_batches."""
    module, trainer = fitted_trainer
    schedulers = module.lr_schedulers()
    assert schedulers is not None, "configure_optimizers did not return a scheduler"
    scheduler = schedulers if not isinstance(schedulers, list) else schedulers[0]
    # LambdaLR stores the lambdas; ours is a functools.partial carrying kwargs.
    lr_lambda = scheduler.lr_lambdas[0]
    assert hasattr(lr_lambda, "keywords"), "expected a functools.partial lambda"
    total = int(trainer.estimated_stepping_batches)
    assert lr_lambda.keywords["num_training_steps"] == total
    assert lr_lambda.keywords["num_warmup_steps"] == max(1, int(0.1 * total))
