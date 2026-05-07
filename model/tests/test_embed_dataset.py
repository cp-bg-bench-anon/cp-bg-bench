"""End-to-end test for ``scripts/embed_dataset.py``.

Builds a tiny synthetic HF dataset with the same ``cell`` (raw uint8
buffer) layout the trainer expects, saves a real predictor checkpoint,
runs the embed script, and checks the parquet output shape + contents.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.filterwarnings("ignore:molecule_encoder not saved:UserWarning")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from cp_bg_bench_model import save_checkpoint
from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.models import CLIPModel

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from embed_dataset import embed_dataset  # noqa: E402

_C, _HW = 5, 64


def _build_dataset(n_rows: int, channels: int = _C, hw: int = _HW) -> Dataset:
    rng = np.random.default_rng(0)
    buffers = [rng.integers(0, 256, size=channels * hw * hw, dtype=np.uint8).tobytes() for _ in range(n_rows)]
    return Dataset.from_dict({DatasetEnum.IMG.value: buffers})


@pytest.fixture()
def clip_checkpoint(tmp_path: Path):
    """Returns (ckpt_path, embed_dim) for a small DenseNet CLIPModel."""
    torch.manual_seed(0)
    model = CLIPModel(
        embed_dim=128,
        image_encoder_name="densenet",
        perturbation_encoder_name="precomputed",
        loss="CLIP",
        image_size=_HW,
        in_channels=_C,
        precomputed_in_dim=64,
        freeze_backbone_when_no_lora=False,
    ).eval()
    ckpt = save_checkpoint(model, tmp_path / "predictor.pt", merge_lora=False)
    return ckpt, 128


def test_embed_dataset_writes_parquet(tmp_path: Path, clip_checkpoint):
    ckpt, embed_dim = clip_checkpoint
    n = 16
    ds_path = tmp_path / "ds"
    _build_dataset(n).save_to_disk(str(ds_path))

    out = embed_dataset(
        checkpoint=ckpt,
        dataset_path=ds_path,
        output=tmp_path / "embeddings.parquet",
        batch_size=8,
        device="cpu",
    )

    df = pd.read_parquet(out)
    assert len(df) == n
    assert list(df["sample_id"]) == list(range(n))
    arr = np.stack(df["embedding"].to_list())
    assert arr.shape == (n, embed_dim)
    assert arr.dtype == np.float16
    assert np.isfinite(arr).all()
    norms = np.linalg.norm(arr.astype(np.float32), axis=1)
    np.testing.assert_allclose(norms, np.ones(n), atol=1e-2)


def test_embed_dataset_metadata_autoinference(tmp_path: Path, clip_checkpoint):
    """channels and image_size are inferred from checkpoint metadata when omitted."""
    ckpt, embed_dim = clip_checkpoint
    n = 8
    ds_path = tmp_path / "ds"
    _build_dataset(n).save_to_disk(str(ds_path))

    # Do NOT pass channels or image_size — must be inferred from metadata.
    out = embed_dataset(
        checkpoint=ckpt,
        dataset_path=ds_path,
        output=tmp_path / "embeddings.parquet",
        batch_size=4,
        device="cpu",
    )
    df = pd.read_parquet(out)
    assert len(df) == n
    assert np.stack(df["embedding"].to_list()).shape == (n, embed_dim)


def test_embed_dataset_empty(tmp_path: Path, clip_checkpoint):
    """embed_dataset on a zero-row dataset writes an empty parquet without crashing.

    Zero-row HF datasets can't be round-tripped through save_to_disk (library
    limitation: 0-shard Arrow files fail on load). We patch load_from_disk to
    return an in-memory empty dataset instead.
    """
    from unittest.mock import patch

    ckpt, _ = clip_checkpoint
    empty_ds = _build_dataset(0)

    with patch("embed_dataset.load_from_disk", return_value=empty_ds):
        out = embed_dataset(
            checkpoint=ckpt,
            dataset_path=tmp_path / "ds",
            output=tmp_path / "embeddings.parquet",
            channels=_C,
            image_size=_HW,
            device="cpu",
        )
    df = pd.read_parquet(out)
    assert len(df) == 0
