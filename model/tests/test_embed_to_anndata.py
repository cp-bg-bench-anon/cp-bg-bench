"""End-to-end test for ``scripts/embed_to_anndata.py``.

Builds a tiny HF dataset containing both the ``cell`` image column and a
``mask`` column with embedded NULL bytes (``large_binary``).  Raw binary
columns must be excluded from ``obs`` — h5py's VLEN strings reject
embedded NULLs, so leaving them in would crash ``write_h5ad``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from datasets import Dataset, Features, Value

from cp_bg_bench_model.models._export import ImageEncoderWithHead

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from embed_to_anndata import embed_to_anndata  # noqa: E402

_C, _HW, _D = 6, 32, 16


class _LinearEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def _build_dataset(n: int) -> Dataset:
    rng = np.random.default_rng(0)
    features = Features(
        {
            "cell": Value("large_binary"),
            "mask": Value("large_binary"),
            "plate": Value("string"),
            "perturbation": Value("string"),
        }
    )
    rows = [
        {
            "cell": rng.integers(0, 256, size=_C * _HW * _HW, dtype=np.uint8).tobytes(),
            "mask": b"\x00\x01\x02" * 100,
            "plate": f"p{i % 2}",
            "perturbation": f"g{i % 3}",
        }
        for i in range(n)
    ]
    return Dataset.from_list(rows, features=features)


def test_binary_cols_excluded_from_obs(tmp_path: Path):
    torch.manual_seed(0)
    encoder = ImageEncoderWithHead(_LinearEncoder(_C * _HW * _HW, _D)).eval()
    ckpt = tmp_path / "imgenc.pt"
    torch.save(encoder, ckpt)

    ds_path = tmp_path / "ds"
    _build_dataset(n=8).save_to_disk(str(ds_path))

    out = embed_to_anndata(
        config_id="99_Test_Enc_Pert_C",
        checkpoint=ckpt,
        dataset_path=ds_path,
        output_dir=tmp_path,
        channels=_C,
        image_size=_HW,
        batch_size=4,
        device="cpu",
    )
    assert out.exists()

    adata = ad.read_h5ad(out)
    assert adata.shape == (8, _D)
    assert "cell" not in adata.obs.columns
    assert "mask" not in adata.obs.columns
    assert {"plate", "perturbation"} <= set(adata.obs.columns)
    assert not np.isnan(adata.X).any()
    np.testing.assert_allclose(np.linalg.norm(adata.X, axis=1), np.ones(8), atol=1e-5)
