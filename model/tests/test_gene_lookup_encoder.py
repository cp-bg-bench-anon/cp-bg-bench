"""Tests for GeneLookupEncoder: config, lookup, error handling, CLIPModel integration, roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cp_bg_bench_model import save_checkpoint
from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.encoders.molecule_encoders import (
    GeneLookupConfig,
    GeneLookupEncoder,
    MoleculeEncoderRegistry,
)
from cp_bg_bench_model.models import CLIPModel

_EMBED_DIM = 64
_GENES = ["BRCA1", "TP53", "EGFR"]
_EMB_DIM_IN = 128
_HW = 32
_IN_CHANNELS = 6


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_parquet(path: Path, genes: list[str], dim: int = _EMB_DIM_IN) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "gene_symbol": genes,
            f"esm2_{dim}": [rng.random(dim).astype(np.float32).tolist() for _ in genes],
        }
    )
    df.to_parquet(path, index=False)
    return path


# ── config validation ─────────────────────────────────────────────────────────


def test_gene_lookup_config_fields():
    cfg = GeneLookupConfig(
        name="gene_lookup",
        embed_dim=128,
        fingerprint_path="/some/path.parquet",
        embedding_column="esm2_1280",
    )
    assert cfg.fingerprint_path == "/some/path.parquet"
    assert cfg.embedding_column == "esm2_1280"
    assert cfg.embed_dim == 128
    assert cfg.projection_head == "mlp"


def test_gene_lookup_empty_path_raises():
    cfg = GeneLookupConfig(name="gene_lookup", embed_dim=64, fingerprint_path="", embedding_column="esm2_1280")
    with pytest.raises(ValueError, match="fingerprint_path"):
        GeneLookupEncoder(cfg)


def test_gene_lookup_empty_column_raises(tmp_path: Path):
    fp = _write_parquet(tmp_path / "fp.parquet", _GENES)
    cfg = GeneLookupConfig(name="gene_lookup", embed_dim=64, fingerprint_path=str(fp), embedding_column="")
    with pytest.raises(ValueError, match="embedding_column"):
        GeneLookupEncoder(cfg)


def test_gene_lookup_missing_file_raises(tmp_path: Path):
    cfg = GeneLookupConfig(
        name="gene_lookup",
        embed_dim=64,
        fingerprint_path=str(tmp_path / "nope.parquet"),
        embedding_column="esm2_1280",
    )
    with pytest.raises(FileNotFoundError):
        GeneLookupEncoder(cfg)


# ── parquet validation ────────────────────────────────────────────────────────


def test_load_parquet_small_dim_accepted(tmp_path: Path):
    path = tmp_path / "small_dim.parquet"
    df = pd.DataFrame({"gene_symbol": _GENES, "esm2_1280": [[1.0, 2.0]] * len(_GENES)})
    df.to_parquet(path, index=False)
    # Shape (N, 2) is valid — loader should accept it without error
    cfg = GeneLookupConfig(name="gene_lookup", embed_dim=64, fingerprint_path=str(path), embedding_column="esm2_1280")
    enc = GeneLookupEncoder(cfg)
    assert enc._embedding_dim == 2


def test_load_parquet_scalar_column_raises(tmp_path: Path):
    path = tmp_path / "scalar.parquet"
    # Embedding column contains plain floats (not lists) → fps.ndim == 1 → ValueError
    df = pd.DataFrame({"gene_symbol": _GENES, "esm2_1280": [1.0, 2.0, 3.0]})
    df.to_parquet(path, index=False)
    cfg = GeneLookupConfig(name="gene_lookup", embed_dim=64, fingerprint_path=str(path), embedding_column="esm2_1280")
    with pytest.raises(ValueError):
        GeneLookupEncoder(cfg)


def test_load_parquet_validates_finite(tmp_path: Path):
    path = tmp_path / "nan.parquet"
    rng = np.random.default_rng(0)
    rows = [rng.random(128).astype(np.float32).tolist() for _ in _GENES]
    rows[0][0] = float("nan")
    df = pd.DataFrame({"gene_symbol": _GENES, "esm2_128": rows})
    df.to_parquet(path, index=False)
    cfg = GeneLookupConfig(name="gene_lookup", embed_dim=64, fingerprint_path=str(path), embedding_column="esm2_128")
    with pytest.raises(ValueError, match="NaN/Inf"):
        GeneLookupEncoder(cfg)


# ── encode ────────────────────────────────────────────────────────────────────


def test_encode_returns_correct_shape(tmp_path: Path):
    fp = _write_parquet(tmp_path / "fp.parquet", _GENES)
    cfg = GeneLookupConfig(
        name="gene_lookup", embed_dim=_EMBED_DIM, fingerprint_path=str(fp), embedding_column=f"esm2_{_EMB_DIM_IN}"
    )
    enc = GeneLookupEncoder(cfg).eval()
    with torch.no_grad():
        out = enc(_GENES)
    assert out.shape == (len(_GENES), _EMBED_DIM)
    assert torch.isfinite(out).all()


def test_missing_gene_raises_key_error(tmp_path: Path):
    fp = _write_parquet(tmp_path / "fp.parquet", _GENES)
    cfg = GeneLookupConfig(
        name="gene_lookup", embed_dim=_EMBED_DIM, fingerprint_path=str(fp), embedding_column=f"esm2_{_EMB_DIM_IN}"
    )
    enc = GeneLookupEncoder(cfg)
    with pytest.raises(KeyError, match="not in embedding lookup"):
        enc(["UNKNOWN_GENE_XYZ"])


# ── registry ──────────────────────────────────────────────────────────────────


def test_registry_lookup(tmp_path: Path):
    fp = _write_parquet(tmp_path / "fp.parquet", _GENES[:1])
    enc = MoleculeEncoderRegistry.build_from_name(
        "gene_lookup",
        embed_dim=_EMBED_DIM,
        fingerprint_path=str(fp),
        embedding_column=f"esm2_{_EMB_DIM_IN}",
    )
    assert isinstance(enc, GeneLookupEncoder)
    assert enc.embed_dim == _EMBED_DIM


# ── CLIPModel integration ─────────────────────────────────────────────────────


def test_clip_model_forward(tmp_path: Path):
    fp = _write_parquet(tmp_path / "fp.parquet", _GENES)
    torch.manual_seed(0)
    model = CLIPModel(
        embed_dim=_EMBED_DIM,
        image_encoder_name="densenet",
        perturbation_encoder_name="gene_lookup",
        loss="CLIP",
        image_size=_HW,
        in_channels=_IN_CHANNELS,
        freeze_backbone_when_no_lora=False,
        fingerprint_path=str(fp),
        embedding_column=f"esm2_{_EMB_DIM_IN}",
    ).eval()

    B = len(_GENES)
    img = torch.randint(0, 256, (B, _IN_CHANNELS, _HW, _HW), dtype=torch.uint8)
    batch = {DatasetEnum.IMG: img, DatasetEnum.PERTURBATION: _GENES}
    with torch.no_grad():
        img_emb, mol_emb = model(batch)
    assert img_emb.shape == (B, _EMBED_DIM)
    assert mol_emb.shape == (B, _EMBED_DIM)
    assert torch.isfinite(img_emb).all()
    assert torch.isfinite(mol_emb).all()


# ── checkpoint roundtrip ──────────────────────────────────────────────────────


def test_save_checkpoint_roundtrip(tmp_path: Path):
    from cp_bg_bench_model import Cp_bg_benchModelPredictor

    fp = _write_parquet(tmp_path / "fp.parquet", _GENES)
    torch.manual_seed(1)
    model = CLIPModel(
        embed_dim=_EMBED_DIM,
        image_encoder_name="densenet",
        perturbation_encoder_name="gene_lookup",
        loss="CLIP",
        image_size=_HW,
        in_channels=_IN_CHANNELS,
        freeze_backbone_when_no_lora=False,
        fingerprint_path=str(fp),
        embedding_column=f"esm2_{_EMB_DIM_IN}",
    ).eval()

    ckpt = save_checkpoint(model, tmp_path / "predictor.pt", merge_lora=False)
    predictor = Cp_bg_benchModelPredictor.load(ckpt, device="cpu")

    img = torch.randint(0, 256, (2, _IN_CHANNELS, _HW, _HW), dtype=torch.uint8)
    with torch.no_grad():
        out = predictor.predict_batch(img)

    assert out.shape == (2, _EMBED_DIM)
    assert np.isfinite(out).all()
    norms = np.linalg.norm(out.astype(np.float32), axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-2)
