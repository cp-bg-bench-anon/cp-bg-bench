"""
Minimal loader for SubCell checkpoints (no SubCellPortable dependency).
Weights are fetched from the CZI public S3 bucket via HTTPS.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Literal

import torch

from cp_bg_bench_model.encoders._subcell_vit import ViTPoolClassifier

logger = logging.getLogger(__name__)

Variant = Literal["er_dna_protein", "all_channels"]
CheckpointName = Literal["mae_cells_prots_pool", "vit_prots_pool"]

_S3_BASE = "https://czi-subcell-public.s3.amazonaws.com/models"

# (url, sha256) per checkpoint key.
# SHA-256: populate by running `sha256sum <cached_file>` after a trusted first download.
# None disables verification for that entry (logs a warning).
_CHECKPOINTS: dict[tuple[str, str], tuple[str, str | None]] = {
    ("er_dna_protein", "mae_cells_prots_pool"): (f"{_S3_BASE}/ER-DNA-Protein_MAE-CellS-ProtS-Pool.pth", None),
    ("er_dna_protein", "vit_prots_pool"): (f"{_S3_BASE}/ER-DNA-Protein_ViT-ProtS-Pool.pth", None),
    ("all_channels", "mae_cells_prots_pool"): (f"{_S3_BASE}/all_channels_MAE-CellS-ProtS-Pool.pth", None),
    ("all_channels", "vit_prots_pool"): (f"{_S3_BASE}/all_channels_ViT-ProtS-Pool.pth", None),
}

# ViTConfig params from SubCellPortable models/*/mae_contrast_supcon_model/model_config.yaml
_VIT_BASE_CONFIG = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "image_size": 448,
    "patch_size": 16,
    "qkv_bias": True,
}

_MODEL_CONFIGS: dict[str, dict] = {
    "er_dna_protein": {
        "vit_model": {**_VIT_BASE_CONFIG, "num_channels": 3},
        "pool_model": {"dim": 768, "int_dim": 512, "num_heads": 2},
        "num_classes": 31,
    },
    "all_channels": {
        "vit_model": {**_VIT_BASE_CONFIG, "num_channels": 4},
        "pool_model": {"dim": 768, "int_dim": 512, "num_heads": 2},
        "num_classes": 31,
    },
}

# out_dim = hidden_size * num_heads = 768 * 2 = 1536
SUBCELL_BASE_DIM = 1536


def _resolve(
    variant: str, checkpoint_name: str, checkpoint_url: str | None
) -> tuple[str, str | None]:
    """Return (url, sha256) for the requested checkpoint."""
    if checkpoint_url is not None:
        return checkpoint_url, None  # custom URL — no registered hash
    key = (variant, checkpoint_name)
    if key not in _CHECKPOINTS:
        raise KeyError(
            f"Unknown SubCell (variant={variant!r}, checkpoint={checkpoint_name!r}). Known: {list(_CHECKPOINTS)}"
        )
    return _CHECKPOINTS[key]


def _local_path(url: str, cache_dir: str) -> Path:
    effective = os.environ.get("SUBCELL_CACHE_DIR", cache_dir)
    cache = Path(os.path.expanduser(effective))
    cache.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    return cache / filename


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hash(path: Path, expected: str | None) -> None:
    if expected is None:
        logger.warning(
            "No expected SHA-256 registered for %s — skipping integrity check. "
            "Populate _CHECKPOINTS in _subcell_loader.py after a trusted download.",
            path.name,
        )
        return
    actual = _sha256(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SubCell checkpoint SHA-256 mismatch for {path.name}.\n"
            f"  expected: {expected}\n"
            f"  got:      {actual}\n"
            "The cached file has been removed. Re-run to re-download."
        )


def _download_if_needed(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info(f"SubCell checkpoint already cached at {dest}")
        return
    logger.info(f"Downloading SubCell checkpoint from {url} → {dest}")
    torch.hub.download_url_to_file(url, str(dest), progress=True)


def load_subcell_trunk(
    variant: str,
    checkpoint_name: str,
    cache_dir: str,
    checkpoint_url: str | None = None,
    device: str = "cpu",
) -> tuple[ViTPoolClassifier, int]:
    """Build and return a SubCell ViTPoolClassifier with loaded weights and base_dim."""
    if variant not in _MODEL_CONFIGS:
        raise KeyError(f"Unknown SubCell variant {variant!r}. Known: {list(_MODEL_CONFIGS)}")

    url, expected_hash = _resolve(variant, checkpoint_name, checkpoint_url)
    local = _local_path(url, cache_dir)
    _download_if_needed(url, local)
    _verify_hash(local, expected_hash)

    model_cfg = _MODEL_CONFIGS[variant]
    model = ViTPoolClassifier(model_cfg)
    model.load_encoder_weights(str(local), device=device)
    return model, SUBCELL_BASE_DIM
