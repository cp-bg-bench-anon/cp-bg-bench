"""
Self-contained inference checkpoint + Predictor for cp-bg-bench-model.

A trained ``CLIPModel`` is heavyweight: Lightning module wrapper, LoRA
adapters, training-time forward hooks, multi-positive loss state, etc.
``save_checkpoint`` strips that down to a portable, atomic ``.pt`` file
that holds only the image encoder (preprocessor + backbone + head) and,
optionally, the molecule encoder. ``Cp_bg_benchModelPredictor`` loads
that file on any device and exposes a small embed/predict surface so
downstream eval scripts don't re-import the training stack.

Usage::

    from cp_bg_bench_model import Cp_bg_benchModelPredictor, save_checkpoint

    save_checkpoint(model, "predictor.pt")  # at end of training

    predictor = Cp_bg_benchModelPredictor.load("predictor.pt", device="cuda")
    embs = predictor.predict_batch(crops_uint8)  # (B, D) float32, L2-normed
"""

from __future__ import annotations

import copy
import datetime
import warnings
from collections.abc import Iterator
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_bg_bench_model.models._export import _try_merge_lora_inplace


class _InferenceImageEncoder(nn.Module):
    """Preprocessor + image backbone + head, returning L2-normalized embeddings.

    Mirrors ``CLIPModel.forward``'s image branch so callers don't need
    ``ImageGpuPreprocess`` separately. Accepts uint8 ``(B, C, H, W)`` and
    resizes / casts to float internally.
    """

    def __init__(self, preprocess: nn.Module, encoder: nn.Module):
        super().__init__()
        self.preprocess = preprocess
        self.encoder = encoder

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        emb = self.encoder(x)
        return F.normalize(emb, p=2, dim=-1)


class _InferenceMoleculeEncoder(nn.Module):
    """Molecule backbone + head, returning L2-normalized embeddings."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @torch.inference_mode()
    def forward(self, mol: Any) -> torch.Tensor:
        emb = self.encoder(mol)
        return F.normalize(emb, p=2, dim=-1)


# All forward-related hook dicts that can carry non-picklable closures.
# Includes PyTorch 2.0+ attrs; getattr with None guard handles older versions.
_HOOK_ATTRS: tuple[str, ...] = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
)


def _strip_forward_hooks(module: nn.Module) -> dict[str, dict]:
    """Remove and return all forward hook dicts so deepcopy can pickle the module.

    PEFT/timm install non-picklable closures in ``_forward_hooks`` and
    ``_forward_pre_hooks`` during training. Returned mapping lets the caller
    restore them on the original module.
    """
    saved: dict[str, dict] = {}
    for name, mod in module.named_modules():
        for attr in _HOOK_ATTRS:
            hooks = getattr(mod, attr, None)
            if hooks:
                saved.setdefault(name, {})[attr] = dict(hooks)
                hooks.clear()
    return saved


def _restore_forward_hooks(module: nn.Module, saved: dict[str, dict]) -> None:
    for name, mod in module.named_modules():
        if name in saved:
            for attr, hook_dict in saved[name].items():
                getattr(mod, attr).update(hook_dict)


def _deepcopy_no_hooks(module: nn.Module) -> nn.Module:
    """Deepcopy a module after temporarily stripping non-picklable forward hooks."""
    saved = _strip_forward_hooks(module)
    try:
        return copy.deepcopy(module).eval()
    finally:
        _restore_forward_hooks(module, saved)


def _build_default_metadata(model: nn.Module) -> dict[str, Any]:
    """Extract provenance fields from the model at save time."""
    img_enc = getattr(model, "image_encoder", None)
    image_size = getattr(img_enc, "image_size", None)
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    # Use the stable config name (e.g. "dinov3") stored on the encoder's cfg
    # dataclass rather than the class name, which is an implementation detail.
    cfg = getattr(img_enc, "cfg", None)
    encoder_name = getattr(cfg, "name", None) or (type(img_enc).__name__ if img_enc is not None else None)
    return {
        "saved_at": datetime.datetime.now(timezone.utc).isoformat(),
        "embed_dim": getattr(img_enc, "embed_dim", None),
        "image_encoder_name": encoder_name,
        "in_channels": getattr(img_enc, "in_channels", None),
        "image_size": image_size,
    }


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    merge_lora: bool = True,
    include_molecule_encoder: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a portable inference checkpoint.

    Parameters
    ----------
    model
        A ``CLIPModel`` (or any module exposing ``image_encoder``,
        ``image_preprocess``, and optionally ``molecule_encoder``).
    path
        Destination ``.pt`` file. Parent directories are created.
    merge_lora
        If ``True``, merge LoRA adapter weights into the base layers
        before saving. Recommended — produces a smaller, dependency-free
        checkpoint that doesn't require PEFT at load time.
    include_molecule_encoder
        If ``True``, also save the molecule encoder so the predictor can
        embed SMILES. Default ``False`` (image-only — most eval paths).
    metadata
        Optional dict merged on top of the auto-extracted provenance fields
        (``saved_at``, ``embed_dim``, ``in_channels``, ``image_size``,
        ``image_encoder_name``). Caller-supplied keys take precedence.

    Returns
    -------
    Path
        The written checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Support both old (molecule_encoder) and new (perturbation_encoder) attribute names
    mol_enc = getattr(model, "perturbation_encoder", None) or getattr(model, "molecule_encoder", None)
    if not include_molecule_encoder and mol_enc is not None:
        if next(mol_enc.parameters(), None) is not None:
            warnings.warn(
                "save_checkpoint: molecule_encoder not saved. Pass include_molecule_encoder=True to include it.",
                stacklevel=2,
            )

    image_encoder = _deepcopy_no_hooks(model.image_encoder)
    image_preprocess = _deepcopy_no_hooks(model.image_preprocess)

    if merge_lora:
        _try_merge_lora_inplace(image_encoder)

    inference_image = _InferenceImageEncoder(image_preprocess, image_encoder)

    inference_molecule: _InferenceMoleculeEncoder | None = None
    if include_molecule_encoder and mol_enc is not None:
        molecule_encoder = _deepcopy_no_hooks(mol_enc)
        if merge_lora:
            _try_merge_lora_inplace(molecule_encoder)
        inference_molecule = _InferenceMoleculeEncoder(molecule_encoder)

    combined_metadata = _build_default_metadata(model)
    combined_metadata.update(metadata or {})

    checkpoint: dict[str, Any] = {
        "image_encoder": inference_image,
        "molecule_encoder": inference_molecule,
        "metadata": combined_metadata,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.rename(path)
    return path


class Cp_bg_benchModelPredictor:
    """Image (and optionally molecule) embedder loaded from a checkpoint.

    ``embed`` returns a ``torch.Tensor`` on the predictor's device.
    ``predict_batch`` / ``predict_stream`` return numpy for downstream
    persistence.

    Example::

        predictor = Cp_bg_benchModelPredictor.load("predictor.pt", device="cuda")
        embs = predictor.predict_batch(crops)            # (B, D) numpy
        for batch in predictor.predict_stream(it, 256):
            ...
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        molecule_encoder: nn.Module | None = None,
        device: str | torch.device = "cpu",
        metadata: dict[str, Any] | None = None,
    ):
        self.device = torch.device(device)
        self.image_encoder = image_encoder.to(self.device).eval()
        self.molecule_encoder = molecule_encoder.to(self.device).eval() if molecule_encoder is not None else None
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Cp_bg_benchModelPredictor:
        """Load from a ``save_checkpoint`` output."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            image_encoder=ckpt["image_encoder"],
            molecule_encoder=ckpt.get("molecule_encoder"),
            device=device,
            metadata=ckpt.get("metadata", {}),
        )

    def _to_image_tensor(self, crops: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input to ``(B, C, H, W)`` tensor on device. dtype preserved."""
        if isinstance(crops, np.ndarray):
            crops = torch.from_numpy(crops.copy())
        if crops.ndim == 3:
            crops = crops.unsqueeze(0)
        return crops.to(self.device, non_blocking=True)

    @torch.inference_mode()
    def embed(self, crops: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode image crops to L2-normed embeddings.

        Accepts ``(C, H, W)`` or ``(B, C, H, W)``, uint8 or float. The
        baked-in preprocessor handles dtype/scale/resize.
        """
        return self.image_encoder(self._to_image_tensor(crops))

    @torch.inference_mode()
    def predict_batch(self, crops: np.ndarray | torch.Tensor) -> np.ndarray:
        """``embed`` + ``.cpu().numpy()`` shortcut for streaming pipelines."""
        return self.embed(crops).cpu().numpy()

    @torch.inference_mode()
    def predict(self, crop: np.ndarray | torch.Tensor) -> np.ndarray:
        """Single-tile convenience wrapper. Returns ``(D,)``."""
        return self.predict_batch(crop)[0]

    @torch.inference_mode()
    def predict_stream(
        self,
        crop_iterator: Iterator[np.ndarray | torch.Tensor],
        batch_size: int = 256,
    ) -> Iterator[np.ndarray]:
        """Yield ``(B, D)`` numpy batches from an iterator of single tiles."""
        buffer: list = []
        for crop in crop_iterator:
            buffer.append(crop)
            if len(buffer) >= batch_size:
                yield self.predict_batch(np.stack(buffer))
                buffer.clear()
        if buffer:
            yield self.predict_batch(np.stack(buffer))

    @torch.inference_mode()
    def embed_molecule(self, mol: Any) -> torch.Tensor:
        """Encode a molecule input to an L2-normed embedding.

        Input shape depends on the encoder: ``list[str]`` of SMILES for
        ECFP/WHIMF/ChemBERTa; ``(B, in_dim)`` float tensor for precomputed.
        """
        if self.molecule_encoder is None:
            raise RuntimeError(
                "This checkpoint was saved without a molecule encoder. Re-save with include_molecule_encoder=True."
            )
        if isinstance(mol, torch.Tensor):
            mol = mol.to(self.device, non_blocking=True)
        return self.molecule_encoder(mol)

    @property
    def has_molecule_encoder(self) -> bool:
        return self.molecule_encoder is not None

    def __repr__(self) -> str:
        parts = [f"device={self.device}", f"has_mol={self.has_molecule_encoder}"]
        if self.metadata:
            parts.append(f"metadata_keys={sorted(self.metadata)}")
        return f"Cp_bg_benchModelPredictor({', '.join(parts)})"


__all__: list[str] = ["save_checkpoint", "Cp_bg_benchModelPredictor"]
