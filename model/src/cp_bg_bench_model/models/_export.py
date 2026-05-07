from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning.core.module import LightningModule

from cp_bg_bench_model.encoders.image_encoders import (
    DinoV3ViTConfig,
    ImageEncoderConfig,
    OpenPhenomConfig,
    SubCellConfig,
    TimmCNNConfig,
    build_encoder_from_config,
)
from cp_bg_bench_model.lora import LoraConfig

_CONFIG_CLASS_REGISTRY: dict[str, type[ImageEncoderConfig]] = {
    "DinoV3ViTConfig": DinoV3ViTConfig,
    "OpenPhenomConfig": OpenPhenomConfig,
    "SubCellConfig": SubCellConfig,
    "TimmCNNConfig": TimmCNNConfig,
}


class ImageEncoderWithHead(nn.Module):
    """
    Wraps an image encoder (which already includes its projection/head).
    Forward is encoder(x). LoRA modules are preserved unless merged.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = copy.deepcopy(encoder).eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _try_merge_lora_inplace(module: nn.Module) -> None:
    """
    Best-effort LoRA merge that supports common method names.
    Safe to call even if no LoRA is present.
    """
    merge_names = ("merge_lora_", "merge_and_unload", "merge_adapter", "merge_weights_")
    for m in module.modules():
        for name in merge_names:
            fn = getattr(m, name, None)
            if callable(fn):
                try:
                    fn()  # in-place merge
                except Exception:
                    pass


def _find_attr(obj: object, candidates: Iterable[str]) -> str:
    for n in candidates:
        if hasattr(obj, n):
            return n
    raise AttributeError(f"None of {list(candidates)} found on {type(obj).__name__}")


def _encoder_cfg_to_dict(cfg: ImageEncoderConfig) -> dict[str, Any]:
    """Serialize an ImageEncoderConfig to a JSON-safe dict."""
    return {
        "encoder_config_class": type(cfg).__name__,
        "encoder_config": dataclasses.asdict(cfg),
    }


def _encoder_cfg_from_dict(d: dict[str, Any]) -> ImageEncoderConfig:
    """Rehydrate an ImageEncoderConfig from its JSON-roundtripped dict.

    Handles the nested ``LoraConfig`` and restores tuple fields that JSON
    flattens to lists (frozen dataclasses don't enforce types at runtime,
    but the encoder code paths expect tuples in several places).
    """
    cls_name = d["encoder_config_class"]
    if cls_name not in _CONFIG_CLASS_REGISTRY:
        raise ValueError(
            f"Unknown encoder config class {cls_name!r}. "
            f"Known: {sorted(_CONFIG_CLASS_REGISTRY)}"
        )
    cls = _CONFIG_CLASS_REGISTRY[cls_name]
    raw = dict(d["encoder_config"])

    lora = raw.get("lora")
    if isinstance(lora, dict):
        raw["lora"] = LoraConfig(**lora)

    for f in dataclasses.fields(cls):
        v = raw.get(f.name)
        # With `from __future__ import annotations`, f.type is a string.
        if v is not None and isinstance(v, list) and "tuple" in str(f.type):
            raw[f.name] = tuple(v)

    return cls(**raw)


def export_image_encoder_with_head(
    lit_model: LightningModule,
    *,
    out_dir: str | Path,
    file_stem: str = "image_encoder_with_head",
    merge_lora: bool = False,
    encoder_attr_candidates: list[str] = ["image_encoder", "encoder", "backbone", "img_encoder"],
    container_attr_candidates: list[str] = ["model", "module"],  # PretrainModule.model holds CLIPModel
) -> dict[str, str]:
    """
    Extracts lit_model.[model].image_encoder, optionally merges LoRA, and saves
    three artifacts so consumers can choose a load path:
      - <out_dir>/<file_stem>.pt   pickled ``ImageEncoderWithHead`` (legacy)
      - <out_dir>/<file_stem>.pth  state_dict of the wrapper (new primary)
      - <out_dir>/<file_stem>.json encoder config sidecar (new primary)

    Load via :func:`load_image_encoder_with_head`, which prefers the
    ``.pth + .json`` pair and falls back to the pickled ``.pt`` for
    checkpoints produced before the sidecar was introduced.  The sidecar
    is required for HF ``trust_remote_code`` encoders (e.g. OpenPhenom),
    whose pickled graph references a dynamically-created
    ``transformers_modules`` namespace that does not exist in a fresh
    ``torch.load`` process.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = lit_model
    try:
        # Many Lightning setups wrap the actual nets under .model or .module
        container = getattr(lit_model, _find_attr(lit_model, container_attr_candidates))
    except AttributeError:
        container = root

    enc_name = _find_attr(container, encoder_attr_candidates)
    encoder = getattr(container, enc_name)

    enc = copy.deepcopy(encoder).eval()
    if merge_lora:
        _try_merge_lora_inplace(enc)

    exported = ImageEncoderWithHead(enc)

    module_path = out / f"{file_stem}.pt"
    sd_path = out / f"{file_stem}.pth"
    cfg_path = out / f"{file_stem}.json"

    torch.save(exported, str(module_path))
    torch.save(exported.state_dict(), str(sd_path))

    # Sidecar is only writable for encoders with an ImageEncoderConfig; skip
    # silently on any exotic encoder so the legacy .pt export still lands.
    if hasattr(enc, "cfg") and isinstance(enc.cfg, ImageEncoderConfig):
        with cfg_path.open("w") as fh:
            json.dump(_encoder_cfg_to_dict(enc.cfg), fh, indent=2, sort_keys=True)
        return {"module": str(module_path), "state_dict": str(sd_path), "config": str(cfg_path)}
    return {"module": str(module_path), "state_dict": str(sd_path)}


def load_image_encoder_with_head(path: str | Path) -> ImageEncoderWithHead:
    """
    Reconstruct an ``ImageEncoderWithHead`` from a training export.

    Prefers the config-sidecar path (``.pth`` state_dict + ``.json`` config)
    because it is architecture-agnostic and immune to the
    ``transformers_modules`` problem that breaks HF ``trust_remote_code``
    pickles.  Falls back to ``torch.load`` on the legacy ``.pt`` pickle for
    checkpoints exported before the sidecar existed — those pre-date
    OpenPhenom support and loaded fine under the old path.

    Accepts any of the file paths (``.pt``/``.pth``/``.json``) or the shared
    stem; sibling artifacts are resolved by extension.
    """
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in {".pt", ".pth", ".json"} else p
    sd_path = stem.with_suffix(".pth")
    cfg_path = stem.with_suffix(".json")
    module_path = stem.with_suffix(".pt")

    if cfg_path.is_file() and sd_path.is_file():
        with cfg_path.open() as fh:
            cfg_dict = json.load(fh)
        cfg = _encoder_cfg_from_dict(cfg_dict)
        encoder = build_encoder_from_config(cfg)
        wrapper = ImageEncoderWithHead(encoder)
        state = torch.load(str(sd_path), map_location="cpu", weights_only=True)
        wrapper.load_state_dict(state, strict=True)
        return wrapper.eval()

    if module_path.is_file():
        loaded = torch.load(str(module_path), map_location="cpu", weights_only=False)
        if not isinstance(loaded, ImageEncoderWithHead):
            raise TypeError(
                f"Legacy pickle at {module_path} is {type(loaded).__name__}, "
                "expected ImageEncoderWithHead."
            )
        return loaded.eval()

    raise FileNotFoundError(
        f"No encoder export found at stem {stem}. Expected either "
        f"{cfg_path.name} + {sd_path.name} (preferred) or {module_path.name} (legacy)."
    )
