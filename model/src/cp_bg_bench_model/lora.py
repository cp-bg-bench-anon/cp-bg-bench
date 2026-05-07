from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from peft.tuners.lora import LoraConfig as _PeftLoraConfig

# Use the generic PEFT LoraModel (vision-safe; no text wrappers that inject input_ids)
from peft.tuners.lora import LoraModel as _PeftLoraModel

try:
    # Newer PEFT
    from peft.utils.config import TaskType as _TaskType
except Exception:
    # Older PEFT
    from peft import TaskType as _TaskType

from cp_bg_bench_model._utils import freeze_batch_norm_2d


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    bias: Literal["none", "all", "lora_only"] = "none"
    targets: list[str] | None = None
    task_type: Literal["FEATURE_EXTRACTION", "SEQ_CLS", "TOKEN_CLS", "CAUSAL_LM"] = "FEATURE_EXTRACTION"


def _set_trainable_for_lora_only(module: nn.Module) -> None:
    """Freeze everything, then enable only LoRA adapter params."""
    for _, p in module.named_parameters():
        p.requires_grad = False
    for name, p in module.named_parameters():
        if "lora_" in name or "lora" in name:
            p.requires_grad = True


def apply_lora_if_enabled(
    trunk: nn.Module,
    config: LoraConfig | None,
    *,
    default_targets: Sequence[str] | None = None,
    freeze_if_disabled: bool = True,
    freeze_bn: bool = True,
) -> tuple[nn.Module, bool]:
    """
    Wrap `trunk` with PEFT LoRA if enabled; otherwise optionally freeze.
    Uses the generic LoraModel to avoid task-specific wrappers that inject `input_ids`.
    Returns (possibly-wrapped trunk, using_lora_flag).
    """
    if not config or not config.enabled:
        if freeze_if_disabled:
            for p in trunk.parameters():
                p.requires_grad = False
            if freeze_bn:
                freeze_batch_norm_2d(trunk)
        return trunk, False

    targets = list(config.targets) if config.targets else list(default_targets or [])

    # Safely map task type; default to FEATURE_EXTRACTION if unsure
    tt_name = config.task_type
    try:
        task_type = getattr(_TaskType, tt_name)
    except Exception:
        task_type = getattr(_TaskType, "FEATURE_EXTRACTION", "FEATURE_EXTRACTION")

    peft_cfg = _PeftLoraConfig(
        r=int(config.r),
        lora_alpha=int(config.alpha),
        lora_dropout=float(config.dropout),
        bias=str(config.bias),
        target_modules=targets if targets else None,
        task_type=task_type,
    )

    wrapped = _PeftLoraModel(trunk, peft_cfg, adapter_name="default")

    # Train only LoRA params
    _set_trainable_for_lora_only(wrapped)

    # Optionally freeze BN stats/params inside CNN backbones (harmless for ViT)
    if freeze_bn:
        freeze_batch_norm_2d(wrapped)

    return wrapped, True


def collect_lora_params(module: nn.Module) -> list[nn.Parameter]:
    """Heuristic: gather trainable parameters created by PEFT that include 'lora' in the name."""
    params: list[nn.Parameter] = []
    for name, p in module.named_parameters():
        if p.requires_grad and ("lora_" in name or "lora" in name):
            params.append(p)
    return params


__all__ = ["LoraConfig", "apply_lora_if_enabled", "collect_lora_params"]
