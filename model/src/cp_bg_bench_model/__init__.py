from __future__ import annotations

from cp_bg_bench_model.checkpoint import Cp_bg_benchModelPredictor, save_checkpoint
from cp_bg_bench_model.models import CLIPModel, PretrainModule

__version__ = "0.2.0"

__all__ = [
    "CLIPModel",
    "Cp_bg_benchModelPredictor",
    "PretrainModule",
    "save_checkpoint",
]
