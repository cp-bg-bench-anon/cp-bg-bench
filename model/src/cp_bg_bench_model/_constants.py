from __future__ import annotations

from enum import Enum


class DatasetEnum(str, Enum):
    """Enum of datasets covariates."""

    PERTURBATION = "perturbation"
    IMG = "cell"
    MASK = "mask"
