"""Dataset build, quality-filter, and reshard helpers (rules H, I, M)."""

from cp_bg_bench.datasets.hf import (
    VARIANT_NAMES,
    build_hf_features,
    parquet_dir_to_hf,
    reshard_dataset,
)
from cp_bg_bench.datasets.quality_filter import (
    compute_thresholds,
    filter_dataframe,
    filter_hf_dataset,
)

__all__ = [
    "VARIANT_NAMES",
    "build_hf_features",
    "compute_thresholds",
    "filter_dataframe",
    "filter_hf_dataset",
    "parquet_dir_to_hf",
    "reshard_dataset",
]
