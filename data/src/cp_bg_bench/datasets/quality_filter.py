"""Quality-filter logic for rule I: threshold computation and row filtering."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["compute_thresholds", "filter_dataframe", "filter_hf_dataset"]


def compute_thresholds(
    df: pd.DataFrame,
    fields: list[str],
    quantiles: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Return ``{field: (lo, hi)}`` quantile bounds from ``df``.

    Missing fields are silently skipped so the filter degrades gracefully
    when a field is absent from a particular parquet schema.
    """
    lo_q, hi_q = quantiles
    thresholds: dict[str, tuple[float, float]] = {}
    for field in fields:
        if field not in df.columns:
            logger.warning("quality_filter: field %r not in dataframe, skipping", field)
            continue
        vals = df[field].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            logger.warning("quality_filter: field %r has no valid values, skipping", field)
            continue
        thresholds[field] = (float(np.quantile(vals, lo_q)), float(np.quantile(vals, hi_q)))
    return thresholds


def filter_dataframe(
    df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Filter ``df`` to rows where every field is within its threshold bounds."""
    mask = pd.Series(True, index=df.index)
    for field, (lo, hi) in thresholds.items():
        if field in df.columns:
            mask &= df[field].between(lo, hi)
    return df[mask].reset_index(drop=True)


def filter_hf_dataset(
    ds: Any,
    thresholds: dict[str, tuple[float, float]],
    batch_size: int = 10_000,
    num_proc: int = 1,
) -> Any:
    """Apply ``thresholds`` to a HuggingFace Dataset via batched map.

    Rows where any field falls outside its [lo, hi] bound are dropped.
    Returns a filtered Dataset; the schema is unchanged.
    """

    def _keep(batch: dict) -> dict:
        n = len(next(iter(batch.values())))
        mask = np.ones(n, dtype=bool)
        for field, (lo, hi) in thresholds.items():
            if field in batch:
                vals = np.asarray(batch[field], dtype=float)
                mask &= (vals >= lo) & (vals <= hi)
        return {k: [v for v, m in zip(vs, mask, strict=False) if m] for k, vs in batch.items()}

    return ds.map(
        _keep,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="quality_filter",
    )
