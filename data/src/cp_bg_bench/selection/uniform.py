"""Uniform cell selection strategies (rule E)."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "select_uniform_per_compound_source",
    "select_uniform_per_well",
    "select_uniform_total",
]


def select_uniform_per_well(
    seg_df: pd.DataFrame,
    cells_per_well: int,
    seed: int,
) -> pd.DataFrame:
    """Sample up to ``cells_per_well`` cells per ``(plate_key, well)`` group.

    ``seg_df`` must have a ``fov_id`` column formatted as
    ``"{source}__{batch}__{plate}__{well}__{site}"``. Cells are selected
    without replacement; groups with fewer cells are kept whole.

    Returns a new DataFrame with the same columns plus no extra columns.
    The output row order is deterministic for a given ``seed``.
    """
    df = seg_df.copy()
    # Parse fov_id → well + plate_key
    parts = df["fov_id"].str.split("__", n=4, expand=True)
    df["_plate_key"] = parts[0] + "__" + parts[1] + "__" + parts[2]
    df["_well"] = parts[3]

    rng = np.random.default_rng(seed)
    groups: list[pd.DataFrame] = []
    for (_plate_key, _well), grp in df.groupby(["_plate_key", "_well"], sort=True):
        if len(grp) <= cells_per_well:
            groups.append(grp)
        else:
            # sort to preserve original row order so output is stable for same seed
            chosen = sorted(rng.choice(len(grp), size=cells_per_well, replace=False).tolist())
            groups.append(grp.iloc[chosen])

    if not groups:
        return seg_df.iloc[:0].copy()

    result = pd.concat(groups, ignore_index=True)
    return result.drop(columns=["_plate_key", "_well"])


def select_uniform_per_compound_source(
    seg_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    target_cells: int,
    seed: int,
    control_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Sample up to ``target_cells`` cells per ``(Metadata_InChIKey × source)`` group.

    ``meta_df`` must have columns ``fov_id`` and ``Metadata_InChIKey``.
    ``source`` is derived from the ``fov_id`` prefix
    (``source__batch__plate__well__site``).

    Rows with a null / empty InChIKey (e.g. JUMP DMSO) are kept intact — they
    bypass the per-(perturbation × source) cap. ``control_labels`` extends
    that bypass to explicit label strings (e.g. Rxrx1's ``"EMPTY"``
    negative controls, which carry a non-null identity label).

    Returns a new DataFrame with the same columns as ``seg_df`` (no extras added).
    """
    if len(seg_df) == 0:
        return seg_df.copy()

    ctrl_set: set[str] = set(control_labels) if control_labels else set()

    # Map fov_id → InChIKey from the pre-joined metadata
    ik_map = (
        meta_df[["fov_id", "Metadata_InChIKey"]]
        .drop_duplicates("fov_id")
        .set_index("fov_id")["Metadata_InChIKey"]
    )
    ik = seg_df["fov_id"].map(ik_map)
    source = seg_df["fov_id"].str.split("__", n=1).str[0]

    missing_fov_ids = set(seg_df["fov_id"].unique()) - set(meta_df["fov_id"].unique())
    if missing_fov_ids:
        warnings.warn(
            f"{len(missing_fov_ids)} fov_id(s) in seg_df have no matching entry in meta_df "
            f"and will be treated as controls (bypass cap). "
            f"First missing: {next(iter(sorted(missing_fov_ids)))}",
            UserWarning,
            stacklevel=2,
        )

    is_treatment = ik.notna() & (ik != "") & (~ik.isin(ctrl_set))
    control_df = seg_df[~is_treatment]
    treat_df = seg_df[is_treatment].copy()
    treat_df["_ik"] = ik[is_treatment].to_numpy()
    treat_df["_src"] = source[is_treatment].to_numpy()

    rng = np.random.default_rng(seed)
    groups: list[pd.DataFrame] = []
    for (_ik, _src), grp in treat_df.groupby(["_ik", "_src"], sort=True):
        grp_clean = grp.drop(columns=["_ik", "_src"])
        if len(grp_clean) <= target_cells:
            groups.append(grp_clean)
        else:
            # sort to preserve original row order so output is stable for same seed
            chosen = sorted(rng.choice(len(grp_clean), size=target_cells, replace=False).tolist())
            groups.append(grp_clean.iloc[chosen])

    if not groups:
        return control_df.reset_index(drop=True)

    return pd.concat(groups + [control_df], ignore_index=True)


def select_uniform_total(
    seg_df: pd.DataFrame,
    max_cells: int,
    seed: int,
) -> pd.DataFrame:
    """Sample up to ``max_cells`` cells uniformly from the entire pool.

    Cells are drawn without replacement across all FOVs and wells.  If fewer
    than ``max_cells`` cells are available the full pool is returned without
    raising — the caller gets what exists.

    Returns a new DataFrame in deterministic order for a given ``seed``.
    """
    if len(seg_df) == 0:
        return seg_df.copy()
    if len(seg_df) <= max_cells:
        logger.info(
            "select_uniform_total: requested %d cells but only %d available; returning all",
            max_cells,
            len(seg_df),
        )
        return seg_df.copy()
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(seg_df), size=max_cells, replace=False).tolist())
    return seg_df.iloc[chosen].reset_index(drop=True)
