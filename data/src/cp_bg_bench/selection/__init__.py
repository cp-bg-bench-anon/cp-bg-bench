"""Cell-selection module for rule E."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cp_bg_bench.selection.uniform import (
    select_uniform_per_compound_source,
    select_uniform_per_well,
    select_uniform_total,
)

logger = logging.getLogger(__name__)

__all__ = ["select_cells"]

# Metadata columns from selected_metadata.parquet carried into the output.
_META_JOIN_COLS = [
    "Metadata_JCP2022",
    "Metadata_InChIKey",
    "Metadata_InChI",
    "Metadata_PlateType",
    "Metadata_SMILES",
]


def select_cells(
    seg_parquet_dir: Path | str,
    meta_parquet: Path | str,
    strategy: str,
    seed: int,
    cells_per_well: int | None = None,
    max_cells: int | None = None,
    target_cells: int | None = None,
    control_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Concatenate per-plate seg parquets, join metadata, apply selection.

    Returns ``(selected_df, scaling_stats)`` where ``scaling_stats`` is
    ``{"n_min": float, "n_max": float}`` for audit.

    Output columns include all SEG_COLUMNS plus ``row_key``, ``source``,
    ``batch``, ``plate``, ``well``, ``tile``, ``n_cells_scaled``, and the
    metadata join columns.
    """
    seg_dir = Path(seg_parquet_dir)
    parquet_paths = sorted(seg_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No segmentation parquets found under {seg_dir}")

    seg_df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    logger.info("select_cells: read %d cells from %d plates", len(seg_df), len(parquet_paths))

    # Load metadata upfront — uniform_per_compound_source needs InChIKey during
    # selection; other strategies use it only for the final join below.
    import pyarrow.parquet as pq

    available = {f.name for f in pq.read_schema(meta_parquet)}
    present_meta_cols = [c for c in _META_JOIN_COLS if c in available]
    meta_df = pd.read_parquet(
        meta_parquet,
        columns=["id"] + present_meta_cols,
    ).rename(columns={"id": "fov_id"})

    # Apply selection strategy
    if strategy == "uniform_per_well":
        if cells_per_well is None:
            raise ValueError("cells_per_well is required for strategy=uniform_per_well")
        selected = select_uniform_per_well(seg_df, cells_per_well=cells_per_well, seed=seed)
    elif strategy == "uniform_total":
        if max_cells is None:
            raise ValueError("max_cells is required for strategy=uniform_total")
        selected = select_uniform_total(seg_df, max_cells=max_cells, seed=seed)
    elif strategy == "uniform_per_compound_source":
        if target_cells is None:
            raise ValueError("target_cells is required for strategy=uniform_per_compound_source")
        if "Metadata_InChIKey" not in meta_df.columns:
            raise ValueError(
                "strategy=uniform_per_compound_source requires Metadata_InChIKey in meta parquet"
            )
        selected = select_uniform_per_compound_source(
            seg_df,
            meta_df=meta_df,
            target_cells=target_cells,
            seed=seed,
            control_labels=control_labels,
        )
    elif strategy == "all":
        selected = seg_df.copy()
    else:
        raise ValueError(f"Unknown selection strategy: {strategy!r}")

    logger.info("select_cells: %d cells selected (strategy=%s)", len(selected), strategy)

    # Compute n_cells_scaled from the selected pool
    n_vals = selected["n_cells_in_fov"].to_numpy(dtype=float)
    n_min = float(n_vals.min()) if len(n_vals) else 0.0
    n_max = float(n_vals.max()) if len(n_vals) else 1.0
    denom = max(n_max - n_min, 1.0)
    selected = selected.copy()
    selected["n_cells_scaled"] = np.clip((n_vals - n_min) / denom * 255.0, 0.0, 255.0)

    # Parse fov_id → structural columns
    parts = selected["fov_id"].str.split("__", n=4, expand=True)
    selected["source"] = parts[0]
    selected["batch"] = parts[1]
    selected["plate"] = parts[2]
    selected["well"] = parts[3]
    selected["tile"] = parts[4]
    selected["row_key"] = selected["fov_id"] + "__" + selected["id_local"].astype(str)

    # Join metadata (loaded upfront above). Metadata_JCP2022 is JUMP-specific
    # and absent for Rxrx1 datasets — _META_JOIN_COLS is a superset.
    selected = selected.merge(meta_df, on="fov_id", how="left")

    # Fill any missing metadata cols with empty string
    for col in _META_JOIN_COLS:
        if col not in selected.columns:
            selected[col] = ""
        else:
            selected[col] = selected[col].fillna("").astype(str)

    return selected, {"n_min": n_min, "n_max": n_max}
