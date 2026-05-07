"""Run the canonical DL aggregation pipeline on cp_measure singlecell h5ads.

Mirrors model/scripts/aggregate_embeddings.py exactly. cp_measure features go
through the same Spherize-on-controls + PCA + Harmony pipeline as DL encoders,
removing all preprocessing-asymmetry from the cp_measure ↔ DL comparison.

cp_measure features need extra hygiene that DL features don't:
  * Inf values from divide-by-zero on degenerate cells -> NaN
  * High-NaN columns (>5% missing) dropped
  * Remaining NaN filled with column median
  * Zero-variance-in-controls features dropped (else Spherize divides by zero)

The substantive normalization is unchanged: mean-pool cells->wells,
Spherize on controls, PCA(50), Harmony on `obs.batch`.

Strategy: pre-aggregate cells -> wells (mean, NaN-aware) inside this script so
we can prune features that have zero variance among control wells. Write the
already-well-aggregated profile as a `_singlecell.h5ad` with one "cell" per
well; aggregate_embeddings's group-by-well step then becomes a no-op pass-
through, and the rest of the pipeline (Spherize, PCA, Harmony) runs as-is.

Outputs: evals/baselines/_data/_dl_pipeline/cpmeasure_<dataset>_aggregated.h5ad
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "${WORKSPACE}/model/scripts")
from aggregate_embeddings import aggregate_embeddings  # noqa: E402

CPM_DATA = Path("${WORKSPACE}/evals/baselines/_data")
OUT_DIR = CPM_DATA / "_dl_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NA_FRAC_THRESHOLD = 0.05
CTRL_STD_THRESHOLD = 1e-3
JUMP_DMSO_INCHIKEY = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"

DATASETS = {
    "Rxrx1": {
        "in_path":       CPM_DATA / "rxrx1_cp_singlecell.h5ad",
        "control_value": "EMPTY",
    },
    "Rxrx3C": {
        "in_path":       CPM_DATA / "rxrx3c_cp_singlecell.h5ad",
        "control_value": "EMPTY",
    },
    "JUMP": {
        "in_path":       CPM_DATA / "jump_cp_singlecell.h5ad",
        "control_value": JUMP_DMSO_INCHIKEY,
    },
}


def clean_and_well_aggregate(in_path: Path, control_value: str) -> ad.AnnData:
    """Mean-aggregate cells to wells, drop features that break Spherize on controls.

    Returns a well-level AnnData with `obs` carrying the group-by columns and
    `obs[batch]` set so downstream Harmony works.
    """
    print(f"  [load] {in_path.name}")
    a = ad.read_h5ad(in_path)
    X = np.asarray(a.X, dtype=np.float32)
    print(f"  [load] cells={a.n_obs:,}  features={a.n_vars}")

    n_inf = int(np.isinf(X).sum())
    if n_inf:
        print(f"  [clean] inf -> NaN: {n_inf:,}")
        X[np.isinf(X)] = np.nan

    nan_frac = np.isnan(X).mean(axis=0)
    keep = nan_frac <= NA_FRAC_THRESHOLD
    print(f"  [clean] drop {(~keep).sum()}/{keep.size} features with >{NA_FRAC_THRESHOLD:.0%} NaN")
    X = X[:, keep]
    var = a.var.iloc[keep].copy()

    med = np.nanmedian(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(med, nan_idx[1])

    # Mean-aggregate cells -> wells, grouping by (source, plate, well)
    obs = a.obs.copy()
    keys = obs[["source", "plate", "well"]].astype(str).agg("__".join, axis=1).values
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_wells = len(uniq)
    sums = np.zeros((n_wells, X.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, X.astype(np.float64))
    well_X = (sums / counts[:, None]).astype(np.float32)
    print(f"  [agg] cells -> wells: {n_wells:,} wells")

    # First-row obs per well (deterministic via stable sort)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first_idx = order[starts]
    well_obs = obs.iloc[first_idx].reset_index(drop=True)
    well_obs.index = uniq

    # Variance filter on controls — drop features with std < threshold in controls
    ctrl_mask = (well_obs["perturbation"].astype(str) == control_value).values
    n_ctrl = int(ctrl_mask.sum())
    print(f"  [ctrl] {n_ctrl:,} control wells (perturbation==={control_value!r})")
    if n_ctrl == 0:
        raise ValueError(f"No control wells found")
    ctrl_std = well_X[ctrl_mask].std(axis=0, ddof=0)
    keep_v = ctrl_std >= CTRL_STD_THRESHOLD
    print(f"  [filter] drop {(~keep_v).sum()}/{keep_v.size} features with control std < {CTRL_STD_THRESHOLD}")
    well_X = well_X[:, keep_v]
    var = var.iloc[keep_v].copy()

    # NaN/Inf double-check (mean of all-NaN bins could re-introduce NaN if any made it through)
    bad = np.isnan(well_X) | np.isinf(well_X)
    if bad.any():
        col_med = np.nanmedian(well_X, axis=0)
        bad_idx = np.where(bad)
        well_X[bad_idx] = np.take(col_med, bad_idx[1])
        print(f"  [post-fill] replaced {int(bad.sum())} residual NaN/Inf with col median")

    # Build well-level AnnData with shape (n_wells, n_features). Down the line,
    # aggregate_embeddings group-bys (source, plate, well) — already unique here,
    # so its aggregation is identity. Spherize/PCA/Harmony run as DL pipeline.
    a_well = ad.AnnData(X=well_X, obs=well_obs, var=var)
    return a_well


def run_one(dataset: str, in_path: Path, control_value: str) -> Path:
    print(f"\n=== {dataset} ===")
    a_well = clean_and_well_aggregate(in_path, control_value)

    with tempfile.TemporaryDirectory(prefix="cpm_dl_", dir="${SCRATCH}") as td:
        # aggregate_embeddings derives config_id from input filename (sans
        # `_singlecell`), so name the file accordingly.
        tmp = Path(td) / f"cpmeasure_{dataset}_singlecell.h5ad"
        a_well.obs["row_key"] = a_well.obs.index.astype(str)
        a_well.write_h5ad(tmp)
        print(f"  [tmp] wrote pre-aggregated {tmp.name}  shape={a_well.shape}")

        out_path = aggregate_embeddings(
            input_path=tmp,
            output_dir=OUT_DIR,
            group_cols=["source", "plate", "well"],
            control_col="perturbation",
            control_value=control_value,
            n_pcs=50,
        )
    print(f"  [done] {out_path.name}")
    return out_path


def main() -> None:
    written = []
    for ds, cfg in DATASETS.items():
        out = run_one(ds, cfg["in_path"], cfg["control_value"])
        written.append(out)
    print("\n=== Final outputs ===")
    for p in written:
        a = ad.read_h5ad(p, backed="r")
        print(f"  {p.name}  shape={a.shape}  obsm={list(a.obsm.keys())}")


if __name__ == "__main__":
    main()
