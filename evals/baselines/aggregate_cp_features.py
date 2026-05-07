"""Aggregate cp_measure single-cell features → well-level for tier-2 baseline.

Mirrors the standard pycytominer 1.4 pipeline. Normalisation and feature
selection are implemented in-house (vectorised; pycytominer's pandas paths
materialise a 100k×2000 frame that exhausts memory). The canonical step —
``Spherize`` — uses ``pycytominer.operations.Spherize`` directly so the
algorithm exactly matches the published recipe.

Pipeline:

1. Cell → well median per (source, plate, well).
2. Inner-join obs from the existing DL-side aggregated h5ad to recover
   ``is_control``, ``batch``, canonical ``perturbation``.
3. Drop features with > ``na_cutoff`` NaN; fill remaining with col median.
4. Per-plate MAD-scale (``mad_robustize``-style: median, MAD×1.4826) against
   ``is_control == True`` rows; clip ±4. Plates with <5 controls fall back to
   plate-wide median/MAD.
5. Variance filter (drop std < 1e-3 after normalisation; mirrors pycytominer
   ``variance_threshold(freq_cut=0.05, unique_cut=0.01)`` for continuous data).
6. Correlation filter (greedy: drop one of each |r|>``corr_threshold`` pair,
   ranked by mean abs correlation; mirrors pycytominer's ``correlation_threshold``).
7. ``pycytominer.operations.Spherize`` with the requested method
   (default ``ZCA-cor``; pass ``--spherize none`` to skip).

Inputs (read-only):
  --in_singlecell    /home/.../baselines/_data/<dataset>_cp_singlecell.h5ad
  --in_aggregated    evals/data/<id>_<dataset>_<...>_C_aggregated.h5ad   (any C view)

Outputs (written):
  --out_h5ad         evals/baselines/_data/<dataset>_cp_well.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pycytominer.operations import Spherize

NA_FRAC_THRESHOLD = 0.05
VAR_STD_THRESHOLD = 1e-3
CORR_THRESHOLD = 0.9
CLIP_ABS = 4.0
SPHERE_EPS = 1e-6


def median_aggregate(adata: ad.AnnData, group_cols: list[str]) -> ad.AnnData:
    """Per-well median across cells (sort + slice; O(n_total · n_features))."""
    obs = adata.obs.copy()
    keys = obs[group_cols].astype(str).agg("__".join, axis=1).values
    uniq, inv = np.unique(keys, return_inverse=True)
    n_wells = len(uniq)

    X_in = np.asarray(adata.X, dtype=np.float32)
    order = np.argsort(inv, kind="stable")
    X_sorted = X_in[order]
    counts = np.bincount(inv, minlength=n_wells)
    starts = np.concatenate([[0], np.cumsum(counts)])

    X_out = np.empty((n_wells, adata.n_vars), dtype=np.float32)
    for i in range(n_wells):
        rows = X_sorted[starts[i] : starts[i + 1]]
        X_out[i] = np.nanmedian(rows, axis=0)

    obs_first_idx = order[starts[:-1]]
    obs_out = obs.iloc[obs_first_idx].reset_index(drop=True)
    obs_out.index = uniq
    return ad.AnnData(X=X_out, obs=obs_out, var=adata.var.copy())


def join_canonical_obs(well_adata: ad.AnnData, ref_adata: ad.AnnData) -> ad.AnnData:
    """Inner-join on (source, plate, well) with the DL-side aggregated obs."""
    key_cols = ["source", "plate", "well"]
    well_keys = well_adata.obs[key_cols].astype(str).agg("__".join, axis=1).values
    ref_keys = ref_adata.obs[key_cols].astype(str).agg("__".join, axis=1).values
    ref_idx = pd.Index(ref_keys)
    keep = pd.Series(well_keys).isin(ref_idx)
    well_adata = well_adata[keep.values].copy()
    well_keys = well_keys[keep.values]

    ref_obs = ref_adata.obs.copy()
    ref_obs.index = ref_keys
    ref_aligned = ref_obs.reindex(well_keys)
    well_adata.obs = ref_aligned
    well_adata.obs_names = well_keys
    return well_adata


def drop_high_na_features(X: np.ndarray, var_names: pd.Index) -> tuple[np.ndarray, pd.Index]:
    keep = np.isnan(X).mean(axis=0) <= NA_FRAC_THRESHOLD
    return X[:, keep], var_names[keep]


def fill_na_with_median(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def mad_normalise(
    X: np.ndarray, plate_keys: np.ndarray, control_mask: np.ndarray
) -> np.ndarray:
    out = np.empty_like(X)
    for plate in np.unique(plate_keys):
        plate_mask = plate_keys == plate
        ctrl_in_plate = plate_mask & control_mask
        ref = X[ctrl_in_plate] if ctrl_in_plate.sum() >= 5 else X[plate_mask]
        med = np.nanmedian(ref, axis=0)
        mad = np.nanmedian(np.abs(ref - med), axis=0)
        scale = np.where(mad > 0, mad * 1.4826, 1.0)
        out[plate_mask] = (X[plate_mask] - med) / scale
    return np.clip(out, -CLIP_ABS, CLIP_ABS)


def variance_filter(X: np.ndarray, var_names: pd.Index) -> tuple[np.ndarray, pd.Index]:
    keep = X.std(axis=0) >= VAR_STD_THRESHOLD
    return X[:, keep], var_names[keep]


def correlation_filter(
    X: np.ndarray, var_names: pd.Index, threshold: float = CORR_THRESHOLD
) -> tuple[np.ndarray, pd.Index]:
    if X.shape[1] <= 1:
        return X, var_names
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    n = Xs.shape[0]
    corr = np.abs(Xs.T @ Xs / n)
    np.fill_diagonal(corr, 0.0)

    keep = np.ones(corr.shape[0], dtype=bool)
    while True:
        sub = corr[np.ix_(keep, keep)]
        if sub.max() <= threshold:
            break
        i_loc, j_loc = np.unravel_index(np.argmax(sub), sub.shape)
        kept_idx = np.where(keep)[0]
        i, j = kept_idx[i_loc], kept_idx[j_loc]
        keep[i if sub[i_loc].mean() >= sub[j_loc].mean() else j] = False
    return X[:, keep], var_names[keep]


def run(
    in_singlecell: Path,
    in_aggregated: Path,
    out_h5ad: Path,
    spherize_method: str = "ZCA-cor",
) -> None:
    print(f"[load] {in_singlecell}")
    sc = ad.read_h5ad(in_singlecell)
    print(f"  cells={sc.n_obs:,}  features={sc.n_vars:,}")

    print("[agg] cell → well median by (source, plate, well)")
    well = median_aggregate(sc, group_cols=["source", "plate", "well"])
    print(f"  wells={well.n_obs:,}")

    print(f"[obs] inner-join canonical obs from {in_aggregated}")
    ref = ad.read_h5ad(in_aggregated)
    well = join_canonical_obs(well, ref)
    n_ctrl = int((well.obs["is_control"].astype(str) == "True").sum())
    print(f"  wells (after join)={well.n_obs:,}  controls={n_ctrl:,}")

    X = np.asarray(well.X, dtype=np.float64)
    var_names = well.var_names

    print(f"[na] drop features with >{NA_FRAC_THRESHOLD:.0%} NaN")
    X, var_names = drop_high_na_features(X, var_names)
    X = fill_na_with_median(X)
    print(f"  features={X.shape[1]:,}")

    print("[norm] per-plate MAD-scale vs is_control")
    plate_keys = well.obs["plate"].astype(str).values
    control_mask = well.obs["is_control"].astype(str).values == "True"
    X = mad_normalise(X, plate_keys, control_mask)

    print(f"[var] drop features with std < {VAR_STD_THRESHOLD}")
    X, var_names = variance_filter(X, var_names)
    print(f"  features={X.shape[1]:,}")

    print(f"[corr] drop |r| > {CORR_THRESHOLD} pairs")
    X, var_names = correlation_filter(X, var_names)
    print(f"  features={X.shape[1]:,}")

    if spherize_method.lower() != "none":
        print(f'[sphere] pycytominer.Spherize(method="{spherize_method}", epsilon={SPHERE_EPS})')
        sphere = Spherize(method=spherize_method, epsilon=SPHERE_EPS, return_numpy=True)
        # pycytominer.Spherize expects a DataFrame even with return_numpy=True.
        X_sphered = sphere.fit_transform(pd.DataFrame(X, columns=list(var_names)))
        X_out = np.asarray(X_sphered, dtype=np.float32)
        var_out = pd.DataFrame(index=[f"Sphere{i+1}" for i in range(X_out.shape[1])])
    else:
        print("[sphere] skipped (--spherize none)")
        X_out = X.astype(np.float32)
        var_out = pd.DataFrame(index=list(var_names))

    out = ad.AnnData(X=X_out, obs=well.obs.copy(), var=var_out)
    out.uns["cp_pipeline"] = {
        "na_cutoff": NA_FRAC_THRESHOLD,
        "var_std_threshold": VAR_STD_THRESHOLD,
        "corr_threshold": CORR_THRESHOLD,
        "clip_abs": CLIP_ABS,
        "spherize": spherize_method,
        "sphere_epsilon": SPHERE_EPS,
    }
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_h5ad, compression="gzip")
    print(f"[write] {out_h5ad}  ({out.n_obs:,} wells × {out.n_vars:,} features)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_singlecell", type=Path, required=True)
    ap.add_argument("--in_aggregated", type=Path, required=True)
    ap.add_argument("--out_h5ad", type=Path, required=True)
    ap.add_argument(
        "--spherize",
        type=str,
        default="ZCA-cor",
        choices=["PCA", "ZCA", "PCA-cor", "ZCA-cor", "none"],
        help="pycytominer Spherize method, or 'none' to skip.",
    )
    args = ap.parse_args()
    run(args.in_singlecell, args.in_aggregated, args.out_h5ad, spherize_method=args.spherize)


if __name__ == "__main__":
    main()
