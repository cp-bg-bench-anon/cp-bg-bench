"""Add PCA + Harmony embeddings to cp_measure well-aggregated h5ads.

Mirrors the DL pipeline in ``model/scripts/aggregate_embeddings.py`` so the
classical baseline lands on the same eval surface as the encoders:

    sc.pp.pca(adata, n_comps=50)
    harmonypy.run_harmony(X_pca, obs, ['batch'],
                          max_iter_harmony=100, random_state=42)

Reads:   <ds>_cp_well.h5ad   (output of aggregate_cp_features.py, headline)
Writes:  <ds>_cp_well_harmony.h5ad   (.X unchanged; obsm gets X_pca + X_pca_harmony)

The downstream copairs eval reads ``X_pca`` (pre-Harmony) and
``X_pca_harmony`` (post-Harmony) — same convention as DL.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import harmonypy as hm
import numpy as np
import scanpy as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42
BATCH_COL = "batch"
DEFAULT_N_PCS = 50
HARMONY_MAX_ITER = 100

EXPECTED_BATCHES = {"rxrx1": 11, "rxrx3c": 9, "jump": 10}


def detect_dataset(path: Path) -> str:
    stem = path.name.lower()
    for ds in EXPECTED_BATCHES:
        if stem.startswith(ds):
            return ds
    raise ValueError(f"Cannot detect dataset from filename: {path.name}")


def add_pca_harmony(in_path: Path, out_path: Path, n_pcs: int = DEFAULT_N_PCS) -> None:
    log.info(f"[load] {in_path}")
    adata = ad.read_h5ad(in_path)
    log.info(f"  shape={adata.shape}  obsm_keys={list(adata.obsm.keys())}")

    if BATCH_COL not in adata.obs.columns:
        raise KeyError(f"obs[{BATCH_COL!r}] not found")

    ds_key = detect_dataset(in_path)
    n_batches = adata.obs[BATCH_COL].nunique()
    expected = EXPECTED_BATCHES[ds_key]
    if n_batches != expected:
        log.warning(
            f"[batch] {ds_key}: expected {expected} batches, got {n_batches}. "
            "Continuing — verify upstream pipeline if this is unexpected."
        )
    else:
        log.info(f"[batch] {ds_key}: {n_batches} batches (matches DL pipeline)")

    n_pcs_actual = min(n_pcs, adata.n_obs - 1, adata.n_vars)
    log.info(f"[pca] running PCA with n_comps={n_pcs_actual}")
    np.random.seed(RANDOM_SEED)
    sc.pp.pca(adata, n_comps=n_pcs_actual, random_state=RANDOM_SEED)

    log.info(
        f"[harmony] run_harmony(batch={BATCH_COL!r}, max_iter={HARMONY_MAX_ITER}, "
        f"random_state={RANDOM_SEED})"
    )
    pcs = np.ascontiguousarray(adata.obsm["X_pca"])
    harmony_out = hm.run_harmony(
        pcs,
        adata.obs,
        [BATCH_COL],
        max_iter_harmony=HARMONY_MAX_ITER,
        random_state=RANDOM_SEED,
    )
    adata.obsm["X_pca_harmony"] = np.asarray(harmony_out.Z_corr).astype(np.float32)

    adata.uns.setdefault("cp_pipeline", {})
    adata.uns["cp_pipeline"]["pca_n_comps"] = int(n_pcs_actual)
    adata.uns["cp_pipeline"]["harmony_max_iter"] = HARMONY_MAX_ITER
    adata.uns["cp_pipeline"]["harmony_random_state"] = RANDOM_SEED
    adata.uns["cp_pipeline"]["harmony_batch_col"] = BATCH_COL

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    log.info(f"[write] {out_path}  (obsm: {list(adata.obsm.keys())})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_h5ad", type=Path, required=True)
    ap.add_argument("--out_h5ad", type=Path, required=True)
    ap.add_argument("--n_pcs", type=int, default=DEFAULT_N_PCS)
    args = ap.parse_args()
    add_pca_harmony(args.in_h5ad, args.out_h5ad, n_pcs=args.n_pcs)


if __name__ == "__main__":
    main()
