"""Run copairs phenotypic_activity on cp_measure embeddings produced by the
canonical DL pipeline (Spherize on controls + PCA + Harmony).

Reads:   _data/_dl_pipeline/cpmeasure_<dataset>_aggregated.h5ad
Output:  phenotypic_activity_cp_measure_dl_pipeline.csv

Schema matches the original cp_measure CSV (group/view/embedding/n_perturbations/
mean_mAP/sig_frac_p/sig_frac_q) so the notebook figure cell can swap data
sources by changing one path.
"""

from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from copairs import map as cmap
from copairs.matching import assign_reference_index

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EMBEDDING_KEYS = ("X_pca", "X_pca_harmony")
PERTURBATION_COL = "perturbation"
CONTROL_COL = "is_control"
PLATE_COLS = ("batch", "plate")
NULL_SIZE = 10000
P_THRESHOLD = 0.05
BATCH_SIZE = 20000

DATA_DIR = Path("${WORKSPACE}/evals/baselines/_data/_dl_pipeline")
OUT_CSV = Path(
    "${WORKSPACE}/evals/baselines/phenotypic_activity_cp_measure_dl_pipeline.csv"
)

DATASETS = {
    "Rxrx1":  DATA_DIR / "cpmeasure_Rxrx1_aggregated.h5ad",
    "Rxrx3C": DATA_DIR / "cpmeasure_Rxrx3C_aggregated.h5ad",
    "JUMP":   DATA_DIR / "cpmeasure_JUMP_aggregated.h5ad",
}


def phenotypic_activity(adata: ad.AnnData, embedding_key: str) -> pd.DataFrame:
    feats = np.asarray(adata.obsm[embedding_key])
    feat_cols = [f"feat_{i}" for i in range(feats.shape[1])]

    keep_cols = list(PLATE_COLS) + ["well", PERTURBATION_COL, CONTROL_COL]
    obs = adata.obs[keep_cols].astype(str).reset_index(drop=True)
    obs.columns = [f"Metadata_{c}" for c in keep_cols]
    df = pd.concat([obs, pd.DataFrame(feats, columns=feat_cols)], axis=1)

    ref_col = "Metadata_reference_index"
    df = assign_reference_index(
        df,
        f"Metadata_{CONTROL_COL} != 'True'",
        reference_col=ref_col,
        default_value=-1,
    )

    pos_sameby = [f"Metadata_{PERTURBATION_COL}"]
    pos_diffby = [f"Metadata_{c}" for c in PLATE_COLS]
    neg_sameby = [f"Metadata_{c}" for c in PLATE_COLS]
    neg_diffby = [ref_col]

    meta = df.filter(regex="^Metadata")
    fv = df.filter(regex="^feat_").values

    ap = cmap.average_precision(
        meta=meta, feats=fv,
        pos_sameby=pos_sameby, pos_diffby=pos_diffby,
        neg_sameby=neg_sameby, neg_diffby=neg_diffby,
        batch_size=BATCH_SIZE,
    )
    ap = ap.query(f"Metadata_{CONTROL_COL} != 'True'")
    maps = cmap.mean_average_precision(
        ap, pos_sameby, null_size=NULL_SIZE, threshold=P_THRESHOLD, seed=RANDOM_SEED,
    )
    return maps


def main() -> None:
    rows = []
    for ds, path in DATASETS.items():
        print(f"=== {ds} ===  {path.name}")
        adata = ad.read_h5ad(path)
        print(f"  wells={adata.n_obs:,}  obsm={list(adata.obsm.keys())}")
        for emb in EMBEDDING_KEYS:
            t0 = time.time()
            maps = phenotypic_activity(adata, emb)
            tag = "pre_harmony" if emb == "X_pca" else "post_harmony"
            row = {
                "group": f"{ds} — cp_measure",
                "view": "S",
                "embedding": tag,
                "n_perturbations": int(len(maps)),
                "mean_mAP": float(maps["mean_average_precision"].mean()),
                "sig_frac_p": float(maps["below_p"].mean()),
                "sig_frac_q": float(maps["below_corrected_p"].mean()),
            }
            rows.append(row)
            print(
                f"  {emb:>15s}  n={row['n_perturbations']:>5d}  mAP={row['mean_mAP']:.3f}  "
                f"sig(p)={row['sig_frac_p']:.3f}  sig(q)={row['sig_frac_q']:.3f}  "
                f"({time.time()-t0:.1f}s)"
            )

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
