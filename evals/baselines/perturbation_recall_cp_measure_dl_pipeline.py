"""Run held-out cross-set perturbation Recall@k on cp_measure embeddings
(produced by the canonical DL pipeline: Spherize on controls + PCA + Harmony).

Mirrors the algorithm in evals/01_perturbation_recall/<DS>_perturbation_recall.ipynb
exactly (same cosine similarity, same query/gallery split rules, same held-out
batch values). Output schema matches perturbation_recall_<ds>.csv so the
notebook figure cell can swap data sources by changing one path.

Reads:   _data/_dl_pipeline/cpmeasure_<dataset>_aggregated.h5ad
Output:  perturbation_recall_cp_measure_dl_pipeline.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EMBEDDING_KEYS = ("X_pca", "X_pca_harmony")
PERTURBATION_COL = "perturbation"
CONTROL_COL = "is_control"
HELD_OUT_COL = "batch"
K_MAX = 10

DATA_DIR = Path("${WORKSPACE}/evals/baselines/_data/_dl_pipeline")
OUT_CSV = Path(
    "${WORKSPACE}/evals/baselines/perturbation_recall_cp_measure_dl_pipeline.csv"
)

# Held-out batch per dataset matches the DataModule split used by the DL eval
# notebooks (val_frac=0.1, seed=42 → one specific batch per dataset).
DATASETS = {
    "Rxrx1":  {"path": DATA_DIR / "cpmeasure_Rxrx1_aggregated.h5ad",  "held_out": "HEPG2-07"},
    "Rxrx3C": {"path": DATA_DIR / "cpmeasure_Rxrx3C_aggregated.h5ad", "held_out": "plate_4"},
    "JUMP":   {"path": DATA_DIR / "cpmeasure_JUMP_aggregated.h5ad",   "held_out": "source_4"},
}


def compute_held_out_recall(
    adata: ad.AnnData,
    query_mask: np.ndarray,
    perturbation_col: str = "perturbation",
    control_col: str = "is_control",
    embedding_key: str | None = "X_pca_harmony",
    k_max: int = 10,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Recall@k where queries come from query_mask, candidates from the rest.

    Identical to the function in the DL eval notebooks; reproduced here so the
    cp_measure baseline is computed with byte-identical logic.
    """
    not_ctrl = adata.obs[control_col].astype(str) != "True"
    adata = adata[not_ctrl].copy()
    query_mask = np.asarray(query_mask)[not_ctrl.values]

    if embedding_key is not None and embedding_key in adata.obsm:
        X = np.asarray(adata.obsm[embedding_key])
    else:
        X = np.asarray(adata.X)
        if hasattr(X, "toarray"):
            X = X.toarray()

    X = normalize(X.astype(np.float64), norm="l2")
    labels = np.array([str(x) for x in adata.obs[perturbation_col]])

    q_idx = np.where(query_mask)[0]
    c_idx = np.where(~query_mask)[0]
    if q_idx.size == 0:
        raise ValueError("query_mask selected zero non-control wells")
    if c_idx.size < k_max:
        raise ValueError(f"gallery has {c_idx.size} wells but k_max={k_max}")

    X_q = X[q_idx]
    X_c = X[c_idx]
    labels_q = labels[q_idx]
    labels_c = labels[c_idx]

    n_q = len(q_idx)
    cumulative_hits = np.zeros((n_q, k_max), dtype=bool)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        sim = X_q[start:end] @ X_c.T
        top_k_idx = np.argpartition(-sim, kth=k_max, axis=1)[:, :k_max]
        for i in range(end - start):
            order = np.argsort(-sim[i, top_k_idx[i]])
            top_k_idx[i] = top_k_idx[i][order]
        top_k_labels = labels_c[top_k_idx]
        hits = top_k_labels == labels_q[start:end, None]
        cumulative_hits[start:end] = np.cumsum(hits, axis=1) > 0

    rows = []
    for pert in np.unique(labels_q):
        pert_mask = labels_q == pert
        n_wells = int(pert_mask.sum())
        if n_wells < 1:
            continue
        recall_k = cumulative_hits[pert_mask].mean(axis=0)
        row = {"perturbation": pert, "n_query_wells": n_wells}
        for k in range(1, k_max + 1):
            row[f"R@{k}"] = recall_k[k - 1]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    rows = []
    for ds, cfg in DATASETS.items():
        path, held_out = cfg["path"], cfg["held_out"]
        print(f"=== {ds} ===  {path.name}  (held-out={held_out})")
        adata = ad.read_h5ad(path)
        print(f"  wells={adata.n_obs:,}  obsm={list(adata.obsm.keys())}")

        query_mask = (adata.obs[HELD_OUT_COL].astype(str) == held_out).values
        n_q = int(query_mask.sum())
        if n_q == 0:
            available = sorted(adata.obs[HELD_OUT_COL].astype(str).unique().tolist())
            raise SystemExit(
                f"  No wells matched obs[{HELD_OUT_COL}]=={held_out!r}. "
                f"Available: {available}"
            )

        for emb in EMBEDDING_KEYS:
            t0 = time.time()
            df = compute_held_out_recall(
                adata,
                query_mask=query_mask,
                perturbation_col=PERTURBATION_COL,
                control_col=CONTROL_COL,
                embedding_key=emb,
                k_max=K_MAX,
            )
            tag = "pre_harmony" if emb == "X_pca" else "post_harmony"
            row = {
                "group": f"{ds} — cp_measure",
                "view": "S",
                "embedding": tag,
                "n_perturbations": len(df),
                "n_query_wells": int(df["n_query_wells"].sum()),
            }
            for k in range(1, K_MAX + 1):
                row[f"R@{k}"] = float(df[f"R@{k}"].mean())
            rows.append(row)
            print(
                f"  {emb:>15s}  perts={row['n_perturbations']:>5d}  "
                f"R@1={row['R@1']:.4f}  R@5={row['R@5']:.4f}  R@10={row['R@10']:.4f}  "
                f"({time.time()-t0:.1f}s)"
            )

    df_all = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(df_all[["group", "embedding", "n_perturbations", "R@1", "R@5", "R@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
