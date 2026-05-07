"""Compute within-batch and across-batch Recall@k on cp_measure embeddings
produced by the canonical DL pipeline (Spherize on controls + PCA + Harmony).

Mirrors evals/01_perturbation_recall/compute_within_and_across_batch_recall.py
exactly — same metrics, same held-out batch values per dataset, same self-mask
logic. Output schema matches so the analysis notebook can concat both.

Reads:   _data/_dl_pipeline/cpmeasure_<dataset>_aggregated.h5ad
Output:  perturbation_recall_within_and_across_cp_measure.csv
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
    "${WORKSPACE}/evals/baselines/perturbation_recall_within_and_across_cp_measure.csv"
)

DATASETS = {
    "Rxrx1":  {"path": DATA_DIR / "cpmeasure_Rxrx1_aggregated.h5ad",  "held_out": "HEPG2-07"},
    "Rxrx3C": {"path": DATA_DIR / "cpmeasure_Rxrx3C_aggregated.h5ad", "held_out": "plate_4"},
    "JUMP":   {"path": DATA_DIR / "cpmeasure_JUMP_aggregated.h5ad",   "held_out": "source_4"},
}


def compute_recall(
    adata: ad.AnnData,
    query_mask: np.ndarray,
    gallery_mask: np.ndarray,
    embedding_key: str,
    k_max: int = 10,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Recall@k where queries come from query_mask, candidates from gallery_mask.

    A well that appears in both masks is excluded from its own gallery
    (self-similarity → −∞). Drops controls before computing similarity.
    """
    not_ctrl = adata.obs[CONTROL_COL].astype(str) != "True"
    adata = adata[not_ctrl].copy()
    query_mask = np.asarray(query_mask)[not_ctrl.values]
    gallery_mask = np.asarray(gallery_mask)[not_ctrl.values]

    if embedding_key in adata.obsm:
        X = np.asarray(adata.obsm[embedding_key])
    else:
        X = np.asarray(adata.X)
        if hasattr(X, "toarray"):
            X = X.toarray()

    X = normalize(X.astype(np.float64), norm="l2")
    labels = np.array([str(x) for x in adata.obs[PERTURBATION_COL]])

    q_idx = np.where(query_mask)[0]
    g_idx = np.where(gallery_mask)[0]
    if q_idx.size == 0:
        raise ValueError("query_mask selected zero non-control wells")
    if g_idx.size < k_max + 1:
        raise ValueError(f"gallery has {g_idx.size} wells but need >={k_max+1}")

    g_idx_to_pos = {gi: pos for pos, gi in enumerate(g_idx)}
    self_pos = np.array([g_idx_to_pos.get(qi, -1) for qi in q_idx], dtype=np.int64)

    X_q = X[q_idx]
    X_g = X[g_idx]
    labels_q = labels[q_idx]
    labels_g = labels[g_idx]

    n_q = len(q_idx)
    cumulative_hits = np.zeros((n_q, k_max), dtype=bool)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        sim = X_q[start:end] @ X_g.T
        for i in range(end - start):
            if self_pos[start + i] >= 0:
                sim[i, self_pos[start + i]] = -np.inf
        top_k_idx = np.argpartition(-sim, kth=k_max, axis=1)[:, :k_max]
        for i in range(end - start):
            order = np.argsort(-sim[i, top_k_idx[i]])
            top_k_idx[i] = top_k_idx[i][order]
        top_k_labels = labels_g[top_k_idx]
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
        print(f"=== {ds} ===  {path.name}  (held-out={held_out!r})")
        adata = ad.read_h5ad(path)
        in_batch = (adata.obs[HELD_OUT_COL].astype(str) == held_out).values
        if in_batch.sum() == 0:
            available = sorted(adata.obs[HELD_OUT_COL].astype(str).unique().tolist())
            raise SystemExit(f"  No wells in held-out batch. Available: {available}")

        for emb in EMBEDDING_KEYS:
            tag = "pre_harmony" if emb == "X_pca" else "post_harmony"
            t0 = time.time()
            df_across = compute_recall(adata, in_batch, ~in_batch, emb, k_max=K_MAX)
            t_across = time.time() - t0
            t0 = time.time()
            df_within = compute_recall(adata, in_batch, in_batch, emb, k_max=K_MAX)
            t_within = time.time() - t0

            for recall_type, df in [("across_batch", df_across), ("within_batch", df_within)]:
                row = {
                    "group": f"{ds} — cp_measure",
                    "view": "S",
                    "embedding": tag,
                    "recall_type": recall_type,
                    "n_perturbations": len(df),
                    "n_query_wells": int(df["n_query_wells"].sum()),
                }
                for k in range(1, K_MAX + 1):
                    row[f"R@{k}"] = float(df[f"R@{k}"].mean())
                rows.append(row)

            print(
                f"  {emb:>15s}  across R@10={df_across['R@10'].mean():.4f} ({t_across:.1f}s)  "
                f"within R@10={df_within['R@10'].mean():.4f} ({t_within:.1f}s)"
            )

    df_all = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(df_all[["group", "embedding", "recall_type", "R@1", "R@5", "R@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
