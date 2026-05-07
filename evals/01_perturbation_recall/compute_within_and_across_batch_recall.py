"""Compute both within-batch and across-batch Recall@k for all DL configs.

Writes one CSV per dataset with both metrics. Schema:
  group, view, embedding, recall_type, n_perturbations, n_query_wells, R@1..R@10

`recall_type` ∈ {'within_batch', 'across_batch'}:
  - across_batch: query=held-out batch, gallery=other batches (same as the
    existing per-dataset notebooks). Tests cross-batch generalization.
  - within_batch: query=held-out batch, gallery=same held-out batch (excluding
    self). Tests retrieval *without* batch effects to overcome — a ceiling.

The held-out batch values match the existing notebooks (matches DataModule split).
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

EMBED_DIR = Path("${WORKSPACE}/evals/data")
OUT_DIR = Path("${WORKSPACE}/evals/01_perturbation_recall")

EMBEDDING_KEYS = ("X_pca", "X_pca_harmony")
PERTURBATION_COL = "perturbation"
CONTROL_COL = "is_control"
HELD_OUT_COL = "batch"
K_MAX = 10

DATASETS = {
    "JUMP":   {
        "held_out": "source_4",
        "groups": {
            'JUMP — DINO':       {'C': '01_JUMP_DINO_ECFP4_C_aggregated.h5ad',
                                  'CD': '02_JUMP_DINO_ECFP4_CD_aggregated.h5ad',
                                  'S':  '03_JUMP_DINO_ECFP4_S_aggregated.h5ad',
                                  'SD': '04_JUMP_DINO_ECFP4_SD_aggregated.h5ad'},
            'JUMP — OpenPhenom': {'C': '05_JUMP_OpenPhenom_ECFP4_C_aggregated.h5ad',
                                  'CD': '06_JUMP_OpenPhenom_ECFP4_CD_aggregated.h5ad',
                                  'S':  '07_JUMP_OpenPhenom_ECFP4_S_aggregated.h5ad',
                                  'SD': '08_JUMP_OpenPhenom_ECFP4_SD_aggregated.h5ad'},
            'JUMP — SubCell':    {'C': '09_JUMP_SubCell_ECFP4_C_aggregated.h5ad',
                                  'CD': '10_JUMP_SubCell_ECFP4_CD_aggregated.h5ad',
                                  'S':  '11_JUMP_SubCell_ECFP4_S_aggregated.h5ad',
                                  'SD': '12_JUMP_SubCell_ECFP4_SD_aggregated.h5ad'},
        },
        "out_csv": OUT_DIR / "perturbation_recall_within_across_jump.csv",
    },
    "Rxrx1": {
        "held_out": "HEPG2-07",
        "groups": {
            'Rxrx1 — DINO':       {'C': '13_Rxrx1_DINO_ESM2_C_aggregated.h5ad',
                                   'CD': '14_Rxrx1_DINO_ESM2_CD_aggregated.h5ad',
                                   'S':  '15_Rxrx1_DINO_ESM2_S_aggregated.h5ad',
                                   'SD': '16_Rxrx1_DINO_ESM2_SD_aggregated.h5ad'},
            'Rxrx1 — OpenPhenom': {'C': '17_Rxrx1_OpenPhenom_ESM2_C_aggregated.h5ad',
                                   'CD': '18_Rxrx1_OpenPhenom_ESM2_CD_aggregated.h5ad',
                                   'S':  '19_Rxrx1_OpenPhenom_ESM2_S_aggregated.h5ad',
                                   'SD': '20_Rxrx1_OpenPhenom_ESM2_SD_aggregated.h5ad'},
            'Rxrx1 — SubCell':    {'C': '21_Rxrx1_SubCell_ESM2_C_aggregated.h5ad',
                                   'CD': '22_Rxrx1_SubCell_ESM2_CD_aggregated.h5ad',
                                   'S':  '23_Rxrx1_SubCell_ESM2_S_aggregated.h5ad',
                                   'SD': '24_Rxrx1_SubCell_ESM2_SD_aggregated.h5ad'},
        },
        "out_csv": OUT_DIR / "perturbation_recall_within_across_rxrx1.csv",
    },
    "Rxrx3C": {
        "held_out": "plate_4",
        "groups": {
            'Rxrx3C — DINO':       {'C': '25_Rxrx3C_DINO_ESM2_C_aggregated.h5ad',
                                    'CD': '26_Rxrx3C_DINO_ESM2_CD_aggregated.h5ad',
                                    'S':  '27_Rxrx3C_DINO_ESM2_S_aggregated.h5ad',
                                    'SD': '28_Rxrx3C_DINO_ESM2_SD_aggregated.h5ad'},
            'Rxrx3C — OpenPhenom': {'C': '29_Rxrx3C_OpenPhenom_ESM2_C_aggregated.h5ad',
                                    'CD': '30_Rxrx3C_OpenPhenom_ESM2_CD_aggregated.h5ad',
                                    'S':  '31_Rxrx3C_OpenPhenom_ESM2_S_aggregated.h5ad',
                                    'SD': '32_Rxrx3C_OpenPhenom_ESM2_SD_aggregated.h5ad'},
            'Rxrx3C — SubCell':    {'C': '33_Rxrx3C_SubCell_ESM2_C_aggregated.h5ad',
                                    'CD': '34_Rxrx3C_SubCell_ESM2_CD_aggregated.h5ad',
                                    'S':  '35_Rxrx3C_SubCell_ESM2_S_aggregated.h5ad',
                                    'SD': '36_Rxrx3C_SubCell_ESM2_SD_aggregated.h5ad'},
        },
        "out_csv": OUT_DIR / "perturbation_recall_within_across_rxrx3c.csv",
    },
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
    Returns one row per perturbation with R@1..R@k_max.
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

    # For self-masking, find each query's position in the gallery (if present).
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
        sim = X_q[start:end] @ X_g.T  # (b, n_gallery)
        # Self-mask: prevent a well from retrieving itself when q⊂g overlap.
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


def run_dataset(ds_key: str, cfg: dict) -> Path:
    print(f"\n=== {ds_key} ===  held-out={cfg['held_out']!r}")
    summary_rows = []

    for group_name, view_files in cfg["groups"].items():
        print(f"  --- {group_name} ---")
        for view, fname in view_files.items():
            adata = ad.read_h5ad(EMBED_DIR / fname)
            in_batch = (adata.obs[HELD_OUT_COL].astype(str) == cfg["held_out"]).values
            for emb in EMBEDDING_KEYS:
                tag = "pre_harmony" if emb == "X_pca" else "post_harmony"
                # Across: query=held-out, gallery=other batches
                t0 = time.time()
                df_across = compute_recall(adata, in_batch, ~in_batch, emb, k_max=K_MAX)
                t_across = time.time() - t0
                # Within: query=held-out, gallery=same batch (with self-mask)
                t0 = time.time()
                df_within = compute_recall(adata, in_batch, in_batch, emb, k_max=K_MAX)
                t_within = time.time() - t0

                for recall_type, df in [("across_batch", df_across), ("within_batch", df_within)]:
                    row = {
                        "group": group_name,
                        "view": view,
                        "embedding": tag,
                        "recall_type": recall_type,
                        "n_perturbations": len(df),
                        "n_query_wells": int(df["n_query_wells"].sum()),
                    }
                    for k in range(1, K_MAX + 1):
                        row[f"R@{k}"] = float(df[f"R@{k}"].mean())
                    summary_rows.append(row)

                print(
                    f"    {view:<3} [{tag[:4]}]  across R@10={df_across['R@10'].mean():.4f} ({t_across:.1f}s)  "
                    f"within R@10={df_within['R@10'].mean():.4f} ({t_within:.1f}s)"
                )

    df_out = pd.DataFrame(summary_rows)
    cfg["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cfg["out_csv"], index=False)
    print(f"  Saved: {cfg['out_csv']}  ({len(df_out)} rows)")
    return cfg["out_csv"]


def main() -> None:
    written = []
    for ds_key, cfg in DATASETS.items():
        written.append(run_dataset(ds_key, cfg))
    print("\nAll outputs:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
