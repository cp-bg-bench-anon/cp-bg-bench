from __future__ import annotations

import torch
import torch.nn.functional as F


def knn_recall(
    left_embedding: torch.Tensor,
    right_embedding: torch.Tensor,
    embedding_type: str,
    stage: str,
    recall_range: tuple[int, ...] = (1, 5, 10),
    left_labels: torch.Tensor | None = None,
    right_labels: torch.Tensor | None = None,
    normalize: bool = False,
) -> dict[str, float]:
    """
    If left_labels/right_labels are None -> exact index-pair retrieval (your current behavior).
    If labels are provided -> any top-K item with the same label counts as a hit.
    """
    if normalize:
        left_embedding = F.normalize(left_embedding, p=2, dim=-1)
        right_embedding = F.normalize(right_embedding, p=2, dim=-1)

    sim = left_embedding @ right_embedding.T  # (N, M)
    N = sim.shape[0]
    metrics: dict[str, float] = {}

    # Index-paired mode (requires N == M)
    if left_labels is None or right_labels is None:
        if sim.shape[0] != sim.shape[1]:
            raise ValueError("Index-paired recall requires left and right to have the same length.")
        idx = torch.arange(N, device=sim.device)
        for K in recall_range:
            topk_idx = sim.topk(k=K, dim=1, largest=True, sorted=False).indices  # (N, K)
            hits = (topk_idx == idx.unsqueeze(1)).any(dim=1)
            metrics[f"{stage}_R@{K}_{embedding_type}"] = hits.float().mean().item()
        return metrics

    # Grouped mode: any same-label match in top-K is a hit
    if left_labels.device != sim.device:
        left_labels = left_labels.to(sim.device)
    if right_labels.device != sim.device:
        right_labels = right_labels.to(sim.device)

    for K in recall_range:
        topk_idx = sim.topk(k=K, dim=1, largest=True, sorted=False).indices  # (N, K)
        right_topk_labels = right_labels[topk_idx]  # (N, K)
        hits = (right_topk_labels == left_labels.unsqueeze(1)).any(dim=1)  # (N,)
        metrics[f"{stage}_R@{K}_{embedding_type}"] = hits.float().mean().item()

    return metrics


def _prototypes_from_inv(
    emb: torch.Tensor,
    inv: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-class mean prototypes.

    Args:
        emb: (N, D) float tensor
        inv: (N,) long tensor mapping each row to [0..n_classes-1]
        n_classes: number of unique classes

    Returns:
        proto: (n_classes, D)
        counts: (n_classes,) float counts
    """
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2D, got shape={tuple(emb.shape)}")
    if inv.ndim != 1:
        raise ValueError(f"inv must be 1D, got shape={tuple(inv.shape)}")
    if emb.shape[0] != inv.shape[0]:
        raise ValueError("emb and inv length mismatch")

    n_classes = int(n_classes)
    d = int(emb.shape[1])

    emb = emb.float()
    inv = inv.long()

    sums = torch.zeros((n_classes, d), dtype=torch.float32, device=emb.device)
    sums.index_add_(0, inv, emb)

    counts = torch.bincount(inv, minlength=n_classes).to(dtype=torch.float32, device=emb.device)
    proto = sums / counts.clamp_min(1.0).unsqueeze(1)
    return proto, counts


def _recall_at_k_compact_ids(
    queries: torch.Tensor,
    index: torch.Tensor,
    *,
    query_compact_ids: torch.Tensor,
    recall_range: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 2048,
    device: torch.device | str = "cuda",
    normalize: bool = False,
) -> tuple[dict[int, float], dict[int, torch.Tensor]]:
    """
    Recall@K where the "correct" target for query i is having compact id == query_compact_ids[i]
    appear in the top-K *indices* (because the index is ordered by compact id).

    Returns:
        micro: {K: float}
        hits_by_k: {K: BoolTensor(N_queries)} on CPU
    """
    if normalize:
        queries = F.normalize(queries, p=2, dim=-1)
        index = F.normalize(index, p=2, dim=-1)

    if queries.ndim != 2 or index.ndim != 2:
        raise ValueError("queries and index must be 2D")
    if query_compact_ids.ndim != 1:
        raise ValueError("query_compact_ids must be 1D")
    if queries.shape[0] != query_compact_ids.shape[0]:
        raise ValueError("queries and query_compact_ids length mismatch")

    n_q = int(queries.shape[0])
    n_i = int(index.shape[0])
    if n_i == 0 or n_q == 0:
        return {k: float("nan") for k in recall_range}, {k: torch.empty((0,), dtype=torch.bool) for k in recall_range}

    kmax = min(int(max(recall_range)), n_i)

    index_t = index.to(device=device, dtype=torch.float32, non_blocking=True).t()  # (D, n_i)
    q_ids_cpu = query_compact_ids.long().cpu()

    hits_by_k: dict[int, torch.Tensor] = {}
    hit_counts = {int(k): 0 for k in recall_range}

    for start in range(0, n_q, int(chunk_size)):
        end = min(start + int(chunk_size), n_q)

        q = queries[start:end].to(device=device, dtype=torch.float32, non_blocking=True)  # (B, D)
        sims = q @ index_t  # (B, n_i)
        topk_idx = sims.topk(k=kmax, dim=1, largest=True, sorted=False).indices.cpu()  # (B, kmax)

        tgt = q_ids_cpu[start:end].unsqueeze(1)  # (B, 1)
        eq = topk_idx.eq(tgt)  # (B, kmax)

        for k in recall_range:
            kk = min(int(k), kmax)
            h = eq[:, :kk].any(dim=1)  # (B,)
            hit_counts[int(k)] += int(h.sum().item())
            hits_by_k.setdefault(int(k), []).append(h)

    hits_by_k_out: dict[int, torch.Tensor] = {k: torch.cat(v, dim=0) for k, v in hits_by_k.items()}
    micro = {k: hit_counts[k] / max(n_q, 1) for k in hit_counts}
    return micro, hits_by_k_out


def full_val_recall_compound_level(
    img_emb: torch.Tensor,
    mol_emb: torch.Tensor,
    compound_ids: torch.Tensor,
    *,
    recall_range: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 2048,
    device: torch.device | str = "cuda",
    normalize: bool = False,
    compute_macro_i2p: bool = True,
    prefix: str = "val",
) -> dict[str, float]:
    """
    Full-validation retrieval evaluation aligned with multi-positive-by-compound training.

    Definitions:
      - Build compact class ids 0..C-1 for unique compounds in this validation epoch.
      - Molecule index: per-compound molecule prototype (mean of mol_emb within compound).
      - Image prototypes: per-compound image prototype (mean of img_emb within compound).

    Metrics:
      - val_R@K_I2P_micro: average over *images* (all val samples)
      - val_R@K_I2P_macro: average over *compounds* (mean of per-compound image hit rate) [optional]
      - val_R@K_P2I: perturbation prototype queries vs image prototypes index (one query per compound)

    Note:
      - This evaluates retrieval against the full val set (at compound level), not per-batch.
    """
    if img_emb.ndim != 2 or mol_emb.ndim != 2:
        raise ValueError("img_emb and mol_emb must be 2D")
    if compound_ids.ndim != 1:
        raise ValueError("compound_ids must be 1D")
    if img_emb.shape[0] != mol_emb.shape[0] or img_emb.shape[0] != compound_ids.shape[0]:
        raise ValueError("img_emb/mol_emb/compound_ids length mismatch")

    img_emb = img_emb.detach()
    mol_emb = mol_emb.detach()
    compound_ids = compound_ids.detach().long()

    uniq, inv = torch.unique(compound_ids.cpu(), sorted=True, return_inverse=True)
    inv = inv.to(dtype=torch.long, device=img_emb.device)
    c = int(uniq.numel())
    if c == 0:
        return {}

    # Prototypes on CPU for stability; move indices to GPU for fast topk.
    img_cpu = img_emb.cpu()
    mol_cpu = mol_emb.cpu()
    inv_cpu = inv.cpu()

    mol_proto_cpu, _ = _prototypes_from_inv(mol_cpu, inv_cpu, c)  # (C, D)
    img_proto_cpu, img_counts_cpu = _prototypes_from_inv(img_cpu, inv_cpu, c)  # (C, D), (C,)

    # I2P: queries are all images; index is per-compound perturbation prototypes
    i2p_micro, i2p_hits = _recall_at_k_compact_ids(
        img_cpu,
        mol_proto_cpu,
        query_compact_ids=inv_cpu,
        recall_range=recall_range,
        chunk_size=chunk_size,
        device=device,
        normalize=normalize,
    )

    out: dict[str, float] = {}
    for k, v in i2p_micro.items():
        out[f"{prefix}_R@{k}_I2P"] = float(v)

    if compute_macro_i2p:
        # Macro: average per compound of (hit rate over that compound's images)
        counts = img_counts_cpu.clamp_min(1.0)  # (C,)
        for k in recall_range:
            h = i2p_hits[int(k)].to(dtype=torch.float32)  # (N,)
            per_comp_sum = torch.zeros((c,), dtype=torch.float32)
            per_comp_sum.index_add_(0, inv_cpu, h)
            per_comp_rate = per_comp_sum / counts
            out[f"{prefix}_R@{int(k)}_I2P_macro"] = float(per_comp_rate.mean().item())

    # P2I: queries are per-compound perturbation prototypes; index is per-compound image prototypes
    # Compact id for each query is 0..C-1
    q_ids = torch.arange(c, dtype=torch.long)
    p2i_micro, _ = _recall_at_k_compact_ids(
        mol_proto_cpu,
        img_proto_cpu,
        query_compact_ids=q_ids,
        recall_range=recall_range,
        chunk_size=chunk_size,
        device=device,
        normalize=normalize,
    )
    for k, v in p2i_micro.items():
        out[f"{prefix}_R@{k}_P2I"] = float(v)

    return out
