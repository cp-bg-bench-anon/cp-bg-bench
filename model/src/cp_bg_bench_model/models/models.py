from __future__ import annotations

import hashlib
import math
import os
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from open_clip.loss import ClipLoss, SigLipLoss
from pytorch_lightning.core import LightningModule
from torch import Tensor

from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model._evals import full_val_recall_compound_level
from cp_bg_bench_model._logging import logger
from cp_bg_bench_model.encoders import ImageEncoderRegistry, MoleculeEncoderRegistry
from cp_bg_bench_model.lora import LoraConfig
from cp_bg_bench_model.models._utils import _get_cosine_schedule_with_warmup_lr_lambda


def _stable_str_id(s: str) -> int:
    # deterministic helper for hashing if we ever go multi-GPU
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


def world_info_from_env():
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def _dist_ready() -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return False
    try:
        return dist.get_world_size() > 1
    except Exception:
        return False


def _all_gather_with_grad(x: Tensor) -> Tensor:
    if not _dist_ready():
        return x
    chunks = dist.nn.all_gather(x)  # type: ignore[attr-defined]
    return torch.cat(chunks, dim=0)


def _all_gather_no_grad(x: Tensor) -> Tensor:
    if not _dist_ready():
        return x
    xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=0)


def _combine_dir_losses(
    loss_i2p: Tensor,
    loss_p2i: Tensor,
    *,
    i2p_weight: float,
    p2i_weight: float,
    eps: float,
) -> Tensor:
    denom = float(i2p_weight + p2i_weight)
    if denom <= 0:
        return 0.5 * (loss_i2p + loss_p2i)
    return (loss_i2p * i2p_weight + loss_p2i * p2i_weight) / max(denom, eps)


def _weighted_multipos_clip_from_logits(
    logits: Tensor,
    w_pos: Tensor,
    *,
    normalize_pos_weights: bool,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """
    Compute a weighted multi-positive CLIP loss from logits and a nonnegative weight matrix.

    logits: (N, N) image->molecule logits
    w_pos:  (N, N) nonnegative weights; 0 means "not a positive"
    If normalize_pos_weights=True, weights are normalized per-anchor (row-wise) separately for each direction.
    """

    def _dir_loss(lg: Tensor, w: Tensor) -> Tensor:
        # lg: (A, B), w: (A, B)
        logp = lg.log_softmax(dim=-1)

        if normalize_pos_weights:
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=eps)

        logw = torch.where(w > 0, w.log(), torch.full_like(w, -torch.inf))
        return -torch.logsumexp(logp + logw, dim=-1).mean()

    loss_i2p = _dir_loss(logits, w_pos)
    loss_p2i = _dir_loss(logits.t(), w_pos.t())
    return loss_i2p, loss_p2i


class ChemSimSoftMultiPositiveClipLoss(nn.Module):
    """
    Weighted multi-positive CLIP loss with chemical-similarity "soft positives".

    Also tracks lightweight debug stats in `self._last_stats`.
    """

    def __init__(
        self,
        *,
        i2p_weight: float = 1.0,
        p2i_weight: float = 1.0,
        min_sim: float = 0.25,
        topk: int = 5,
        neighbor_scale: float = 0.5,
        neighbor_gamma: float = 1.0,
        compound_level: bool = True,
        normalize_pos_weights: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.i2p_weight = float(i2p_weight)
        self.p2i_weight = float(p2i_weight)

        self.min_sim = float(min_sim)
        self.topk = int(topk)
        self.neighbor_scale = float(neighbor_scale)
        self.neighbor_gamma = float(neighbor_gamma)
        self.compound_level = bool(compound_level)
        self.normalize_pos_weights = bool(normalize_pos_weights)
        self.eps = float(eps)

        self.register_buffer("compound_id_to_sim_index", torch.empty((0,), dtype=torch.long), persistent=False)
        self.register_buffer("compound_sim", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self.register_buffer("alpha", torch.tensor(1.0, dtype=torch.float32), persistent=False)

        self._last_stats: dict[str, torch.Tensor] = {}

    def set_similarity(self, *, compound_id_to_sim_index: torch.Tensor, compound_sim: torch.Tensor) -> None:
        dev = self.alpha.device
        self.compound_id_to_sim_index = compound_id_to_sim_index.to(dtype=torch.long, device=dev)
        self.compound_sim = compound_sim.to(dtype=torch.float32, device=dev)

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(float(alpha))

    def _build_compound_to_compound_weights(self, uniq_ids: Tensor, inv: Tensor) -> Tensor:
        device = uniq_ids.device
        k = int(uniq_ids.numel())
        if k <= 1 or self.topk <= 0:
            return torch.zeros((k, k), dtype=torch.float32, device=device)

        if self.compound_sim.numel() == 0 or self.compound_id_to_sim_index.numel() == 0:
            return torch.zeros((k, k), dtype=torch.float32, device=device)

        id_to_idx = self.compound_id_to_sim_index
        sim_idx = torch.full((k,), -1, dtype=torch.long, device=device)

        in_range = (uniq_ids >= 0) & (uniq_ids < id_to_idx.numel())
        if in_range.any():
            sim_idx[in_range] = id_to_idx[uniq_ids[in_range]]

        valid = sim_idx.ge(0)
        sim_k = torch.zeros((k, k), dtype=torch.float32, device=device)

        if valid.any():
            vpos = torch.nonzero(valid, as_tuple=False).squeeze(1)
            idx = sim_idx[vpos]
            sub = self.compound_sim.index_select(0, idx).index_select(1, idx)
            sim_k[vpos[:, None], vpos[None, :]] = sub

        sim_k.fill_diagonal_(-torch.inf)

        topk = min(self.topk, k - 1)
        vals, inds = sim_k.topk(topk, dim=1)

        keep = torch.isfinite(vals) & vals.ge(self.min_sim)
        w = torch.zeros_like(vals, dtype=torch.float32)

        if keep.any():
            scaled = (vals - self.min_sim) / (1.0 - self.min_sim + self.eps)
            scaled = scaled.clamp(min=0.0, max=1.0).pow(self.neighbor_gamma) * self.neighbor_scale
            scaled = scaled * self.alpha.clamp(min=0.0)
            w = torch.where(keep, scaled.to(dtype=torch.float32), w)

        W = torch.zeros((k, k), dtype=torch.float32, device=device)
        W.scatter_add_(1, inds, w)
        W.fill_diagonal_(0.0)
        return W

    def forward(
        self,
        img: Tensor,
        mol: Tensor,
        *,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        labels: Tensor,
    ) -> Tensor:
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape={tuple(labels.shape)}")

        img_g = _all_gather_with_grad(img)
        mol_g = _all_gather_with_grad(mol)
        lab_g = _all_gather_no_grad(labels.detach().long()).to(device=img_g.device)

        logits = logit_scale * (img_g @ mol_g.t())
        if logit_bias is not None:
            logits = logits + logit_bias

        uniq_ids, inv = torch.unique(lab_g, return_inverse=True)
        k = int(uniq_ids.shape[0])

        M = F.one_hot(inv, num_classes=k).to(dtype=torch.float32)
        w_pos = M @ M.t()

        alpha_val = float(self.alpha.detach().item())

        W_ck = torch.zeros((k, k), dtype=torch.float32, device=img_g.device)
        w_neighbor = torch.zeros_like(w_pos)

        if alpha_val > 0.0 and k > 1 and self.topk > 0:
            W_ck = self._build_compound_to_compound_weights(uniq_ids, inv)

            if self.compound_level:
                counts = torch.bincount(inv, minlength=k).to(dtype=torch.float32).clamp(min=1.0)
                W_ck = W_ck / counts[None, :]

            w_neighbor = M @ W_ck @ M.t()
            w_pos = w_pos + w_neighbor

        loss_i2p, loss_p2i = _weighted_multipos_clip_from_logits(
            logits,
            w_pos,
            normalize_pos_weights=self.normalize_pos_weights,
            eps=self.eps,
        )

        out = _combine_dir_losses(
            loss_i2p,
            loss_p2i,
            i2p_weight=self.i2p_weight,
            p2i_weight=self.p2i_weight,
            eps=self.eps,
        )

        # --- stats (detached, cheap) ---
        with torch.no_grad():
            neigh_mass = w_neighbor.sum(dim=-1)  # (N,)
            has_sample_neighbor = neigh_mass > 0
            n_samples_with_neighbor = has_sample_neighbor.to(torch.float32).sum()
            frac_samples_with_neighbor = has_sample_neighbor.to(torch.float32).mean()

            has_compound_neighbor = W_ck.sum(dim=-1) > 0
            n_compounds_with_neighbor = has_compound_neighbor.to(torch.float32).sum()
            frac_compounds_with_neighbor = (
                has_compound_neighbor.to(torch.float32).mean() if k > 0 else torch.tensor(0.0, device=out.device)
            )

            deg = (W_ck > 0).to(torch.float32).sum(dim=-1)
            mean_neighbors_per_compound = deg.mean() if k > 0 else torch.tensor(0.0, device=out.device)

            self._last_stats = {
                "chemsim_alpha": torch.tensor(alpha_val, device=out.device),
                "chemsim_k_unique_compounds": torch.tensor(float(k), device=out.device),
                "chemsim_n_compounds_with_neighbor": n_compounds_with_neighbor.to(device=out.device),
                "chemsim_n_samples_with_neighbor": n_samples_with_neighbor.to(device=out.device),
                "chemsim_frac_samples_with_neighbor": frac_samples_with_neighbor,
                "chemsim_frac_compounds_with_neighbor": frac_compounds_with_neighbor,
                "chemsim_mean_neighbors_per_compound": mean_neighbors_per_compound,
                "chemsim_mean_neighbor_mass_per_sample": neigh_mass.mean(),
            }

        return out


# Backwards-compatible name (so your CLIPModel branch can stay almost unchanged)
BestNeighborSoftMultiPositiveClipLoss = ChemSimSoftMultiPositiveClipLoss


class MultiPositiveClipLoss(nn.Module):
    """
    Multi-positive CLIP loss: positives are defined by label equality.
    This keeps the original "sum probability mass over positives" behavior by default.
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        *,
        i2p_weight: float = 1.0,
        p2i_weight: float = 1.0,
        eps: float = 1e-12,
        normalize_pos_weights: bool = False,
    ) -> None:
        super().__init__()
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.i2p_weight = float(i2p_weight)
        self.p2i_weight = float(p2i_weight)
        self.eps = float(eps)
        self.normalize_pos_weights = bool(normalize_pos_weights)

    def forward(
        self,
        img: Tensor,
        mol: Tensor,
        *,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        labels: Tensor,
    ) -> Tensor:
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape={tuple(labels.shape)}")

        img_g = _all_gather_with_grad(img)
        mol_g = _all_gather_with_grad(mol)
        lab_g = _all_gather_no_grad(labels.detach().long()).to(device=img_g.device)

        logits = logit_scale * (img_g @ mol_g.t())
        if logit_bias is not None:
            logits = logits + logit_bias

        w_pos = lab_g[:, None].eq(lab_g[None, :]).to(dtype=torch.float32)

        loss_i2p, loss_p2i = _weighted_multipos_clip_from_logits(
            logits,
            w_pos,
            normalize_pos_weights=self.normalize_pos_weights,
            eps=self.eps,
        )

        return _combine_dir_losses(
            loss_i2p,
            loss_p2i,
            i2p_weight=self.i2p_weight,
            p2i_weight=self.p2i_weight,
            eps=self.eps,
        )


class MultiPositiveSigLipLoss(nn.Module):
    """
    Multi-positive SigLIP-style loss:
      positives are label-equal pairs; negatives are label-unequal pairs.
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        *,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

    def forward(
        self,
        img: Tensor,
        mol: Tensor,
        *,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        labels: Tensor,
    ) -> Tensor:
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape={tuple(labels.shape)}")

        img_g = _all_gather_with_grad(img)
        mol_g = _all_gather_with_grad(mol)
        lab_g = _all_gather_no_grad(labels.detach().long())

        logits = logit_scale * (img_g @ mol_g.t())
        if logit_bias is not None:
            logits = logits + logit_bias

        pos = lab_g[:, None].eq(lab_g[None, :])
        neg = ~pos

        pos_vals = logits[pos]
        neg_vals = logits[neg]

        if pos_vals.numel() == 0:
            raise RuntimeError("No positive pairs in batch for MultiPositiveSigLipLoss (check labels).")
        if neg_vals.numel() == 0:
            raise RuntimeError("No negative pairs in batch for MultiPositiveSigLipLoss (batch collapsed to 1 label).")

        pos_loss = F.softplus(-pos_vals).mean()
        neg_loss = F.softplus(neg_vals).mean()

        return self.pos_weight * pos_loss + self.neg_weight * neg_loss


def _batch_labels_from_compound(batch: dict[str, Any], device: torch.device | None = None) -> torch.Tensor | None:
    """
    Build per-sample integer labels for compound identity.

    Priority:
      1) batch["compound_id"] if present
      2) batch["perturbation"] / batch["Metadata_SMILES"] / batch["smiles"] / batch["SMILES"] (factorized per batch)

    Returns: LongTensor on `device` if provided, else CPU. No grad.
    """
    target = device

    # Already encoded integer labels
    if "compound_id" in batch:
        val = batch["compound_id"]
        if torch.is_tensor(val):
            t = val.detach().long()
            return t.to(target, non_blocking=True) if target is not None else t.to("cpu", non_blocking=True)
        if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (int, np.integer)):
            t = torch.as_tensor(val, dtype=torch.long)
            return t.to(target, non_blocking=True) if target is not None else t

    # Factorize strings per batch (fallback)
    for key in ("perturbation", "Metadata_SMILES", "smiles", "SMILES"):
        if key in batch:
            val = batch[key]
            if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], str):
                uniq: dict[str, int] = {}
                labels = [uniq.setdefault(s, len(uniq)) for s in val]
                t = torch.as_tensor(labels, dtype=torch.long)
                return t.to(target, non_blocking=True) if target is not None else t

    return None


class PretrainModule(LightningModule):
    def __init__(
        self,
        embed_dim: int,
        image_encoder_name: Literal["densenet", "openphenom", "dinov3", "subcell"],
        perturbation_encoder_name: Literal[
            "chemberta", "ecfp", "whimf", "precomputed", "random", "ecfp_cached", "gene_lookup"
        ],
        loss: Literal["CLIP", "CLIP_multi", "SIGLIP", "SIGLIP_multi"],
        image_size: int,
        temperature: float,
        lr: float,
        weight_decay: float,
        cfg: DictConfig,
        strategy: Any,
        *,
        in_channels: int = 5,
        image_lora: DictConfig | dict | bool | None = None,
        perturbation_lora: DictConfig | dict | bool | None = None,
        freeze_backbone_when_no_lora: bool = True,
        perturbation_embedding_column: str | None = None,
        precomputed_in_dim: int | None = None,
        image_channelwise: bool = False,
        perturbation_head: Literal["mlp", "linear"] = "mlp",
        perturbation_head_dropout: float = 0.0,
        fingerprint_path: str = "",
        key_column: str = "Metadata_SMILES",
        embedding_column: str = "",
        sirna_to_gene_path: str = "",
        control_key: str = "",
        # Legacy parameter names kept for backward compatibility
        molecule_encoder_name: str | None = None,
        molecule_lora: DictConfig | dict | bool | None = None,
        molecule_embedding_column: str | None = None,
        molecule_head: Literal["mlp", "linear"] | None = None,
        molecule_head_dropout: float | None = None,
        image_encoder_channel_names: list[str] | None = None,
        image_channel_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        # Support legacy parameter names
        if molecule_encoder_name is not None:
            perturbation_encoder_name = molecule_encoder_name
        if molecule_lora is not None and perturbation_lora is None:
            perturbation_lora = molecule_lora
        if molecule_embedding_column is not None and perturbation_embedding_column is None:
            perturbation_embedding_column = molecule_embedding_column
        if molecule_head is not None:
            perturbation_head = molecule_head
        if molecule_head_dropout is not None:
            perturbation_head_dropout = molecule_head_dropout

        lg = getattr(self.cfg, "logging", None)
        self.train_loss_every = int(getattr(lg, "train_loss_every_n_steps", 1)) if lg else 1

        self.save_hyperparameters(ignore=["cfg", "strategy"])

        self.perturbation_embedding_column = perturbation_embedding_column
        local_rank, global_rank, world_size = world_info_from_env()
        logger.info(f"LOCAL RANK: {local_rank}")
        logger.info(f"GLOBAL RANK: {global_rank}")
        logger.info(f"WORLD SIZE: {world_size}")

        def _normalize(x):
            if isinstance(x, bool):
                return {"enabled": bool(x)}
            if isinstance(x, DictConfig):
                return OmegaConf.to_container(x, resolve=True)  # type: ignore
            return x or {}

        image_lora_cfg = _normalize(image_lora)
        perturbation_lora_cfg = _normalize(perturbation_lora)

        self.in_channels = int(in_channels)

        self.model = CLIPModel(
            embed_dim=embed_dim,
            image_encoder_name=image_encoder_name,
            perturbation_encoder_name=perturbation_encoder_name,
            loss=loss,
            image_size=image_size,
            in_channels=self.in_channels,
            temperature=temperature,
            world_size=world_size,
            rank=local_rank,
            image_lora_cfg=image_lora_cfg,
            perturbation_lora_cfg=perturbation_lora_cfg,
            freeze_backbone_when_no_lora=freeze_backbone_when_no_lora,
            precomputed_in_dim=precomputed_in_dim,
            image_channelwise=bool(image_channelwise),
            perturbation_projection_head=perturbation_head,
            perturbation_head_dropout=float(perturbation_head_dropout),
            fingerprint_path=fingerprint_path,
            key_column=key_column,
            embedding_column=embedding_column,
            sirna_to_gene_path=sirna_to_gene_path,
            control_key=control_key,
            image_encoder_channel_names=image_encoder_channel_names,
            image_channel_mode=image_channel_mode,
        )

        self.model.image_encoder.log_trainable_summary(logger)
        self.model.perturbation_encoder.log_trainable_summary(logger)

        self.lr = lr
        self.weight_decay = weight_decay
        self.strategy = strategy

    def on_train_epoch_start(self) -> None:
        self._train_img_embs: list[torch.Tensor] = []
        self._train_pert_embs: list[torch.Tensor] = []
        self._train_compound_ids: list[torch.Tensor] = []

    def on_validation_epoch_start(self) -> None:
        self._val_img_embs: list[torch.Tensor] = []
        self._val_pert_embs: list[torch.Tensor] = []
        self._val_compound_ids: list[torch.Tensor] = []

    @torch.inference_mode()
    def on_train_epoch_end(self) -> None:
        if not getattr(self, "_train_img_embs", None):
            return

        ie = torch.cat(self._train_img_embs, dim=0)
        me = torch.cat(self._train_pert_embs, dim=0)
        lab = torch.cat(self._train_compound_ids, dim=0)

        dev: torch.device | str
        if self.device.type == "cuda":
            dev = self.device
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        metrics = full_val_recall_compound_level(
            ie,
            me,
            lab,
            recall_range=(1, 5, 10),
            chunk_size=2048,
            device=dev,
            normalize=False,
            compute_macro_i2p=True,
            prefix="train",
        )
        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)

        self._train_img_embs.clear()
        self._train_pert_embs.clear()
        self._train_compound_ids.clear()

    def on_fit_start(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return

        fps = getattr(dm, "train_compound_fps_log1p", None)
        cids = getattr(dm, "train_compound_ids_unique", None)
        if fps is None or cids is None:
            logger.info("No cached train compound fps found on datamodule; skipping similarity precompute.")
            return

        if not hasattr(self.model, "loss") or not hasattr(self.model.loss, "set_similarity"):
            logger.info("Model loss does not support set_similarity(); skipping similarity precompute.")
            return

        sim_cpu = _tanimoto_matrix_binary_from_log1p_torch(fps)  # (C,C) float32 on CPU

        max_id = int(np.max(cids))
        id_to_idx = torch.full((max_id + 1,), -1, dtype=torch.long)
        id_to_idx[torch.from_numpy(cids.astype(np.int64))] = torch.arange(int(cids.shape[0]), dtype=torch.long)

        dev = self.model.logit_scale.device
        self.model.loss.set_similarity(
            compound_id_to_sim_index=id_to_idx.to(device=dev),
            compound_sim=sim_cpu.to(device=dev),
        )

        # Optional: start with alpha=0 and ramp later (keeps the “start simple” option available)
        # self.model.loss.set_alpha(0.0)

        logger.info(f"Precomputed compound similarity: n_train_compounds={int(cids.shape[0])}")

    def configure_optimizers(self):
        head_lr = float(getattr(self.cfg.model.optimizer, "lr_head", 3e-4))
        lora_lr = float(getattr(self.cfg.model.optimizer, "lr_lora", 1e-4))
        temp_lr = float(getattr(self.cfg.model.optimizer, "lr_temp", 1e-5))
        wd = float(self.weight_decay)

        head_decay, head_no_decay, lora_params, temp_param = [], [], [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("logit_scale"):
                temp_param.append(p)
                continue
            if "lora" in name.lower():
                lora_params.append(p)
                continue
            n = name.lower()
            if n.endswith(".bias") or "layernorm" in n or ".ln" in n or ".norm" in n:
                head_no_decay.append(p)
            else:
                head_decay.append(p)

        param_groups: list[dict] = []
        if head_decay:
            param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": wd})
        if head_no_decay:
            param_groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": 0.0})
        if temp_param:
            param_groups.append({"params": temp_param, "lr": temp_lr, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        # Build the cosine schedule from the trainer's step budget, which
        # already accounts for max_epochs, dataloader length, and
        # accumulate_grad_batches. Runs after datamodule setup, so the
        # stepping count is exact.
        est = self.trainer.estimated_stepping_batches
        if not math.isfinite(est) or est < 1:
            raise ValueError(
                f"Cannot build LR schedule: trainer.estimated_stepping_batches={est}. "
                "Set max_epochs (or max_steps) and ensure the train dataloader is non-empty."
            )
        total_steps = int(est)
        sched_cfg = getattr(self.cfg.model, "lr_schedule", None)
        warmup_frac = float(getattr(sched_cfg, "warmup_frac", 0.1)) if sched_cfg is not None else 0.1
        num_cycles = float(getattr(sched_cfg, "num_cycles", 0.5)) if sched_cfg is not None else 0.5
        warmup_steps = max(1, int(warmup_frac * total_steps))

        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # ---------------- steps ----------------
    def training_step(self, batch, batch_idx):
        img_emb, pert_emb = self.model(batch)

        labels = None
        if getattr(self.model, "loss_requires_labels", False):
            labels = _batch_labels_from_compound(batch, device=img_emb.device)
            if labels is None:
                raise KeyError(
                    "Multi-positive loss selected but no labels found in batch. "
                    'Provide batch["compound_id"] (recommended) or a "perturbation" column.'
                )

        loss = self.model.compute_loss(img_emb, pert_emb, labels=labels)

        img = batch[DatasetEnum.IMG]
        bs = int(img.shape[0]) if torch.is_tensor(img) else int(len(img))
        step = int(self.global_step) + 1

        if step % self.train_loss_every == 0:
            self.log(
                "train_loss_step",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
                batch_size=bs,
            )

        self.log(
            "logit_scale",
            self.model.logit_scale.detach().exp(),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )

        # Accumulate embeddings for the end-of-epoch train retrieval metrics.
        # Buffers are allocated in on_train_epoch_start; CPU + float16 to bound memory.
        lab = _batch_labels_from_compound(batch, device=img_emb.device)
        if lab is not None:
            self._train_img_embs.append(img_emb.detach().to(dtype=torch.float16, device="cpu"))
            self._train_pert_embs.append(pert_emb.detach().to(dtype=torch.float16, device="cpu"))
            self._train_compound_ids.append(lab.detach().to(dtype=torch.long, device="cpu"))

        return loss

    @torch.inference_mode()
    def on_validation_epoch_end(self) -> None:
        if not hasattr(self, "_val_img_embs") or len(self._val_img_embs) == 0:
            return

        ie = torch.cat(self._val_img_embs, dim=0)
        me = torch.cat(self._val_pert_embs, dim=0)
        lab = torch.cat(self._val_compound_ids, dim=0)

        # Pick device for similarity/topk. Keep it on GPU if available.
        dev: torch.device | str
        if self.device.type == "cuda":
            dev = self.device
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        metrics = full_val_recall_compound_level(
            ie,
            me,
            lab,
            recall_range=(1, 5, 10),
            chunk_size=2048,
            device=dev,
            normalize=False,  # embeddings are already normalized in CLIPModel.forward()
            compute_macro_i2p=True,  # recommended; you can set False if you only want micro
        )

        # Log once per epoch. rank_zero_only prevents duplicates if you ever go DDP.
        # val_R@1_I2P_macro is the checkpoint/progress-bar target.
        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=(k == "val_R@1_I2P_macro"), sync_dist=True, rank_zero_only=True)

        # Free memory
        self._val_img_embs.clear()
        self._val_pert_embs.clear()
        self._val_compound_ids.clear()

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        img_emb, pert_emb = self.model(batch)

        labels = None
        if getattr(self.model, "loss_requires_labels", False):
            labels = _batch_labels_from_compound(batch, device=img_emb.device)
            if labels is None:
                raise KeyError(
                    "Multi-positive loss selected but no labels found in batch. "
                    'Provide batch["compound_id"] (recommended) or a "perturbation" column.'
                )

        loss = self.model.compute_loss(img_emb, pert_emb, labels=labels)

        img = batch[DatasetEnum.IMG]
        bs = int(img.shape[0]) if torch.is_tensor(img) else int(len(img))

        if dataloader_idx == 0:
            self.log("val_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)

            lab = _batch_labels_from_compound(batch, device=img_emb.device)
            if lab is None:
                raise KeyError(
                    "Full validation retrieval requires stable compound ids. "
                    'Your datamodule should provide batch["compound_id"].'
                )

            # Store on CPU to avoid GPU memory growth. float16 is usually fine for retrieval metrics.
            self._val_img_embs.append(img_emb.detach().to(dtype=torch.float16, device="cpu"))
            self._val_pert_embs.append(pert_emb.detach().to(dtype=torch.float16, device="cpu"))
            self._val_compound_ids.append(lab.detach().to(dtype=torch.long, device="cpu"))

        return {"val_loss": loss}

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        img_emb, pert_emb = self.model(batch)

        labels = None
        if getattr(self.model, "loss_requires_labels", False):
            labels = _batch_labels_from_compound(batch, device=img_emb.device)
            if labels is None:
                raise KeyError(
                    "Multi-positive loss selected but no labels found in batch. "
                    'Provide batch["compound_id"] (recommended) or a "perturbation" column.'
                )

        loss = self.model.compute_loss(img_emb, pert_emb, labels=labels)

        img = batch[DatasetEnum.IMG]
        bs = int(img.shape[0]) if torch.is_tensor(img) else int(len(img))

        self.log("test_loss_step", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        return {"test_loss_step": loss}


def _tanimoto_matrix_binary_from_log1p_torch(fps_log1p: np.ndarray) -> torch.Tensor:
    """
    fps_log1p: (N,D) float32 log1p(counts)
    returns: (N,N) float32 tanimoto on (fps>0)
    """
    x = torch.from_numpy(np.asarray(fps_log1p) > 0).to(dtype=torch.int32)  # (N,D) {0,1}
    inter = x @ x.t()  # (N,N) int32
    pop = x.sum(dim=1, dtype=torch.int32)  # (N,)
    denom = pop[:, None] + pop[None, :] - inter
    out = torch.zeros_like(denom, dtype=torch.float32)
    out = out.masked_scatter(denom != 0, (inter[denom != 0].to(torch.float32) / denom[denom != 0].to(torch.float32)))
    return out


class ImageGpuPreprocess(nn.Module):
    def __init__(
        self,
        *,
        out_hw: tuple[int, int],
        out_dtype: torch.dtype = torch.float32,
        scale_uint8_to_unit: bool = True,
    ) -> None:
        super().__init__()
        self.out_hw = out_hw
        self.out_dtype = out_dtype
        self.scale_uint8_to_unit = bool(scale_uint8_to_unit)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W), typically uint8 from the dataloader
        if x.dtype == torch.uint8:
            x = x.to(dtype=self.out_dtype)
            if self.scale_uint8_to_unit:
                x = x.div_(255.0)
        else:
            x = x.to(dtype=self.out_dtype)

        if (x.shape[-2], x.shape[-1]) != self.out_hw:
            x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)

        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_encoder_name: Literal["densenet", "openphenom", "dinov3", "subcell"],
        perturbation_encoder_name: Literal[
            "chemberta", "ecfp", "WHIMF", "precomputed", "random", "ecfp_cached", "gene_lookup"
        ],
        loss: Literal["CLIP", "SIGLIP", "CLIP_multi", "SIGLIP_multi"],
        image_size: int = 224,
        temperature: float = 0.07,
        world_size: int = 1,
        rank: int = 0,
        *,
        in_channels: int = 5,
        image_lora_cfg: dict[str, Any] | None = None,
        perturbation_lora_cfg: dict[str, Any] | None = None,
        freeze_backbone_when_no_lora: bool = True,
        precomputed_in_dim: int | None = None,
        image_channelwise: bool = False,
        perturbation_projection_head: Literal["mlp", "linear"] = "mlp",
        perturbation_head_dropout: float = 0.0,
        fingerprint_path: str = "",
        key_column: str = "Metadata_SMILES",
        embedding_column: str = "",
        sirna_to_gene_path: str = "",
        control_key: str = "",
        # Legacy aliases kept for compatibility with saved hyperparameters
        molecule_encoder_name: str | None = None,
        molecule_lora_cfg: dict[str, Any] | None = None,
        molecule_projection_head: str | None = None,
        molecule_head_dropout: float | None = None,
        image_encoder_channel_names: list[str] | None = None,
        image_channel_mode: str | None = None,
    ):
        super().__init__()

        # Resolve legacy param names
        if molecule_encoder_name is not None:
            perturbation_encoder_name = molecule_encoder_name
        if molecule_lora_cfg is not None and perturbation_lora_cfg is None:
            perturbation_lora_cfg = molecule_lora_cfg
        if molecule_projection_head is not None:
            perturbation_projection_head = molecule_projection_head
        if molecule_head_dropout is not None:
            perturbation_head_dropout = molecule_head_dropout

        self.in_channels = int(in_channels)

        encoder_overrides: dict[str, Any] = dict(
            embed_dim=embed_dim,
            image_size=image_size,
            in_channels=self.in_channels,
            lora=(LoraConfig(**image_lora_cfg) if isinstance(image_lora_cfg, dict) else image_lora_cfg),
            freeze_backbone_when_no_lora=freeze_backbone_when_no_lora,
            channelwise=bool(image_channelwise),
            channel_names=(tuple(image_encoder_channel_names) if image_encoder_channel_names is not None else None),
        )
        if image_channel_mode is not None:
            encoder_overrides["channel_mode"] = image_channel_mode

        self.image_encoder = ImageEncoderRegistry.build_from_name(
            image_encoder_name.lower(),
            **encoder_overrides,
        )

        pert_name = perturbation_encoder_name.lower()
        pert_kwargs: dict[str, Any] = {
            "embed_dim": embed_dim,
            "lora": (
                LoraConfig(**perturbation_lora_cfg)
                if isinstance(perturbation_lora_cfg, dict)
                else perturbation_lora_cfg
            ),
            "projection_head": perturbation_projection_head,
            "dropout": float(perturbation_head_dropout),
            # Unknown fields are silently dropped for encoders that don't declare them,
            # via _filter_overrides in build_from_name.
            "fingerprint_path": fingerprint_path,
            "key_column": key_column,
            "embedding_column": embedding_column,
            "sirna_to_gene_path": sirna_to_gene_path,
            "control_key": control_key,
        }

        if pert_name == "precomputed":
            # If None -> use lazy linear head via in_dim <= 0
            pert_kwargs["in_dim"] = int(precomputed_in_dim) if precomputed_in_dim is not None else 0

        self.perturbation_encoder = MoleculeEncoderRegistry.build_from_name(pert_name, **pert_kwargs)

        self.image_preprocess = ImageGpuPreprocess(
            out_hw=(int(image_size), int(image_size)),
            out_dtype=torch.float32,
            scale_uint8_to_unit=True,
        )

        self.temperature = float(temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / self.temperature))

        loss_key = str(loss).strip().lower()
        self.loss_requires_labels = False
        self.logit_bias = None

        if loss_key == "clip":
            self.loss = ClipLoss(world_size=world_size, rank=rank)
        elif loss_key in {"clip_multi", "clip_multi_positive", "clip_multipos"}:
            self.loss = MultiPositiveClipLoss()
            self.loss_requires_labels = True
        elif loss_key in {"clip_soft", "clip_soft_neighbor", "clip_softpos", "clip_soft_positive"}:
            self.loss = BestNeighborSoftMultiPositiveClipLoss(
                min_sim=0.25,
                topk=5,
                neighbor_scale=0.5,
                neighbor_gamma=1.0,
                compound_level=False,
                normalize_pos_weights=False,
            )
            self.loss_requires_labels = True
        elif loss_key == "siglip":
            self.logit_bias = nn.Parameter(torch.ones([]) * -10)
            self.loss = SigLipLoss(world_size=world_size, rank=rank)
        elif loss_key in {"siglip_multi", "siglip_multi_positive", "siglip_multipos"}:
            self.logit_bias = nn.Parameter(torch.ones([]) * -10)
            self.loss = MultiPositiveSigLipLoss()
            self.loss_requires_labels = True
        else:
            raise ValueError(f"Loss {loss} not supported.")

    def get_trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = self.image_encoder.trainable_param_groups()

        if hasattr(self.perturbation_encoder, "trainable_param_groups"):
            pert_groups = self.perturbation_encoder.trainable_param_groups()
            groups["head"].extend(pert_groups.get("head", []))
            groups["lora"].extend(pert_groups.get("lora", []))
        else:
            pert_head = [p for p in self.perturbation_encoder.head.parameters() if p.requires_grad]
            groups["head"].extend(pert_head)

        return groups

    def forward(self, batch: dict[str, Tensor], norm: bool = True):
        img = batch[DatasetEnum.IMG]
        img = self.image_preprocess(img)  # unified, GPU-side

        img_embed = self.image_encoder(img)
        pert_embed = self.perturbation_encoder(batch[DatasetEnum.PERTURBATION])

        if norm:
            img_embed = F.normalize(img_embed, p=2, dim=-1)
            pert_embed = F.normalize(pert_embed, p=2, dim=-1)

        return img_embed, pert_embed

    def compute_loss(self, img_embed: Tensor, pert_embed: Tensor, labels: Tensor | None = None) -> Tensor:
        logit_scale = self.logit_scale.clamp(min=np.log(1 / 100), max=np.log(100)).exp()

        if self.loss_requires_labels:
            if labels is None:
                raise ValueError("This loss requires labels but labels=None was passed.")
            labels = labels.to(device=img_embed.device, non_blocking=True).long()

            if self.logit_bias is not None:
                return self.loss(
                    img_embed,
                    pert_embed,
                    logit_scale=logit_scale,
                    logit_bias=self.logit_bias,
                    labels=labels,
                )
            return self.loss(img_embed, pert_embed, logit_scale=logit_scale, labels=labels)

        if self.logit_bias is not None:
            return self.loss(img_embed, pert_embed, logit_scale=logit_scale, logit_bias=self.logit_bias)
        return self.loss(img_embed, pert_embed, logit_scale=logit_scale)
