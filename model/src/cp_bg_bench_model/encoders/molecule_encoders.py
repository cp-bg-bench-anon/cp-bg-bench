from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from dataclasses import fields as dc_fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from cp_bg_bench_model._heads import LinearHead, ProjectionHead
from cp_bg_bench_model.lora import LoraConfig, apply_lora_if_enabled, collect_lora_params

# =========================================================
# Internal helpers (fingerprints + optional tokenizer)
# =========================================================


class _FP(nn.Module):
    """
    Lightweight wrapper around scikit-fingerprints (skfp) to compute ECFP or WHIMF.
    Accepts a list[str] of SMILES and returns a float32 tensor on CPU.
    """

    def __init__(self, fp_type: Literal["ecfp", "whimf"]):
        super().__init__()
        self.fp_type = fp_type
        try:
            from skfp.fingerprints import ECFPFingerprint, WHIMFingerprint
            from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
        except Exception as e:
            raise ImportError(
                "scikit-fingerprints (skfp) is required for ECFP/WHIMF encoders. "
                "Install it or switch to a different encoder."
            ) from e

        # cache transformers
        self._mol_from_smiles = MolFromSmilesTransformer()
        self._ecfp = ECFPFingerprint() if fp_type == "ecfp" else None
        self._whim = WHIMFingerprint() if fp_type == "whimf" else None
        self._confgen = ConformerGenerator() if fp_type == "whimf" else None

    @torch.no_grad()
    def forward(self, smiles: list[str]) -> torch.Tensor:
        if not isinstance(smiles, (list, tuple)) or (len(smiles) > 0 and not isinstance(smiles[0], str)):
            raise TypeError("Expected a list[str] of SMILES.")
        mols = self._mol_from_smiles.transform(smiles)

        if self.fp_type == "ecfp":
            fps = self._ecfp.transform(mols)  # ndarray [B, 2048]
        else:  # whimf
            mols3d = self._confgen.transform(mols)
            fps = self._whim.transform(mols3d)  # ndarray [B, 114]

        return torch.tensor(fps, dtype=torch.float32)


def chemberta_concat_tokenizer(
    hf_model: str = "DeepChem/ChemBERTa-77M-MLM",
    max_length: int = 128,
):
    """
    Optional helper (for your datamodule): returns a function that tokenizes SMILES
    and concatenates [input_ids | attention_mask] along dim=1 (legacy format).
    """
    tok = AutoTokenizer.from_pretrained(hf_model, use_fast=True)

    def _fn(smiles: list[str] | str) -> torch.Tensor:
        if isinstance(smiles, str):
            smiles = [smiles]
        inputs = tok(
            smiles,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return torch.cat([inputs["input_ids"], inputs["attention_mask"]], dim=1)

    return _fn


# --- LoRA target inference (for HF encoders) ---
def _infer_lora_targets(backbone: nn.Module) -> list[str]:
    """
    Heuristic: choose a sensible set of attention projection names for PEFT target_modules.
    Prefers q/k/v/out style; falls back to BERT/RoBERTa query/key/value; finally to in/out proj.
    """
    names = [n for n, _ in backbone.named_modules()]

    candidates = [
        ["q_proj", "k_proj", "v_proj", "out_proj"],  # GPT/OPT/BLOOM-like
        ["query", "key", "value"],  # BERT/RoBERTa self-attn
        ["in_proj", "attn_out_proj"],  # custom variants
    ]

    def coverage(ts: list[str]) -> int:
        return sum(any(t in n for n in names) for t in ts)

    candidates.sort(key=coverage, reverse=True)
    best = candidates[0]
    # ensure at least one actually matches; otherwise return a conservative default
    return best if coverage(best) > 0 else ["query", "key", "value"]


# =========================================================
# Config dataclasses (internal)
# =========================================================


@dataclass(frozen=True)
class MoleculeEncoderConfig:
    """Base config shared by all molecule encoders (internal use)."""

    name: str
    embed_dim: int = 128
    projection_head: Literal["mlp", "linear"] = "mlp"


@dataclass(frozen=True)
class ChembertaConfig(MoleculeEncoderConfig):
    """Config for ChemBERTa-like encoders (HF)."""

    hf_model: str = "DeepChem/ChemBERTa-77M-MLM"
    pool: Literal["cls", "mean"] = "cls"
    freeze_backbone: bool = True  # kept for backwards-compat; see freeze_backbone_when_no_lora
    tokenizer_max_length: int | None = None  # None => tokenizer default
    lora: LoraConfig | None = None
    freeze_backbone_when_no_lora: bool = True


@dataclass(frozen=True)
class PresetEncoderConfig(MoleculeEncoderConfig):
    """
    Config for fixed-dimension / classical features.
    family:
      • 'ecfp'       -> compute ECFP from SMILES on the fly (in_dim ignored)
      • 'whimf'      -> compute WHIMF from SMILES on the fly (in_dim ignored)
      • 'precomputed'-> treat inputs as dense [B, in_dim]
    """

    family: Literal["ecfp", "whimf", "precomputed"] = "ecfp"
    in_dim: int = 2048
    dropout: float = 0.0


# =========================================================
# Base & concrete encoders (unified interface)
# =========================================================


class MoleculeEncoderBase(nn.Module):
    """
    Unified interface:
      forward(x) accepts:
        • list/tuple[str]      -> perturbation strings: SMILES (compounds) or gene symbols (siRNA/CRISPR)
        • dict[str, Tensor]    -> HF-style tokenized inputs
        • Tensor               -> dense features (or legacy concat [ids|mask])
    Encoders implement any subset of:
        encode_strings(strings), encode_smiles(smiles), encode_tokenized(tok), encode_dense(x)

    Dispatch order for list[str]: supports_strings → encode_strings, supports_smiles → encode_smiles.
    Use supports_strings for gene-symbol encoders; supports_smiles for SMILES-based encoders.
    """

    supports_smiles = False
    supports_strings = False
    supports_tokenized = False
    supports_dense = True

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    # ----- optional hooks for subclasses -----
    def encode_strings(self, strings: list[str]) -> torch.Tensor:
        raise NotImplementedError("encode_strings not implemented for this encoder")

    def encode_smiles(self, smiles: list[str]) -> torch.Tensor:
        raise NotImplementedError("encode_smiles not implemented for this encoder")

    def encode_tokenized(self, tok: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("encode_tokenized not implemented for this encoder")

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        # default: identity (precomputed features)
        return x

    def trainable_param_groups(self) -> dict[str, list[nn.Parameter]]:
        head_params = [p for p in getattr(self, "head", nn.Identity()).parameters() if p.requires_grad]
        lora_params = collect_lora_params(self)  # LoRA adapters anywhere in this module
        return {"head": head_params, "lora": lora_params}

    @staticmethod
    def _count_params(m: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in m.parameters())
        train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, train

    def log_trainable_summary(self, logger) -> None:
        trunk = getattr(self, "model", None)
        head = getattr(self, "head", nn.Identity())

        # counts
        trunk_total, trunk_train = self._count_params(trunk) if trunk is not None else (0, 0)
        head_total, head_train = self._count_params(head)

        # head io (robust to Linear/ProjectionHead)
        def _head_io(h: nn.Module):
            kind = "linear" if isinstance(h, nn.Linear) else "mlp"
            in_d = getattr(h, "in_features", None)
            out_d = getattr(h, "embed_dim", None) or getattr(h, "out_features", None)
            # try a nested final linear commonly named "proj"
            if out_d is None and hasattr(h, "proj"):
                out_d = getattr(h.proj, "out_features", None)
                in_d = in_d or getattr(h.proj, "in_features", None)
            return kind, in_d, out_d

        head_kind, in_d, out_d = _head_io(head)

        # lora only under the trunk (to mirror image logging)
        from cp_bg_bench_model.lora import collect_lora_params

        lora_train = sum(p.numel() for p in collect_lora_params(trunk)) if trunk is not None else 0
        using_lora = bool(getattr(self, "using_lora", False))

        # optional provenance (for precomputed)
        source_col = getattr(self, "source_column", None)

        parts = [
            f"PerturbationEncoder[{type(self).__name__}] "
            f"| trunk: total={trunk_total / 1e6:.1f}M, train={trunk_train / 1e6:.3f}M ",
            f"| head train={head_train / 1e6:.3f}M ",
        ]
        if in_d is not None and out_d is not None:
            parts.append(f"| proj: {in_d}→{out_d} ({head_kind}) ")
        parts.append(f"| lora train={lora_train / 1e6:.3f}M | using_lora={using_lora}")
        if source_col:
            parts.append(f" | source='{source_col}'")

        logger.info("".join(parts))

    # ----- legacy helper: split concat [ids|mask] -> dict -----
    @staticmethod
    def _concat_to_hf_dict(x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2 or (x.shape[1] % 2) != 0:
            raise RuntimeError(f"Expected [B, 2*L] concat tensor, got {tuple(x.shape)}")
        half = x.shape[1] // 2
        return {"input_ids": x[:, :half].long(), "attention_mask": x[:, half:].long()}

    # ----- the unified public entrypoint -----
    def forward(self, x: object) -> torch.Tensor:
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], str):
            if self.supports_strings:
                return self.encode_strings(list(x))
            if self.supports_smiles:
                return self.encode_smiles(list(x))
            raise TypeError(f"{type(self).__name__} does not accept string inputs")

        # HF tokenized dict
        if isinstance(x, dict):
            if not self.supports_tokenized:
                raise TypeError(f"{type(self).__name__} does not accept tokenized inputs")
            return self.encode_tokenized(x)  # type: ignore[arg-type]

        # Tensor: dense or legacy concat
        if isinstance(x, torch.Tensor):
            # legacy concat ids|mask
            if x.ndim == 2 and x.dtype in (torch.int64, torch.int32) and (x.shape[1] % 2) == 0:
                tok = self._concat_to_hf_dict(x)
                if not self.supports_tokenized:
                    raise TypeError(f"{type(self).__name__} cannot decode tokenized concat input")
                return self.encode_tokenized(tok)
            # dense features
            if not self.supports_dense:
                raise TypeError(f"{type(self).__name__} does not accept dense feature tensors")
            return self.encode_dense(x)

        raise TypeError(f"Unsupported input type for {type(self).__name__}: {type(x)}")


def _make_head(
    in_features: int,
    embed_dim: int,
    kind: Literal["mlp", "linear"],
    *,
    dropout: float = 0.0,
    activation: Literal["geglu", "silu"] = "geglu",
) -> nn.Module:
    if kind == "linear":
        do = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # If in_features <= 0, build lazily on first forward (supports varying precomputed dims)
        if int(in_features) > 0:
            proj: nn.Module = nn.Linear(int(in_features), int(embed_dim), bias=True)
        else:
            proj = nn.LazyLinear(int(embed_dim), bias=True)

        # LN after projection (dimension known = embed_dim)
        ln = nn.LayerNorm(int(embed_dim))
        return nn.Sequential(do, proj, ln)

    return ProjectionHead(
        in_features=int(in_features),
        embed_dim=int(embed_dim),
        width_alpha=1.2,
        round_to=64,
        min_hidden=256,
        max_hidden=4096,
        dropout=float(dropout),
        use_skip=True,
        activation=activation,
    )


class ChembertaEncoder(MoleculeEncoderBase):
    supports_smiles = True
    supports_tokenized = True
    supports_dense = False

    def __init__(self, cfg: ChembertaConfig):
        super().__init__(cfg.embed_dim)

        # backbone + tokenizer
        self.model = AutoModel.from_pretrained(cfg.hf_model)
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_model, use_fast=True)

        # LoRA: target attention projections by default
        default_targets = _infer_lora_targets(self.model)
        self.model, using_lora = apply_lora_if_enabled(
            self.model,
            cfg.lora,
            default_targets=default_targets,
            freeze_if_disabled=cfg.freeze_backbone_when_no_lora,
            freeze_bn=False,  # transformers have no 2D BN
        )
        self.using_lora = using_lora

        self.pool = cfg.pool
        self.max_len: int | None = cfg.tokenizer_max_length

        self.head = _make_head(
            in_features=self.model.config.hidden_size,
            embed_dim=cfg.embed_dim,
            kind=cfg.projection_head,
            dropout=0.0,
            activation="geglu",
        )

        # optional: GC to reduce memory when LoRA is active
        if self.using_lora and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # hidden: [B, L, D], mask: [B, L]
        if self.pool == "cls":
            return hidden[:, 0, :]
        if self.pool == "mean":
            if mask is None:
                return hidden.mean(dim=1)
            lengths = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            return (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths
        raise ValueError(f"Unknown pool mode: {self.pool}")

    def encode_smiles(self, smiles: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tok_kwargs: dict[str, Any] = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.max_len is not None:
            tok_kwargs["max_length"] = int(self.max_len)
        batch = self.tok(smiles, **tok_kwargs)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = self.model(**batch).last_hidden_state
        pooled = self._pool(out, batch.get("attention_mask"))
        return self.head(pooled)

    def encode_tokenized(self, tok: dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.model(**tok).last_hidden_state
        pooled = self._pool(out, tok.get("attention_mask"))
        return self.head(pooled)


class ECFPEncoder(MoleculeEncoderBase):
    supports_smiles = True
    supports_tokenized = False
    supports_dense = True

    def __init__(self, cfg: PresetEncoderConfig):
        super().__init__(cfg.embed_dim)
        assert cfg.family == "ecfp"
        self.fp = _FP("ecfp")
        self.in_dim = 2048  # ECFP nBits (skfp default)
        self.head = _make_head(
            in_features=self.in_dim,
            embed_dim=cfg.embed_dim,
            kind=cfg.projection_head,
            dropout=cfg.dropout,
            activation="silu",
        )

    def encode_smiles(self, smiles: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self.fp(smiles).to(device)  # [B, 2048]
        return self.head(x)

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        # assumes [B, 2048]
        return self.head(x)


class WHIMFEncoder(MoleculeEncoderBase):
    supports_smiles = True
    supports_tokenized = False
    supports_dense = True

    def __init__(self, cfg: PresetEncoderConfig):
        super().__init__(cfg.embed_dim)
        assert cfg.family == "whimf"
        self.fp = _FP("whimf")
        self.in_dim = 114
        self.head = _make_head(
            in_features=self.in_dim,
            embed_dim=cfg.embed_dim,
            kind=cfg.projection_head,
            dropout=cfg.dropout,
            activation="silu",
        )

    def encode_smiles(self, smiles: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self.fp(smiles).to(device)  # [B, 114]
        return self.head(x)

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        # assumes [B, 114]
        return self.head(x)


class PrecomputedDenseEncoder(MoleculeEncoderBase):
    supports_smiles = False
    supports_tokenized = False
    supports_dense = True

    def __init__(self, cfg: PresetEncoderConfig):
        super().__init__(cfg.embed_dim)
        assert cfg.family == "precomputed"

        # cfg.in_dim <= 0 => infer at first forward (lazy head)
        self.in_dim = int(cfg.in_dim) if int(cfg.in_dim) > 0 else -1

        self.head = _make_head(
            in_features=self.in_dim,
            embed_dim=cfg.embed_dim,
            kind=cfg.projection_head,
            dropout=cfg.dropout,
            activation="silu",
        )

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


@dataclass(frozen=True)
class ECFPCachedConfig(MoleculeEncoderConfig):
    """Config for ECFPCachedEncoder: precomputed ECFP4 lookup by SMILES string."""

    fingerprint_path: str = ""
    key_column: str = "Metadata_SMILES"


@dataclass(frozen=True)
class GeneLookupConfig(MoleculeEncoderConfig):
    """Config for GeneLookupEncoder: precomputed gene embedding lookup by gene symbol."""

    fingerprint_path: str = ""
    embedding_column: str = ""
    # Optional CSV (columns: sirna_id, gene_symbol) that translates siRNA identifiers
    # (e.g. "s21721") to gene symbols before the embedding lookup.
    # When set, the lookup dict is keyed by siRNA ID rather than gene symbol,
    # so datasets that store siRNA IDs in their perturbation column work unchanged.
    sirna_to_gene_path: str = ""
    # Key for the control/DMSO/EMPTY condition. When set, the mean of all gene
    # embeddings is inserted under this key so control wells can be included in
    # training batches without a corresponding gene target.
    control_key: str = ""


@dataclass(frozen=True)
class RandomEncoderConfig(MoleculeEncoderConfig):
    """Config for RandomMoleculeEncoder. ``projection_head`` is unused (no head; direct random output)."""


class ECFPCachedEncoder(MoleculeEncoderBase):
    """Precomputed ECFP4 fingerprint lookup by perturbation key.

    Fingerprints are loaded once at construction from a parquet with columns
    ``[<key_column>, ecfp4_2048]`` (default key_column: ``Metadata_SMILES``).
    Each ``ecfp4_2048`` entry is a log1p(count) Morgan r=2 2048-bit vector (float32).
    The projection head maps it to ``embed_dim`` the same way as ``ECFPEncoder``.
    """

    supports_smiles = True
    supports_tokenized = False
    supports_dense = False

    _ECFP4_DIM: int = 2048

    def __init__(self, cfg: ECFPCachedConfig) -> None:
        super().__init__(cfg.embed_dim)
        if not cfg.fingerprint_path:
            raise ValueError(
                "ECFPCachedEncoder requires fingerprint_path. "
                "Set it in the model Hydra config (model/config/model/ecfp_cached.yaml)."
            )
        path = Path(cfg.fingerprint_path)
        if not path.exists():
            raise FileNotFoundError(f"ECFP4 fingerprint file not found: {path}")
        self._lookup: dict[str, torch.Tensor] = self._load_lookup(path, cfg.key_column)
        self.head = _make_head(
            in_features=self._ECFP4_DIM,
            embed_dim=cfg.embed_dim,
            kind=cfg.projection_head,
            activation="silu",
        )

    @classmethod
    def _load_lookup(cls, path: Path, key_column: str) -> dict[str, torch.Tensor]:
        import numpy as np
        import pandas as pd

        df = pd.read_parquet(path, columns=[key_column, "ecfp4_2048"])
        fps = np.array(df["ecfp4_2048"].tolist(), dtype=np.float32)  # (N, 2048)
        if fps.ndim != 2 or fps.shape[1] != cls._ECFP4_DIM:
            raise ValueError(
                f"ecfp4_2048 column has wrong shape {fps.shape}; expected (N, {cls._ECFP4_DIM}). Regenerate the parquet."
            )
        if not np.isfinite(fps).all():
            n_bad = int((~np.isfinite(fps)).any(axis=1).sum())
            raise ValueError(f"{n_bad} row(s) in {path} contain NaN/Inf. Regenerate the parquet.")
        return dict(zip(df[key_column].tolist(), torch.from_numpy(fps), strict=True))

    def encode_smiles(self, smiles: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        missing = [s for s in smiles if s not in self._lookup]
        if missing:
            raise KeyError(
                f"{len(missing)} SMILES not in fingerprint lookup "
                f"(first: {missing[0]!r}). "
                f"Lookup has {len(self._lookup)} compounds. "
                "Recompute ecfp4_fingerprints.parquet to include all training compounds."
            )
        fps = torch.stack([self._lookup[s] for s in smiles])
        return self.head(fps.to(device))


class GeneLookupEncoder(MoleculeEncoderBase):
    """Fixed gene embedding lookup + learnable projection head.

    Mirrors ECFPCachedEncoder but keyed on gene symbol (siRNA target, CRISPR gene)
    rather than SMILES. The lookup table is precomputed from a gene embedding
    model (ESM2); only the projection head is trained.
    """

    supports_smiles = False
    supports_strings = True
    supports_tokenized = False
    supports_dense = False

    def __init__(self, cfg: GeneLookupConfig) -> None:
        super().__init__(cfg.embed_dim)
        if not cfg.fingerprint_path:
            raise ValueError(
                "GeneLookupEncoder requires fingerprint_path. "
                "Set it in the model Hydra config (model/config/model/gene_*.yaml)."
            )
        if not cfg.embedding_column:
            raise ValueError(
                "GeneLookupEncoder requires embedding_column (e.g. 'esm2_1280'). Set it in the model Hydra config."
            )
        path = Path(cfg.fingerprint_path)
        if not path.exists():
            raise FileNotFoundError(f"Gene embedding file not found: {path}")
        sirna_map = self._load_sirna_map(cfg.sirna_to_gene_path) if cfg.sirna_to_gene_path else None
        self._embedding_dim, self._lookup = self._load_lookup(path, cfg.embedding_column, sirna_map, cfg.control_key)
        self.head = _make_head(
            in_features=self._embedding_dim,
            embed_dim=cfg.embed_dim,
            kind=cfg.projection_head,
            activation="silu",
        )

    @staticmethod
    def _load_sirna_map(csv_path: str) -> dict[str, str]:
        """Load siRNA-ID → gene-symbol mapping from a CSV with columns (sirna_id, gene_symbol)."""
        import pandas as pd

        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"siRNA→gene mapping file not found: {p}")
        df = pd.read_csv(p, usecols=["sirna_id", "gene_symbol"])
        return dict(zip(df["sirna_id"].tolist(), df["gene_symbol"].tolist(), strict=True))

    @staticmethod
    def _load_lookup(
        path: Path, column: str, sirna_map: dict[str, str] | None, control_key: str = ""
    ) -> tuple[int, dict[str, torch.Tensor]]:
        import numpy as np
        import pandas as pd

        df = pd.read_parquet(path, columns=["gene_symbol", column])
        dupes = df["gene_symbol"][df["gene_symbol"].duplicated()].tolist()
        if dupes:
            raise ValueError(
                f"{len(dupes)} duplicate gene_symbol(s) in {path}: {dupes[:5]}. "
                "Regenerate the parquet — duplicates cause silent lookup errors."
            )
        fps = np.array(df[column].tolist(), dtype=np.float32)
        if fps.ndim != 2:
            raise ValueError(f"{column!r} column has wrong shape {fps.shape}; expected (N, D). Regenerate the parquet.")
        if not np.isfinite(fps).all():
            n_bad = int((~np.isfinite(fps)).any(axis=1).sum())
            raise ValueError(f"{n_bad} row(s) in {path} contain NaN/Inf. Regenerate the parquet.")
        embed_dim = fps.shape[1]
        gene_lookup: dict[str, torch.Tensor] = dict(
            zip(df["gene_symbol"].tolist(), torch.from_numpy(fps), strict=True)
        )
        if sirna_map is None:
            lookup = gene_lookup
        else:
            # Re-key by siRNA ID: only include entries where the mapped gene has an embedding.
            lookup = {
                sirna: gene_lookup[gene]
                for sirna, gene in sirna_map.items()
                if gene in gene_lookup
            }
        if control_key:
            # Mean of all gene embeddings (computed from full matrix, not siRNA subset).
            # Only for control/DMSO/EMPTY wells that have no gene target.
            lookup[control_key] = torch.from_numpy(fps.mean(axis=0))
        return embed_dim, lookup

    def encode_strings(self, genes: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        missing = [g for g in genes if g not in self._lookup]
        if missing:
            raise KeyError(
                f"{len(missing)} gene symbol(s) not in embedding lookup "
                f"(first: {missing[0]!r}). "
                f"Lookup has {len(self._lookup)} genes. "
                "Recompute the gene embedding parquet to include all training genes."
            )
        embs = torch.stack([self._lookup[g] for g in genes])
        return self.head(embs.to(device))


class RandomMoleculeEncoder(MoleculeEncoderBase):
    """Returns random unit-norm embeddings — smoke-testing only, no perturbation alignment learned.

    Overrides ``forward`` directly; ``encode_strings`` / ``encode_smiles`` / ``encode_dense`` are not implemented.
    """

    supports_smiles = False
    supports_dense = False

    def forward(self, x: object) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            B, device = x.shape[0], x.device
        elif isinstance(x, (list, tuple)):
            B, device = len(x), torch.device("cpu")
        else:
            B, device = 1, torch.device("cpu")
        out = torch.randn(B, self.embed_dim, device=device)
        return torch.nn.functional.normalize(out, dim=-1)


# =========================================================
# Factory
# =========================================================


def build_encoder_from_config(cfg: MoleculeEncoderConfig) -> MoleculeEncoderBase:
    match cfg:
        case ChembertaConfig():
            return ChembertaEncoder(cfg)
        case GeneLookupConfig():
            return GeneLookupEncoder(cfg)
        case ECFPCachedConfig():
            return ECFPCachedEncoder(cfg)
        case PresetEncoderConfig(family="ecfp"):
            return ECFPEncoder(cfg)
        case PresetEncoderConfig(family="whimf"):
            return WHIMFEncoder(cfg)
        case PresetEncoderConfig(family="precomputed"):
            return PrecomputedDenseEncoder(cfg)
        case RandomEncoderConfig():
            return RandomMoleculeEncoder(cfg.embed_dim)
        case _:
            raise TypeError(f"Unsupported MoleculeEncoderConfig: {type(cfg).__name__}")


# =========================================================
# Enum-backed registry (lazy)
# =========================================================


def _filter_overrides(cfg: MoleculeEncoderConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in dc_fields(cfg)}
    return {k: v for k, v in overrides.items() if k in valid}


class MoleculeEncoderRegistry(Enum):
    """
    Enum registry with lazy configs. Use like a dict:
        MoleculeEncoderRegistry['chemberta'].build(embed_dim=256)
        MoleculeEncoderRegistry.get_config('ecfp', embed_dim=256)
    Heavy HF models are created only when `.build()` is called.
    """

    chemberta = (
        lambda: ChembertaConfig(
            name="chemberta",
            embed_dim=128,
            projection_head="mlp",
            hf_model="DeepChem/ChemBERTa-77M-MLM",
            pool="cls",
            freeze_backbone=True,
            tokenizer_max_length=None,  # use tokenizer default (often 512)
            lora=LoraConfig(enabled=False),
            freeze_backbone_when_no_lora=True,
        ),
    )

    ecfp = (
        lambda: PresetEncoderConfig(
            name="ecfp",
            embed_dim=128,
            projection_head="mlp",
            family="ecfp",
            in_dim=2048,  # ignored internally; kept for clarity
            dropout=0.0,
        ),
    )

    whimf = (
        lambda: PresetEncoderConfig(
            name="whimf",
            embed_dim=128,
            projection_head="mlp",
            family="whimf",
            in_dim=114,  # ignored internally; kept for clarity
            dropout=0.0,
        ),
    )

    precomputed = (
        lambda: PresetEncoderConfig(
            name="precomputed",
            embed_dim=128,
            projection_head="mlp",
            family="precomputed",
            in_dim=512,  # adjust to your dataset
            dropout=0.0,
        ),
    )

    ecfp_cached = (
        lambda: ECFPCachedConfig(
            name="ecfp_cached",
            embed_dim=128,
            projection_head="mlp",
            fingerprint_path="",  # must be overridden via fingerprint_path at build time
        ),
    )

    gene_lookup = (
        lambda: GeneLookupConfig(
            name="gene_lookup",
            embed_dim=128,
            projection_head="mlp",
            fingerprint_path="",
            embedding_column="",
        ),
    )

    random = (
        lambda: RandomEncoderConfig(
            name="random",
            embed_dim=128,
            projection_head="linear",
        ),
    )

    # --- member helpers ---

    @property
    def factory(self) -> Callable[[], MoleculeEncoderConfig]:
        return self.value[0]

    def make_config(self, **overrides: Any) -> MoleculeEncoderConfig:
        base = self.factory()
        ov = _filter_overrides(base, overrides)
        return replace(base, **ov)

    def build(self, **overrides: Any) -> MoleculeEncoderBase:
        cfg = self.make_config(**overrides)
        return build_encoder_from_config(cfg)

    # --- class helpers ---

    @classmethod
    def list_names(cls) -> list[str]:
        return [m.name for m in cls]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.__members__

    @classmethod
    def get_config(cls, name: str, **overrides: Any) -> MoleculeEncoderConfig:
        try:
            return cls[name].make_config(**overrides)
        except KeyError as e:
            raise KeyError(f"Unknown molecule encoder '{name}'. Known: {cls.list_names()}") from e

    @classmethod
    def build_from_name(cls, name: str, **overrides: Any) -> MoleculeEncoderBase:
        try:
            return cls[name].build(**overrides)
        except KeyError as e:
            raise KeyError(f"Unknown molecule encoder '{name}'. Known: {cls.list_names()}") from e


__all__ = [
    "MoleculeEncoderConfig",
    "ChembertaConfig",
    "ECFPCachedConfig",
    "GeneLookupConfig",
    "PresetEncoderConfig",
    "RandomEncoderConfig",
    "MoleculeEncoderBase",
    "ChembertaEncoder",
    "ECFPCachedEncoder",
    "GeneLookupEncoder",
    "ECFPEncoder",
    "WHIMFEncoder",
    "PrecomputedDenseEncoder",
    "RandomMoleculeEncoder",
    "build_encoder_from_config",
    "MoleculeEncoderRegistry",
]
