from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from dataclasses import fields as dc_fields
from enum import Enum
from typing import Any, Literal

import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from timm.data import resolve_model_data_config
from timm.layers import to_2tuple
from transformers import AutoModel

from cp_bg_bench_model._heads import ProjectionHead
from cp_bg_bench_model._logging import logger as log
from cp_bg_bench_model.lora import LoraConfig, apply_lora_if_enabled, collect_lora_params

# =========================================================
# Utility modules (adapter & normalizer)
# =========================================================


class ChannelAdapter(nn.Module):
    """1x1 Conv to go N→3 for ImageNet backbones; stays trainable even if backbone is frozen."""

    def __init__(self, in_ch: int, out_ch: int = 3, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        nn.init.kaiming_uniform_(self.conv.weight, a=1.0)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TimmNormalizer(nn.Module):
    """Per-channel mean/std normalizer for timm backbones."""

    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class LearnedAffineNormalizer(nn.Module):
    """
    Learnable per-channel affine transform.
    Initializes to (x - mean) / std if init_mean/std are provided.
    """

    def __init__(self, num_channels: int, init_mean=None, init_std=None):
        super().__init__()
        scale = torch.ones(1, num_channels, 1, 1, dtype=torch.float32)
        bias = torch.zeros(1, num_channels, 1, 1, dtype=torch.float32)

        if init_mean is not None and init_std is not None:
            m = torch.tensor(init_mean, dtype=torch.float32).view(1, num_channels, 1, 1)
            s = torch.tensor(init_std, dtype=torch.float32).view(1, num_channels, 1, 1)
            scale = 1.0 / s
            bias = -m / s

        self.scale = nn.Parameter(scale)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


# =========================================================
# Config dataclasses (internal)
# =========================================================


@dataclass(frozen=True)
class ImageEncoderConfig:
    """Base config shared by all image encoders (internal use)."""

    name: str
    embed_dim: int = 128
    projection_head: Literal["mlp", "linear"] = "mlp"
    image_size: int = 224
    in_channels: int = 5
    lora: LoraConfig | None = None
    freeze_backbone_when_no_lora: bool = True
    native_channels: bool = False  # True = encoder accepts in_channels directly; skip 5→3 adapter


@dataclass(frozen=True)
class DinoV3ViTConfig(ImageEncoderConfig):
    """
    Config for DINOv3 ViT via timm.
    We keep a 5 -> 3 adapter so pretrained weights remain compatible.
    """

    timm_name: str = "hf-hub:timm/vit_small_patch16_dinov3.lvd1689m"
    pretrained: bool = True
    pool: Literal["cls", "cls_mean", "mean"] = "cls"
    normalizer: Literal["learned_imagenet", "fixed_imagenet", "none"] = "learned_imagenet"
    image_size: int = 256
    in_channels: int = 5


@dataclass(frozen=True)
class OpenPhenomConfig(ImageEncoderConfig):
    """Config for OpenPhenom (HF). Expects 5 x 256 x 256 uint8; datamodule should ensure this."""

    hf_model: str = "recursionpharma/OpenPhenom"
    channelwise: bool = False  # if True: concat per-channel 384-dim feat (C*384)
    image_size: int = 256  # OpenPhenom wants 256; datamodule should match.
    native_channels: bool = True  # OpenPhenom accepts in_channels directly; no adapter needed


@dataclass(frozen=True)
class SubCellConfig(ImageEncoderConfig):
    """
    Config for SubCell (Lundberg lab ViT-B, HPA-trained, 1536-dim embeddings).
    Two channel-adaptation modes are supported:

    rotation (default, matches the published JUMP1 benchmark):
        Uses the ER-DNA-Protein (ybg, 3-channel) SubCell variant.
        Two CP channels serve as ER/DNA reference slots; remaining channels
        are rotated through the Protein slot in separate forward passes.
        Embeddings from each pass are concatenated → out_features = N * 1536.

    projection (ablation):
        Uses the all-channels (rybg, 4-channel) SubCell variant.
        Three CP channels map directly to MT/ER/DNA slots; remaining channels
        are collapsed into the Protein slot via a learned 1x1 conv.
        Single forward pass → out_features = 1536.

    Channel-slot assignment: set channel_names (recommended) or explicit index
    tuples.  When channel_names is provided the encoder derives the correct
    indices at init time; explicit index fields are ignored.  If neither is
    provided, SubCellEncoder raises ValueError — there is no silent default.
    """

    name: str = "subcell"
    image_size: int = 256
    vit_image_size: int = 224  # resize to this before feeding the ViT trunk
    in_channels: int = 5
    native_channels: bool = True  # channel mapping handled internally; skip generic adapter

    variant: Literal["er_dna_protein", "all_channels"] = "er_dna_protein"
    checkpoint_name: Literal["mae_cells_prots_pool", "vit_prots_pool"] = "mae_cells_prots_pool"
    checkpoint_url: str | None = None  # explicit URL override; None = derive from variant/checkpoint_name
    local_cache_dir: str = "~/.cache/cp_bg_bench/subcell"

    channel_mode: Literal["rotation", "projection"] = "projection"

    # --- name-based resolution (preferred) ---
    # Ordered list of channel names in the source tensor, e.g. ["DNA","AGP","ER","Mito","RNA"].
    # Required names: "ER", "DNA"; "Mito" is additionally required for projection mode.
    channel_names: tuple[str, ...] | None = None
    # Optional override for which non-core channels to route/bundle, in that order.
    # If empty, all non-core channels are used in source order.
    protein_bundle_priority: tuple[str, ...] = ()

    # --- explicit index fallback (used when channel_names is None) ---
    # rotation: (er_index, dna_index) in source tensor
    reference_channel_indices: tuple[int, int] | None = None
    # rotation: each index rotated through the Protein slot in a separate pass
    extra_channel_indices: tuple[int, ...] | None = None
    # projection: 3 passthrough indices → MT, ER, DNA HPA slots (in that order)
    passthrough_channel_indices: tuple[int, int, int] | None = None
    # projection: remaining indices collapsed into Protein slot via learned 1x1 Conv
    bundle_channel_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TimmCNNConfig(ImageEncoderConfig):
    """
    Config for timm CNN backbones with global pooling (e.g., densenet121).
    Inputs are float in [0,1]; we do 5→3 via a trainable 1×1 conv and then normalize.
    """

    timm_name: str = "densenet121"
    pretrained: bool = True
    image_size: int = 224
    in_channels: int = 5


# =========================================================
# Base encoder with shared LoRA + head + grouping
# =========================================================


def _make_head(
    in_features: int,
    embed_dim: int,
    kind: Literal["mlp", "linear"],
    *,
    dropout: float = 0.0,
    activation: Literal["geglu", "silu"] = "geglu",
) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_features, embed_dim)
    return ProjectionHead(
        in_features=in_features,
        embed_dim=embed_dim,
        width_alpha=1.2,
        round_to=64,
        min_hidden=256,
        max_hidden=4096,
        dropout=dropout,
        use_skip=True,
        activation=activation,
    )


class ImageEncoderBase(nn.Module):
    """
    Unified interface for image encoders.
      forward(x: Tensor[B,C,H,W]) -> Tensor[B, embed_dim]
    Provides:
      • projection head
      • optimizer param grouping: {"head": [...], "lora": [...]}
      (LoRA wrapping is performed by subclasses via apply_lora_if_enabled.)
    """

    def __init__(self, cfg: ImageEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = int(cfg.embed_dim)
        self.in_channels = int(cfg.in_channels)
        self.image_size = to_2tuple(int(cfg.image_size))
        self.using_lora = False
        self.adapter: nn.Module | None = None
        self.normalizer: nn.Module | None = None
        # subclasses must set: self.trunk and self.out_features

        # Runtime diagnostics
        self._last_runtime_in_dim: int | None = None
        self._last_runtime_channels: int | None = None

    # ----- optimizer groups (head + LoRA + adapter) -----
    def trainable_param_groups(self) -> dict[str, list[nn.Parameter]]:
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        if self.adapter is not None:
            head_params += [p for p in self.adapter.parameters() if p.requires_grad]

        if self.normalizer is not None:
            head_params += [p for p in self.normalizer.parameters() if p.requires_grad]

        lora_params = collect_lora_params(self.trunk)
        return {"head": head_params, "lora": lora_params}

    # ----- logging helpers -----
    @staticmethod
    def _count_params(m: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in m.parameters())
        train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, train

    def log_trainable_summary(self, logger) -> None:
        trunk_total, trunk_train = self._count_params(self.trunk)
        head_total, head_train = self._count_params(self.head)

        # head io (Linear/ProjectionHead)
        def _head_io(h: nn.Module):
            kind = "linear" if isinstance(h, nn.Linear) else "mlp"
            in_d = getattr(h, "in_features", None)
            out_d = getattr(h, "embed_dim", None) or getattr(h, "out_features", None)
            if out_d is None and hasattr(h, "proj"):
                out_d = getattr(h.proj, "out_features", None)
                in_d = in_d or getattr(h.proj, "in_features", None)
            return kind, in_d, out_d

        head_kind, in_d, out_d = _head_io(self.head)

        parts = [
            f"ImageEncoder[{type(self).__name__}] "
            f"| trunk: total={trunk_total / 1e6:.1f}M, train={trunk_train / 1e6:.3f}M "
            f"| head train={head_train / 1e6:.3f}M "
        ]
        if in_d is not None and out_d is not None:
            parts.append(f"| proj: {in_d}→{out_d} ({head_kind}) ")
        if getattr(self, "adapter", None) is not None:
            adapter_train = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
            parts.append(f"| adapter train={adapter_train / 1e6:.3f}M ")
        lora_train = sum(p.numel() for p in collect_lora_params(self.trunk))
        parts.append(f"| lora train={lora_train / 1e6:.3f}M | using_lora={self.using_lora}")
        if isinstance(self.cfg, OpenPhenomConfig):
            parts.append(f" | channelwise={self.cfg.channelwise}")
        if self._last_runtime_channels is not None:
            parts.append(f" | runtime_channels={self._last_runtime_channels}")
        logger.info("".join(parts))

    def _reset_head(self, in_features: int, *, channelwise: bool) -> None:
        """Rebuild head (and implicit LayerNorm inside ProjectionHead) for a new input size."""
        drop = 0.3 if channelwise else 0.1
        new_head = _make_head(
            in_features=in_features,
            embed_dim=self.embed_dim,
            kind=self.cfg.projection_head,
            dropout=drop,
            activation="geglu",
        )
        # Preserve device/dtype
        new_head.to(next(self.parameters()).device)
        self.head = new_head
        self.out_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# =========================================================
# Concrete encoders
# =========================================================


class DinoV3ViTEncoder(ImageEncoderBase):
    """DINOv3 ViT (timm) with 5→3 adapter, (optionally) learnable affine normalizer, and LoRA."""

    def __init__(self, cfg: DinoV3ViTConfig):
        super().__init__(cfg)

        # timm trunk (pretrained expects 3-channel input)
        self.trunk = timm.create_model(
            cfg.timm_name,
            pretrained=bool(cfg.pretrained),
            num_classes=0,
            in_chans=3,
        )

        # 5→3 adapter (trainable even if trunk frozen); skipped when native_channels=True
        if not cfg.native_channels:
            self.adapter = ChannelAdapter(in_ch=self.in_channels, out_ch=3, bias=False)

        # Normalizer initialized from timm checkpoint meta (but learnable by default)
        data_cfg = resolve_model_data_config(self.trunk)
        mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
        std = data_cfg.get("std", (0.229, 0.224, 0.225))

        if cfg.normalizer == "learned_imagenet":
            self.normalizer = LearnedAffineNormalizer(num_channels=3, init_mean=mean, init_std=std)
        elif cfg.normalizer == "fixed_imagenet":
            self.normalizer = TimmNormalizer(mean, std)
        else:
            self.normalizer = nn.Identity()

        # LoRA on ViT blocks (attention + optionally MLP)
        self.trunk, self.using_lora = apply_lora_if_enabled(
            self.trunk,
            cfg.lora,
            default_targets=self._default_lora_targets(),
            freeze_if_disabled=cfg.freeze_backbone_when_no_lora,
            freeze_bn=True,
        )

        base = self._get_base(self.trunk)
        self.base_dim = int(
            getattr(base, "num_features", 0)
            or getattr(base, "embed_dim", 0)
            or getattr(self.trunk, "num_features", 0)
            or getattr(self.trunk, "embed_dim", 0)
        )
        if self.base_dim <= 0:
            raise RuntimeError("Could not infer DINOv3 embedding dim from timm model.")

        self.num_prefix_tokens = int(getattr(base, "num_prefix_tokens", 1))
        if self.num_prefix_tokens < 1:
            self.num_prefix_tokens = 1

        self.out_features = self.base_dim * (2 if cfg.pool == "cls_mean" else 1)

        self.head = _make_head(
            in_features=self.out_features,
            embed_dim=self.embed_dim,
            kind=self.cfg.projection_head,
            dropout=0.1,
            activation="geglu",
        )

    @staticmethod
    def _get_base(m: nn.Module) -> nn.Module:
        return getattr(m, "model", getattr(m, "base_model", m))

    def _default_lora_targets(self) -> list[str]:
        # Good first-pass for timm ViT attention; extend later with fc1/fc2 if you want.
        return ["qkv", "proj", "fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]

    @staticmethod
    def _ensure_float01(x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        pool = self.cfg.pool

        # "cls" uses the trunk's default headless output (usually CLS)
        if pool == "cls":
            return self.trunk(x)

        base = self._get_base(self.trunk)
        if not hasattr(base, "forward_features"):
            return self.trunk(x)

        tokens = base.forward_features(x)
        if isinstance(tokens, (tuple, list)) and len(tokens) > 0:
            tokens = tokens[0]
        if not torch.is_tensor(tokens) or tokens.ndim != 3:
            return self.trunk(x)

        cls = tokens[:, 0, :]
        patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        patch_mean = patch_tokens.mean(dim=1) if patch_tokens.numel() > 0 else cls

        if pool == "mean":
            return patch_mean
        return torch.cat([cls, patch_mean], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (B,C,H,W), got {tuple(x.shape)}"
        x = self._ensure_float01(x)
        if self.adapter is not None:
            x = self.adapter(x)
        x = self.normalizer(x)
        feats = self._pool(x)
        return self.head(feats)


class OpenPhenomEncoder(ImageEncoderBase):
    """OpenPhenom via HF (expects [B,5,256,256] uint8; datamodule should ensure this)."""

    def __init__(self, cfg: OpenPhenomConfig):
        super().__init__(cfg)

        # 1) Load trunk
        self.trunk = AutoModel.from_pretrained(cfg.hf_model, trust_remote_code=True)

        # 2) Apply LoRA with safe, vision-appropriate wrapper
        lora_cfg = cfg.lora
        if isinstance(lora_cfg, LoraConfig) and lora_cfg.enabled:
            # Make sure task type is safe for vision encoders
            if getattr(lora_cfg, "task_type", None) in (None, ""):
                lora_cfg = replace(lora_cfg, task_type="FEATURE_EXTRACTION")

        self.trunk, self.using_lora = apply_lora_if_enabled(
            self.trunk,
            lora_cfg,
            default_targets=self._default_lora_targets(),
            freeze_if_disabled=cfg.freeze_backbone_when_no_lora,
            freeze_bn=True,
        )

        base = self._get_base(self.trunk)
        try:
            setattr(base, "return_channelwise_embeddings", bool(cfg.channelwise))
        except Exception:
            pass

        # 4) Static expectations
        self.base_dim = 384
        self.out_features = self.base_dim * (self.in_channels if cfg.channelwise else 1)

        # 5) Projection head (may be rebuilt lazily on first forward)
        drop = 0.3 if cfg.channelwise else 0.1
        self.head = _make_head(
            in_features=self.out_features,
            embed_dim=self.embed_dim,
            kind=self.cfg.projection_head,
            dropout=drop,
            activation="geglu",
        )

    def _default_lora_targets(self) -> list[str]:
        # Robust defaults to catch common attention proj names across HF ViTs
        return ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "attn_out_proj", "qkv", "proj"]

    @staticmethod
    def _unwrap(o: Any) -> torch.Tensor:
        # MAE implementations vary; normalize to a Tensor of embeddings.
        if hasattr(o, "embeddings"):
            return o.embeddings
        if isinstance(o, dict) and "embeddings" in o:
            return o["embeddings"]
        return o  # already a Tensor

    def _get_base(self, m):
        return getattr(m, "model", getattr(m, "base_model", m))

    def _forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        base = self._get_base(self.trunk)
        # make sure channelwise is enabled on the base model
        try:
            setattr(base, "return_channelwise_embeddings", bool(self.cfg.channelwise))
        except Exception:
            pass

        if hasattr(base, "encode_to_embeddings"):
            # this keeps grad when training LoRA, no inference_mode
            with torch.set_grad_enabled(self.training and self.using_lora):
                return base.encode_to_embeddings(x)  # returns [B, 384] or [B, C, 384]/[B, C*384]
        elif hasattr(base, "predict"):
            # predict often uses no_grad internally → fine for frozen trunk, NOT for LoRA
            with torch.no_grad():
                return base.predict(x)
        else:
            # LAST RESORT: this is the reconstruction path (will expect C=6!)
            return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (B,C,H,W), got {tuple(x.shape)}"
        if x.dtype != torch.uint8:
            if x.dtype != torch.float32:
                x = x.float()
            if x.max() <= 1.5:
                x = (x * 255.0).round()
            x = x.clamp(0, 255).to(torch.uint8)

        feats = self._forward_trunk(x)  # [B, 384] or [B, C, 384] or [B, C*384]

        if isinstance(self.cfg, OpenPhenomConfig) and self.cfg.channelwise:
            # Normalize to [B, C*384]
            if feats.ndim == 3:
                runtime_c = int(feats.size(1))
                feats = feats.reshape(feats.size(0), -1)
            elif feats.ndim == 2:
                in_dim_now = int(feats.size(1))
                runtime_c = max(1, in_dim_now // self.base_dim)
            else:
                raise RuntimeError(f"Unexpected OpenPhenom output shape {tuple(feats.shape)}")

            in_dim = int(feats.size(1))
            self._last_runtime_in_dim = in_dim
            self._last_runtime_channels = runtime_c

            if in_dim != self.out_features:
                log.info(
                    f"OpenPhenomEncoder: detected runtime channelwise dim={in_dim} "
                    f"(C={runtime_c}) != head.in_features={self.out_features} → rebuilding head."
                )
                self._reset_head(in_dim, channelwise=True)
        else:
            # Non-channelwise: ensure [B, 384]
            if feats.ndim == 3:
                feats = feats.mean(dim=1)
            elif feats.ndim != 2:
                raise RuntimeError(f"Unexpected OpenPhenom output shape {tuple(feats.shape)}")

        return self.head(feats)


class TimmCNNEncoder(ImageEncoderBase):
    """timm backbone (e.g., DenseNet121) with 5→3 adapter and normalization."""

    def __init__(self, cnn_config: TimmCNNConfig):
        super().__init__(cnn_config)

        # build timm trunk (3-channel ImageNet weights)
        self.trunk = timm.create_model(
            cnn_config.timm_name,
            pretrained=bool(cnn_config.pretrained),
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )
        # output dim from timm backbone
        self.out_features = int(getattr(self.trunk, "num_features", 0)) or int(getattr(self.trunk, "num_features"))

        # 5→3 adapter (trainable even when backbone is frozen); skipped when native_channels=True
        if not cnn_config.native_channels:
            self.adapter = ChannelAdapter(in_ch=self.in_channels, out_ch=3, bias=False)

        # timm normalization from checkpoint meta
        data_cfg = resolve_model_data_config(self.trunk)
        mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
        std = data_cfg.get("std", (0.229, 0.224, 0.225))
        self.normalizer = TimmNormalizer(mean, std)

        # LoRA with per-encoder defaults (overridden if cfg.lora.targets is set)
        self.trunk, self.using_lora = apply_lora_if_enabled(
            self.trunk,
            cnn_config.lora,
            default_targets=self._default_lora_targets(),
            freeze_if_disabled=cnn_config.freeze_backbone_when_no_lora,
            freeze_bn=True,
        )

        # projection head
        self.head = _make_head(
            in_features=self.out_features,
            embed_dim=self.embed_dim,
            kind=self.cfg.projection_head,
            dropout=0.1,
            activation="geglu",
        )

    def _default_lora_targets(self) -> list[str]:
        # Generic target for conv modules; refine per-arch if you experiment with LoRA on CNNs
        return ["conv"]

    @staticmethod
    def _ensure_float01(x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (B,C,H,W), got {tuple(x.shape)}"
        x = self._ensure_float01(x)
        if self.adapter is not None:
            x = self.adapter(x)
        x = self.normalizer(x)
        feats = self.trunk(x)  # [B, D] pooled
        return self.head(feats)


def _indices_from_names(
    channel_names: tuple[str, ...],
    channel_mode: str,
    protein_bundle_priority: tuple[str, ...],
) -> tuple[
    tuple[int, int],
    tuple[int, ...],
    tuple[int, int, int] | None,
    tuple[int, ...] | None,
]:
    """Derive SubCell slot indices from ordered channel names.

    Args:
        channel_names: Ordered channel names for the dataset (e.g. ``("DNA", "AGP", "ER", "Mito", "RNA")``).
            Must include ``"ER"`` and ``"DNA"``. Projection mode also requires ``"Mito"``.
        channel_mode: ``"rotation"`` or ``"projection"``.
        protein_bundle_priority: Optional ordered subset of non-core channel names to use as
            the protein-slot extras (rotation) or bundle (projection). When empty, all non-core
            channels are used in source order. In projection mode this filters the bundle;
            in rotation mode it filters the extras passed through the protein slot.

    Returns:
        ``(reference, extras, passthrough, bundle)`` where ``passthrough`` and ``bundle``
        are ``None`` for rotation mode.
    """
    names = list(channel_names)
    idx: dict[str, int] = {n: i for i, n in enumerate(names)}

    for required in ("ER", "DNA"):
        if required not in idx:
            raise ValueError(f"channel_names must include '{required}'; got {names}")

    core = {"ER", "DNA"}
    reference: tuple[int, int] = (idx["ER"], idx["DNA"])
    non_core_in_order = [i for i, n in enumerate(names) if n not in core]

    if protein_bundle_priority:
        missing = [n for n in protein_bundle_priority if n not in idx]
        if missing:
            raise ValueError(f"protein_bundle_priority names not in channel_names: {missing}")
        extras: tuple[int, ...] = tuple(idx[n] for n in protein_bundle_priority)
    else:
        extras = tuple(non_core_in_order)

    if channel_mode == "projection":
        if "Mito" not in idx:
            raise ValueError(f"projection mode requires a 'Mito' channel; got {names}")
        passthrough: tuple[int, int, int] | None = (idx["Mito"], idx["ER"], idx["DNA"])
        mt_er_dna = {"Mito", "ER", "DNA"}
        if protein_bundle_priority:
            bundle: tuple[int, ...] | None = tuple(idx[n] for n in protein_bundle_priority)
        else:
            bundle = tuple(i for i, n in enumerate(names) if n not in mt_er_dna)
    else:
        passthrough = None
        bundle = None

    return reference, extras, passthrough, bundle


class SubCellEncoder(ImageEncoderBase):
    """
    SubCell ViT-B image encoder (Lundberg lab, HPA-trained).
    Supports two channel-adaptation modes; see SubCellConfig for details.
    """

    def __init__(self, cfg: SubCellConfig):
        super().__init__(cfg)

        # Resolve channel indices from names (preferred) or explicit fields.
        if cfg.channel_names is not None:
            ref_idx, extra_idx, passthrough_idx, bundle_idx = _indices_from_names(
                cfg.channel_names, cfg.channel_mode, cfg.protein_bundle_priority
            )
        elif cfg.channel_mode == "rotation":
            if cfg.reference_channel_indices is None or cfg.extra_channel_indices is None:
                raise ValueError(
                    "rotation mode requires channel_names or explicit "
                    "reference_channel_indices + extra_channel_indices. "
                    "Set channel_names in the datamodule config."
                )
            ref_idx = cfg.reference_channel_indices
            extra_idx = cfg.extra_channel_indices
            passthrough_idx = None
            bundle_idx = None
        else:  # projection
            if cfg.passthrough_channel_indices is None or cfg.bundle_channel_indices is None:
                raise ValueError(
                    "projection mode requires channel_names or explicit "
                    "passthrough_channel_indices + bundle_channel_indices. "
                    "Set channel_names in the datamodule config."
                )
            ref_idx = (0, 0)  # unused in projection
            extra_idx = ()
            passthrough_idx = cfg.passthrough_channel_indices
            bundle_idx = cfg.bundle_channel_indices

        self._ref_idx: tuple[int, int] = ref_idx
        self._extra_idx: tuple[int, ...] = extra_idx
        self._passthrough_idx: tuple[int, int, int] | None = passthrough_idx
        self._bundle_idx: tuple[int, ...] | None = bundle_idx

        from cp_bg_bench_model.encoders._subcell_loader import load_subcell_trunk

        self.trunk, base_dim = load_subcell_trunk(
            variant=cfg.variant,
            checkpoint_name=cfg.checkpoint_name,
            cache_dir=cfg.local_cache_dir,
            checkpoint_url=cfg.checkpoint_url,
        )

        self.trunk, self.using_lora = apply_lora_if_enabled(
            self.trunk,
            cfg.lora,
            default_targets=self._default_lora_targets(),
            freeze_if_disabled=cfg.freeze_backbone_when_no_lora,
            freeze_bn=True,
        )

        n_extras = len(self._extra_idx)
        self.out_features = base_dim * n_extras if cfg.channel_mode == "rotation" else base_dim

        drop = 0.1
        self.head = _make_head(
            in_features=self.out_features,
            embed_dim=self.embed_dim,
            kind=cfg.projection_head,
            dropout=drop,
            activation="geglu",
        )

        self.vit_resize = T.Resize((cfg.vit_image_size, cfg.vit_image_size), antialias=True)

        if cfg.channel_mode == "projection":
            if self._bundle_idx is None:
                raise ValueError("projection mode requires bundle_channel_indices or channel_names")
            n_bundle = len(self._bundle_idx)
            self.channel_bundle = nn.Conv2d(n_bundle, 1, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(self.channel_bundle.weight, a=1.0)
        else:
            self.channel_bundle = None

    def _default_lora_targets(self) -> list[str]:
        return ["qkv", "proj", "fc1", "fc2", "query", "key", "value", "dense"]

    def trainable_param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = super().trainable_param_groups()
        if self.channel_bundle is not None:
            groups["head"] += [p for p in self.channel_bundle.parameters() if p.requires_grad]
        return groups

    @staticmethod
    def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
        # Global min-max (one scalar per image) matching SubCell dataset preprocessing
        b = x.size(0)
        flat = x.view(b, -1)
        mn = flat.min(dim=1).values.view(b, 1, 1, 1)
        mx = flat.max(dim=1).values.view(b, 1, 1, 1)
        return (x - mn) / (mx - mn + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (B,C,H,W), got {tuple(x.shape)}"
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = self._minmax_norm(x)

        cfg: SubCellConfig = self.cfg  # type: ignore[assignment]

        if cfg.channel_mode == "rotation":
            er_i, dna_i = self._ref_idx
            ref_er = x[:, er_i : er_i + 1]
            ref_dna = x[:, dna_i : dna_i + 1]
            parts = []
            for e in self._extra_idx:
                inp = torch.cat([ref_er, ref_dna, x[:, e : e + 1]], dim=1)
                inp = self.vit_resize(inp)
                out = self.trunk(inp)
                parts.append(out.pool_op)
            feats = torch.cat(parts, dim=1)

        else:  # projection
            assert self._passthrough_idx is not None
            assert self._bundle_idx is not None
            p0, p1, p2 = self._passthrough_idx
            bundle_idx = list(self._bundle_idx)
            mt_ch = x[:, p0 : p0 + 1]
            er_ch = x[:, p1 : p1 + 1]
            dna_ch = x[:, p2 : p2 + 1]
            protein_ch = self.channel_bundle(x[:, bundle_idx])
            inp = torch.cat([mt_ch, er_ch, dna_ch, protein_ch], dim=1)
            inp = self.vit_resize(inp)
            feats = self.trunk(inp).pool_op

        return self.head(feats)


# =========================================================
# Factory
# =========================================================


def build_encoder_from_config(cfg: ImageEncoderConfig) -> ImageEncoderBase:
    match cfg:
        case OpenPhenomConfig():
            return OpenPhenomEncoder(cfg)
        case TimmCNNConfig():
            return TimmCNNEncoder(cfg)
        case DinoV3ViTConfig():
            return DinoV3ViTEncoder(cfg)
        case SubCellConfig():
            return SubCellEncoder(cfg)
        case _:
            raise TypeError(f"Unsupported ImageEncoderConfig: {type(cfg).__name__}")


# =========================================================
# Enum-backed registry (lazy)
# =========================================================


def _filter_overrides(cfg: ImageEncoderConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in dc_fields(cfg)}
    return {k: v for k, v in overrides.items() if k in valid}


class ImageEncoderRegistry(Enum):
    """
    Enum registry with lazy configs. Use like a dict:
        ImageEncoderRegistry["openphenom"].build(embed_dim=256, lora=...)
        ImageEncoderRegistry.get_config("densenet", in_channels=5)
    Models are created only when `.build()` is called.
    """

    openphenom = (
        lambda: OpenPhenomConfig(
            name="openphenom",
            embed_dim=128,
            projection_head="mlp",
            image_size=256,
            in_channels=5,
            channelwise=False,
            lora=LoraConfig(enabled=False),
            freeze_backbone_when_no_lora=True,
        ),
    )

    densenet = (
        lambda: TimmCNNConfig(
            name="densenet",
            embed_dim=128,
            projection_head="mlp",
            timm_name="densenet121",
            image_size=224,
            in_channels=5,
            pretrained=True,
            lora=LoraConfig(enabled=False),
            freeze_backbone_when_no_lora=True,
        ),
    )

    dinov3 = (
        lambda: DinoV3ViTConfig(
            name="dinov3",
            embed_dim=128,
            projection_head="mlp",
            # timm_name="hf-hub:timm/vit_small_patch16_dinov3.lvd1689m",
            timm_name="hf-hub:timm/vit_base_patch16_dinov3.lvd1689m",
            pretrained=True,
            pool="cls",
            normalizer="learned_imagenet",
            image_size=224,
            in_channels=5,
            lora=LoraConfig(enabled=False),
            freeze_backbone_when_no_lora=True,
        ),
    )

    subcell = (
        lambda: SubCellConfig(
            name="subcell",
            embed_dim=128,
            projection_head="mlp",
            image_size=256,
            in_channels=5,
            native_channels=True,
            variant="all_channels",
            checkpoint_name="mae_cells_prots_pool",
            channel_mode="projection",
            reference_channel_indices=None,
            extra_channel_indices=None,
            passthrough_channel_indices=None,
            bundle_channel_indices=None,
            lora=LoraConfig(enabled=False),
            freeze_backbone_when_no_lora=True,
        ),
    )

    @property
    def factory(self) -> Callable[[], ImageEncoderConfig]:
        return self.value[0]

    def make_config(self, **overrides: Any) -> ImageEncoderConfig:
        base = self.factory()
        ov = _filter_overrides(base, overrides)
        return replace(base, **ov)

    def build(self, **overrides: Any) -> ImageEncoderBase:
        cfg = self.make_config(**overrides)
        return build_encoder_from_config(cfg)

    @classmethod
    def list_names(cls) -> list[str]:
        return [m.name for m in cls]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.__members__

    @classmethod
    def get_config(cls, name: str, **overrides: Any) -> ImageEncoderConfig:
        try:
            return cls[name].make_config(**overrides)
        except KeyError as e:
            raise KeyError(f"Unknown image encoder '{name}'. Known: {cls.list_names()}") from e

    @classmethod
    def build_from_name(cls, name: str, **overrides: Any) -> ImageEncoderBase:
        try:
            return cls[name].build(**overrides)
        except KeyError as e:
            raise KeyError(f"Unknown image encoder '{name}'. Known: {cls.list_names()}") from e

    @staticmethod
    def default_transform(name: str) -> T.Compose:
        """Return the CPU-side transform for this encoder (minimal; GPU preprocessing handles resize/dtype)."""
        if name.lower() not in ImageEncoderRegistry.__members__:
            raise KeyError(f"Unknown image encoder '{name}'. Known: {ImageEncoderRegistry.list_names()}")
        return T.Compose([T.ToImage()])


__all__ = [
    "ImageEncoderConfig",
    "OpenPhenomConfig",
    "TimmCNNConfig",
    "DinoV3ViTConfig",
    "SubCellConfig",
    "ImageEncoderBase",
    "OpenPhenomEncoder",
    "TimmCNNEncoder",
    "DinoV3ViTEncoder",
    "SubCellEncoder",
    "build_encoder_from_config",
    "ImageEncoderRegistry",
]
