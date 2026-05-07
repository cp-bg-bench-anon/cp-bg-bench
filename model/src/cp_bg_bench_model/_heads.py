from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_to_multiple(v: int | float, m: int = 64) -> int:
    v = int(round(v))
    return max(m, ((v + m - 1) // m) * m)


class GEGLU(nn.Module):
    """GEGLU: split features -> a, b; y = a * GELU(b)."""

    def __init__(self, in_features: int, hidden: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, 2 * hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class LazyLayerNorm(nn.Module):
    """
    LayerNorm that infers normalized_shape from x.shape[-1] on first forward.
    """

    def __init__(self, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self._ln: nn.LayerNorm | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ln is None:
            n = int(x.shape[-1])
            ln = nn.LayerNorm(n, eps=self.eps, elementwise_affine=self.elementwise_affine)
            self._ln = ln.to(device=x.device, dtype=x.dtype)
        return self._ln(x)


class LinearHead(nn.Module):
    """
    Simple linear projection head:
      LayerNorm(in) -> Dropout -> Linear -> LayerNorm(out)

    If in_features is None or <= 0, infer it on first forward (lazy init).
    """

    def __init__(
        self,
        in_features: int | None,
        embed_dim: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
        norm_out: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features) if (in_features is not None and int(in_features) > 0) else -1
        self.embed_dim = int(embed_dim)

        lazy_in = self.in_features <= 0

        self.norm_in: nn.Module = LazyLayerNorm(eps=eps) if lazy_in else nn.LayerNorm(self.in_features, eps=eps)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # name "proj" so your existing logging can discover dims
        self.proj: nn.Module
        if lazy_in:
            self.proj = nn.LazyLinear(self.embed_dim, bias=bool(bias))
        else:
            self.proj = nn.Linear(self.in_features, self.embed_dim, bias=bool(bias))

        self.norm_out: nn.Module = nn.LayerNorm(self.embed_dim, eps=eps) if norm_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.norm_out(x)
        return x


class ProjectionHead(nn.Module):
    """
    Scalable projection head:
      LayerNorm -> GEGLU -> Dropout -> Linear -> (optional gated skip) -> output

    Args:
      in_features: encoder output dim
      embed_dim:   target embedding dim
      width_alpha: scales hidden = alpha * sqrt(in_features * embed_dim)
      round_to:    round hidden to this multiple
      min_hidden/max_hidden: clamps hidden
      dropout:     dropout after GEGLU
      use_skip:    add learnable skip from input (projected) to output
      activation:  "geglu" | "gelu" | "silu"
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int,
        *,
        width_alpha: float = 1.2,
        round_to: int = 64,
        min_hidden: int = 256,
        max_hidden: int = 4096,
        dropout: float = 0.1,
        use_skip: bool = True,
        activation: Literal["geglu", "gelu", "silu"] = "geglu",
    ):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim

        # --- width heuristic ---
        h_raw = width_alpha * math.sqrt(float(in_features) * float(embed_dim))
        hidden = int(max(min_hidden, min(max_hidden, _round_to_multiple(h_raw, round_to))))

        self.norm_in = nn.LayerNorm(in_features)

        if activation == "geglu":
            self.act = GEGLU(in_features, hidden, bias=True)
            first_is_linear = False
        else:
            self.fc1 = nn.Linear(in_features, hidden, bias=True)
            self.act = nn.GELU() if activation == "gelu" else nn.SiLU()
            first_is_linear = True

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, embed_dim, bias=True)

        # Optional learnable skip (project input to embed_dim and gate it)
        self.use_skip = use_skip
        if use_skip:
            self.skip = nn.Linear(in_features, embed_dim, bias=False)
            # start at 0 → model learns to use the skip if it helps
            self.skip_scale = nn.Parameter(torch.zeros(1))

        # Kaiming/Xavier are fine; LayerNorm keeps things stable.
        # (Defaults are OK; customize if you like.)

        # Cache flag for forward (minor perf)
        self._first_is_linear = first_is_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(x)
        if self._first_is_linear:
            h = self.act(self.fc1(x))
        else:
            h = self.act(x)  # GEGLU block includes its own linear
        h = self.dropout(h)
        y = self.fc2(h)
        if self.use_skip:
            y = y + self.skip_scale * self.skip(x)
        return y
