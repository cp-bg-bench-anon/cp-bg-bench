"""Per-FOV percentile normalisation: uint16 → uint8."""

from __future__ import annotations

import numpy as np

__all__ = ["per_fov_percentile_norm", "naive_u16_to_u8"]


def per_fov_percentile_norm(
    stack: np.ndarray,
    low: float = 0.001,
    high: float = 0.999,
) -> np.ndarray:
    """(C, H, W) uint16 → (C, H, W) uint8 via per-channel percentile clip.

    Each channel is independently normalised: pixels below the ``low``
    quantile map to 0, pixels above the ``high`` quantile map to 255.
    A constant channel (hi == lo) maps entirely to 0.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (C, H, W), got shape {stack.shape}")
    result = np.empty(stack.shape, dtype=np.uint8)
    for c in range(stack.shape[0]):
        ch = stack[c].astype(np.float32)
        lo_val = float(np.quantile(ch, low))
        hi_val = float(np.quantile(ch, high))
        if hi_val <= lo_val:
            result[c] = 0
        else:
            result[c] = np.clip((ch - lo_val) / (hi_val - lo_val) * 255.0, 0.0, 255.0).astype(
                np.uint8
            )
    return result


def naive_u16_to_u8(stack: np.ndarray) -> np.ndarray:
    """(C, H, W) uint16 → (C, H, W) uint8 by dividing by 257 (≈ 65535/255)."""
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (C, H, W), got shape {stack.shape}")
    return (stack.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
