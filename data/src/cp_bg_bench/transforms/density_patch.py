"""Corner intensity patches for density-encoding (rules K and L)."""

from __future__ import annotations

import numpy as np

__all__ = ["draw_corner_patches", "draw_corner_patches_batch"]


def draw_corner_patches(
    arr: np.ndarray,
    intensity: int,
    patch_size: int,
    pad: int,
) -> np.ndarray:
    """Draw four corner squares of ``patch_size`` × ``patch_size`` at ``pad`` inset.

    All channels receive the same ``intensity`` value (0-255 uint8) in the
    corner regions. Returns a copy — does not mutate ``arr``.
    """
    if arr.ndim != 3:
        raise ValueError(f"array must be (C, H, W), got shape {arr.shape}")
    _, H, W = arr.shape
    if pad + patch_size > H or pad + patch_size > W:
        raise ValueError(
            f"patch_size={patch_size} + pad={pad} exceeds array spatial dims ({H}, {W})"
        )

    out = arr.copy()
    i = np.uint8(np.clip(intensity, 0, 255))

    r0, r1 = pad, pad + patch_size
    r2, r3 = H - pad - patch_size, H - pad
    c0, c1 = pad, pad + patch_size
    c2, c3 = W - pad - patch_size, W - pad

    out[:, r0:r1, c0:c1] = i  # top-left
    out[:, r0:r1, c2:c3] = i  # top-right
    out[:, r2:r3, c0:c1] = i  # bottom-left
    out[:, r2:r3, c2:c3] = i  # bottom-right
    return out


def draw_corner_patches_batch(
    cell_list: list[bytes],
    intensities: list[float],
    cell_shape: tuple[int, int, int],
    patch_size: int,
    pad: int,
) -> list[bytes]:
    """Apply corner patches to a list of serialised cell tensors.

    Args:
        cell_list: Serialised ``(C, H, W)`` uint8 blobs.
        intensities: Scalar density value per cell (0-255); NaN → 0.
        cell_shape: ``(C, H, W)`` used for deserialisation.
        patch_size: Side length of each corner patch in pixels.
        pad: Inset from each image edge in pixels.
    """
    intens_arr = np.asarray(intensities, dtype=np.float32)
    np.nan_to_num(intens_arr, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
    intens_u8 = np.clip(intens_arr, 0, 255).astype(np.uint8)

    expected = int(np.prod(cell_shape))
    out: list[bytes] = []
    for blob, i in zip(cell_list, intens_u8, strict=False):
        arr = np.frombuffer(blob, dtype=np.uint8)
        if arr.size != expected:
            raise ValueError(f"blob size {arr.size} != expected {expected} for shape {cell_shape}")
        arr = arr.reshape(cell_shape)
        out.append(draw_corner_patches(arr, int(i), patch_size, pad).tobytes(order="C"))
    return out
