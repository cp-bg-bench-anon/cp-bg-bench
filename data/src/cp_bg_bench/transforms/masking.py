"""Apply nucleus/cell masks to fluorescent channels (rule J: derive_seg)."""

from __future__ import annotations

import numpy as np

__all__ = ["apply_masks"]


def apply_masks(cell: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Apply nucleus/cell masks to fluorescent channels.

    ``cell`` is ``(C_img, H, W)`` uint8 — fluorescence channels only.
    ``seg``  is ``(2, H, W)`` uint8 — [NucMask, CellMask].

    Returns a masked copy of ``cell``:
    - channel 0 (DNA) is zeroed outside the nucleus mask (``seg[0]``).
    - channels 1+ (AGP, ER, Mito, RNA, …) are zeroed outside the cell mask (``seg[1]``).
    """
    if cell.ndim != 3:
        raise ValueError(f"cell must be (C, H, W), got shape {cell.shape}")
    if seg.ndim != 3 or seg.shape[0] != 2:
        raise ValueError(f"seg must be (2, H, W), got shape {seg.shape}")
    out = cell.copy()
    nuc_bool = seg[0] != 0
    cell_bool = seg[1] != 0
    out[0] = cell[0] * nuc_bool
    if cell.shape[0] > 1:
        out[1:] = cell[1:] * cell_bool
    return out


def _apply_masks_batch(
    cell_list: list[bytes],
    seg_list: list[bytes],
    cell_shape: tuple[int, int, int],
    seg_shape: tuple[int, int, int],
) -> list[bytes]:
    """Apply masks to parallel lists of serialised ``(C, H, W)`` uint8 blobs.

    Internal helper used by the Snakemake derive_seg script; not part of the
    public API because shapes must be supplied by the caller.
    """
    expected_cell = cell_shape[0] * cell_shape[1] * cell_shape[2]
    expected_seg = seg_shape[0] * seg_shape[1] * seg_shape[2]
    out: list[bytes] = []
    for cell_blob, seg_blob in zip(cell_list, seg_list, strict=True):
        cell_arr = np.frombuffer(cell_blob, dtype=np.uint8)
        seg_arr = np.frombuffer(seg_blob, dtype=np.uint8)
        if cell_arr.size != expected_cell:
            raise ValueError(
                f"cell blob size {cell_arr.size} != expected {expected_cell} {cell_shape}"
            )
        if seg_arr.size != expected_seg:
            raise ValueError(f"seg blob size {seg_arr.size} != expected {expected_seg} {seg_shape}")
        out.append(
            apply_masks(cell_arr.reshape(cell_shape), seg_arr.reshape(seg_shape)).tobytes(order="C")
        )
    return out
