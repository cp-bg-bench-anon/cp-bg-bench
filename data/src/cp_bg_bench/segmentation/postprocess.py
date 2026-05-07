"""Post-processing for rule D: border-cell drop, nucleus-cell matching, per-cell stats.

All functions are pure numpy — no scipy dependency — so they are testable
in the default pixi env (without the seg-cpu / seg-gpu features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["SEG_COLUMNS", "compute_fov_stats", "drop_border_cells", "match_and_renumber"]

SEG_COLUMNS: list[str] = [
    "fov_id",
    "id_local",
    "nuc_id_global",
    "cell_id_global",
    "cyto_cent_row",
    "cyto_cent_col",
    "nuc_cent_row",
    "nuc_cent_col",
    "nuc_area",
    "cyto_area",
    "nuc_cyto_ratio",
    "n_cells_in_fov",
]


def drop_border_cells(mask: np.ndarray) -> np.ndarray:
    """Zero out any label whose pixels touch the image border.

    ``mask`` must be 2-D with label 0 = background. Returns a copy.
    """
    border = np.concatenate(
        [mask[0, :].ravel(), mask[-1, :].ravel(), mask[1:-1, 0].ravel(), mask[1:-1, -1].ravel()]
    )
    border_labels = set(int(v) for v in border if v > 0)
    if not border_labels:
        return mask.copy()
    result = mask.copy()
    result[np.isin(result, list(border_labels))] = 0
    return result


def _nucleus_centroids(nuc_mask: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return ``{label: (cent_row, cent_col)}`` for all non-zero labels.

    Uses ``np.bincount`` — O(n_pixels) regardless of label count.
    """
    nuc_labels = np.unique(nuc_mask)
    nuc_labels = nuc_labels[nuc_labels > 0]
    if nuc_labels.size == 0:
        return {}
    max_label = int(nuc_labels.max())
    _, cent_r, cent_c = _areas_and_centroids(nuc_mask, max_label)
    return {int(lbl): (int(cent_r[lbl - 1]), int(cent_c[lbl - 1])) for lbl in nuc_labels}


def match_and_renumber(
    nuc_mask: np.ndarray,
    cell_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """Match each nucleus to the cell containing its centroid; renumber 1..N.

    A nucleus is matched to the cell label at its centroid pixel. Each cell
    label can match at most one nucleus (first match wins if two nuclei share
    a cell). Unmatched nuclei and cells are zeroed out in the output.

    Returns ``(new_nuc_mask, new_cell_mask, nuc_orig_to_new, cell_orig_to_new)``
    where the two dicts map original label → new consecutive label.

    All steps are O(n_pixels): centroid computation via ``_areas_and_centroids``
    (bincount), renumbering via a look-up table applied to the full flat mask.
    """
    centroids = _nucleus_centroids(nuc_mask)  # O(n_pixels)

    pairs: list[tuple[int, int]] = []
    seen_cells: set[int] = set()
    for nuc_lbl, (cent_r, cent_c) in centroids.items():
        cell_lbl = int(cell_mask[cent_r, cent_c])
        if cell_lbl > 0 and cell_lbl not in seen_cells:
            seen_cells.add(cell_lbl)
            pairs.append((nuc_lbl, cell_lbl))

    if not pairs:
        return np.zeros_like(nuc_mask), np.zeros_like(cell_mask), {}, {}

    # Build per-label LUT so renumbering is O(n_pixels) via fancy indexing.
    max_nuc = max(n for n, _ in pairs)
    max_cell = max(c for _, c in pairs)
    nuc_lut = np.zeros(max_nuc + 1, dtype=np.uint32)
    cell_lut = np.zeros(max_cell + 1, dtype=np.uint32)
    nuc_orig_to_new: dict[int, int] = {}
    cell_orig_to_new: dict[int, int] = {}

    for new_id, (nuc_lbl, cell_lbl) in enumerate(pairs, start=1):
        nuc_lut[nuc_lbl] = new_id
        cell_lut[cell_lbl] = new_id
        nuc_orig_to_new[nuc_lbl] = new_id
        cell_orig_to_new[cell_lbl] = new_id

    # Clamp labels outside the LUT range to 0 (unmatched), then index.
    nuc_safe = np.where(nuc_mask <= max_nuc, nuc_mask, 0)
    cell_safe = np.where(cell_mask <= max_cell, cell_mask, 0)

    return nuc_lut[nuc_safe], cell_lut[cell_safe], nuc_orig_to_new, cell_orig_to_new


def _areas_and_centroids(
    mask: np.ndarray, max_label: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (areas, cent_rows, cent_cols) for labels 1..max_label.

    Uses ``np.bincount`` — O(n_pixels) regardless of label count.
    """
    rows, cols = np.indices(mask.shape)
    flat = mask.ravel()
    weight_r = rows.ravel().astype(float)
    weight_c = cols.ravel().astype(float)
    size = max_label + 1
    areas = np.bincount(flat, minlength=size).astype(float)[1:]
    sum_r = np.bincount(flat, weights=weight_r, minlength=size)[1:]
    sum_c = np.bincount(flat, weights=weight_c, minlength=size)[1:]
    safe = np.where(areas > 0, areas, 1.0)
    return areas, sum_r / safe, sum_c / safe


def compute_fov_stats(
    nuc_mask: np.ndarray,
    cell_mask: np.ndarray,
    fov_id: str,
    nuc_orig_to_new: dict[int, int],
    cell_orig_to_new: dict[int, int],
) -> pd.DataFrame:
    """Compute per-cell stats from the renumbered masks for one FOV."""
    n_cells = len(nuc_orig_to_new)
    if n_cells == 0:
        return pd.DataFrame(columns=SEG_COLUMNS)

    nuc_areas, nuc_cent_r, nuc_cent_c = _areas_and_centroids(nuc_mask, n_cells)
    cell_areas, cell_cent_r, cell_cent_c = _areas_and_centroids(cell_mask, n_cells)

    new_to_nuc_orig = {v: k for k, v in nuc_orig_to_new.items()}
    new_to_cell_orig = {v: k for k, v in cell_orig_to_new.items()}

    rows = []
    for new_id in range(1, n_cells + 1):
        i = new_id - 1
        nuc_area = int(nuc_areas[i])
        cyto_area = int(cell_areas[i])
        rows.append(
            {
                "fov_id": fov_id,
                "id_local": new_id,
                "nuc_id_global": new_to_nuc_orig[new_id],
                "cell_id_global": new_to_cell_orig[new_id],
                "cyto_cent_row": float(cell_cent_r[i]),
                "cyto_cent_col": float(cell_cent_c[i]),
                "nuc_cent_row": float(nuc_cent_r[i]),
                "nuc_cent_col": float(nuc_cent_c[i]),
                "nuc_area": nuc_area,
                "cyto_area": cyto_area,
                "nuc_cyto_ratio": nuc_area / cyto_area if cyto_area > 0 else 0.0,
                "n_cells_in_fov": n_cells,
            }
        )

    return pd.DataFrame(rows)
