"""Per-cell crop extraction from full-image and segmentation Zarr v3 stores."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from cp_bg_bench.crops.normalize import naive_u16_to_u8, per_fov_percentile_norm
from cp_bg_bench.io.zarr_io import plate_store_path, seg_store_path

logger = logging.getLogger(__name__)

__all__ = ["CROPS_SCHEMA", "extract_plate_crops"]

# Arrow schema for per-plate crops parquet.
# "cell" is (C_img, patch_size, patch_size) uint8 — fluorescence channels only.
# "mask" is (N_MASK_CHANNELS, patch_size, patch_size) uint8 — [NucMask, CellMask].
# "perturbation": unified perturbation identity (gene for RxRx1/3, InChIKey for JUMP).
# "batch": dataset-specific grouping key for train/val/test splitting.
# "treatment": sub-perturbation provenance (siRNA/guide ID; empty for JUMP).
CROPS_SCHEMA = pa.schema(
    [
        ("row_key", pa.string()),
        ("source", pa.string()),
        ("plate", pa.string()),
        ("well", pa.string()),
        ("tile", pa.string()),
        ("id_local", pa.int64()),
        ("nuc_area", pa.int64()),
        ("cyto_area", pa.int64()),
        ("nuc_cyto_ratio", pa.float64()),
        ("n_cells_in_fov", pa.int64()),
        ("n_cells_scaled", pa.float64()),
        ("Metadata_JCP2022", pa.string()),
        ("Metadata_InChIKey", pa.string()),
        ("Metadata_PlateType", pa.string()),
        ("perturbation", pa.string()),
        ("batch", pa.string()),
        ("treatment", pa.string()),
        ("mask", pa.binary()),
        ("cell", pa.binary()),
    ]
)

_META_COLS = [
    "Metadata_JCP2022",
    "Metadata_InChIKey",
    "Metadata_PlateType",
]

N_MASK_CHANNELS = 2  # NucMask, CellMask (always first two channels in the crop)


def _extract_crop(
    img_uint8: np.ndarray,
    label_arr: np.ndarray,
    id_local: int,
    cent_row: int,
    cent_col: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(seg, cell)`` crops for one cell, both ``uint8``.

    ``seg``  is ``(N_MASK_CHANNELS, patch_size, patch_size)`` — [NucMask, CellMask].
    ``cell`` is ``(C_img, patch_size, patch_size)`` — fluorescence channels only.
    Zero-pads if the centroid is within ``patch_size//2`` of the border.
    """
    C, H, W = img_uint8.shape
    half = patch_size // 2

    r0 = max(0, cent_row - half)
    r1 = min(H, cent_row + half)
    c0 = max(0, cent_col - half)
    c1 = min(W, cent_col + half)

    img_crop = img_uint8[:, r0:r1, c0:c1]
    nuc_crop = (label_arr[0, r0:r1, c0:c1] == id_local).astype(np.uint8) * 255
    cell_crop = (label_arr[1, r0:r1, c0:c1] == id_local).astype(np.uint8) * 255
    seg_crop = np.stack([nuc_crop, cell_crop], axis=0)

    rh, rw = img_crop.shape[1], img_crop.shape[2]
    if rh != patch_size or rw != patch_size:
        raise ValueError(
            f"Crop at centroid ({cent_row}, {cent_col}) is ({rh}, {rw}), "
            f"expected ({patch_size}, {patch_size}). "
            "Filter near-border cells before calling _extract_crop."
        )
    return seg_crop, img_crop


def extract_plate_crops(
    plate_key: str,
    selected_df: pd.DataFrame,
    output_root: Path | str,
    patch_size: int = 150,
    norm_scheme: str = "per_fov_percentile",
    norm_low: float = 0.001,
    norm_high: float = 0.999,
) -> pa.Table:
    """Extract crops for all selected cells in one plate.

    Reads full-image and segmentation Zarr v3 stores; processes per FOV
    (computes normalisation once per FOV, then extracts all selected cells).

    Returns a :class:`pyarrow.Table` matching :data:`CROPS_SCHEMA`.
    """
    import zarr

    output_root = Path(output_root)

    # Parse plate_key → (source, batch, plate)
    source, batch, plate = plate_key.split("__", 2)

    full_img_path = plate_store_path(output_root, source, batch, plate)
    seg_path = seg_store_path(output_root, source, batch, plate)

    full_img_group = zarr.open_group(str(full_img_path), mode="r", zarr_format=3)
    seg_group = zarr.open_group(str(seg_path), mode="r", zarr_format=3)

    _required = {"perturbation", "batch", "treatment"}
    missing = _required - set(selected_df.columns)
    if missing:
        raise ValueError(
            f"extract_plate_crops: selected_df is missing required columns: {sorted(missing)}. "
            "Assign perturbation/batch/treatment before calling."
        )

    rows: list[dict] = []

    for fov_id, fov_df in selected_df.groupby("fov_id"):
        fov_id = str(fov_id)
        if fov_id not in full_img_group:
            logger.warning("FOV %s not in full_images zarr, skipping", fov_id)
            continue
        if fov_id not in seg_group:
            logger.warning("FOV %s not in segmentation zarr, skipping", fov_id)
            continue

        # Load and normalise the full FOV image once
        raw_stack = full_img_group[fov_id][:]  # (C, H, W) uint16
        _C, H, W = raw_stack.shape
        half = patch_size // 2
        in_bounds = (
            (fov_df["cyto_cent_row"] >= half)
            & (fov_df["cyto_cent_row"] < H - half)
            & (fov_df["cyto_cent_col"] >= half)
            & (fov_df["cyto_cent_col"] < W - half)
        )
        n_dropped = int((~in_bounds).sum())
        if n_dropped:
            logger.warning(
                "FOV %s: dropping %d cell(s) with centroid within %d px of border",
                fov_id,
                n_dropped,
                half,
            )
            fov_df = fov_df[in_bounds]
        if fov_df.empty:
            continue

        if norm_scheme == "per_fov_percentile":
            img_uint8 = per_fov_percentile_norm(raw_stack, norm_low, norm_high)
        else:
            img_uint8 = naive_u16_to_u8(raw_stack)

        label_arr = seg_group[fov_id][:]  # (2, H, W) uint32

        # Split fov_id → (source, pipeline_batch, plate, well, tile)
        parts = fov_id.split("__", 4)
        src, _pipeline_bat, plt, well, tile = parts

        for row in fov_df.itertuples(index=False):
            id_local = int(row.id_local)
            seg_crop, img_crop = _extract_crop(
                img_uint8,
                label_arr,
                id_local,
                int(row.cyto_cent_row),
                int(row.cyto_cent_col),
                patch_size,
            )
            row_key = f"{fov_id}__{id_local}"

            meta = {col: str(getattr(row, col, "")) for col in _META_COLS}

            rows.append(
                {
                    "row_key": row_key,
                    "source": src,
                    "plate": plt,
                    "well": well,
                    "tile": tile,
                    "id_local": id_local,
                    "nuc_area": int(row.nuc_area),
                    "cyto_area": int(row.cyto_area),
                    "nuc_cyto_ratio": float(row.nuc_cyto_ratio),
                    "n_cells_in_fov": int(row.n_cells_in_fov),
                    "n_cells_scaled": float(row.n_cells_scaled),
                    **meta,
                    "perturbation": str(getattr(row, "perturbation", "")),
                    "batch": str(getattr(row, "batch", "")),
                    "treatment": str(getattr(row, "treatment", "")),
                    "mask": seg_crop.tobytes(order="C"),
                    "cell": img_crop.tobytes(order="C"),
                }
            )

    if not rows:
        return pa.table({col.name: pa.array([], type=col.type) for col in CROPS_SCHEMA})

    df = pd.DataFrame(rows)
    return pa.Table.from_pandas(df, schema=CROPS_SCHEMA, preserve_index=False)
