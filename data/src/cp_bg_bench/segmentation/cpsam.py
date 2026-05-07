"""Cellpose-SAM driver for rule D: full per-plate segmentation.

Cellpose, Zarr, and pandas are imported at call time (lazy) so this module
stays importable in environments without the ``seg-cpu`` / ``seg-gpu``
features installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cp_bg_bench.io.zarr_io import (
    open_plate_group,
    plate_store_path,
    seg_store_path,
    write_label_array,
)
from cp_bg_bench.segmentation.postprocess import (
    SEG_COLUMNS,
    compute_fov_stats,
    drop_border_cells,
    match_and_renumber,
)

logger = logging.getLogger(__name__)

__all__ = ["segment_plate"]


def _get_diameter(seg_config: Any, source: str) -> tuple[float, float]:
    """Return ``(nucleus_diam, cell_diam)`` for *source*, falling back to defaults."""
    if source in seg_config.per_source_diameters:
        d = seg_config.per_source_diameters[source]
        return float(d.nucleus), float(d.cytosol)
    d = seg_config.default_diameters
    logger.warning(
        "source %r not in per_source_diameters; using defaults nucleus=%.1f cytosol=%.1f",
        source,
        d.nucleus,
        d.cytosol,
    )
    return float(d.nucleus), float(d.cytosol)


def _make_cellpose_model(model_type: str, use_gpu: bool) -> Any:
    """Lazy-import CellposeModel (cellpose ≥4)."""
    from cellpose import models

    return models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)


def _build_inference_inputs(
    full_stack: np.ndarray,
    channels_for_nucleus: list[int],
    channels_for_cell: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract nucleus and cell images from a single FOV stack."""
    nuc_img = full_stack[channels_for_nucleus[0]]
    if len(channels_for_cell) >= 2:
        cell_img = np.stack([full_stack[c] for c in channels_for_cell], axis=-1)
    else:
        cell_img = full_stack[channels_for_cell[0]]
    return nuc_img, cell_img


def segment_plate(
    source: str,
    batch: str,
    plate: str,
    output_root: Path | str,
    seg_config: Any,
    use_gpu: bool = False,
    fov_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Segment all FOVs in one plate. Returns a concatenated per-cell DataFrame.

    Reads raw FOV arrays from ``output_root/full_images/{source}__{batch}__{plate}.zarr``.
    Writes ``(2, H, W)`` uint32 label images to
    ``output_root/segmentation/{source}__{batch}__{plate}.zarr``.

    Each label image has nucleus labels in channel 0 and cell labels in channel 1;
    label integer *i* in both channels is a matched nucleus-cell pair.
    Cells touching the FOV border and unmatched nuclei/cells are zeroed out.

    Inference is batched: all nucleus images are submitted in a single
    ``model.eval()`` call, then all cell images in a second call. This lets
    Cellpose pack image tiles from multiple FOVs into the same GPU batch,
    giving 10-20x throughput over per-FOV eval calls.

    Returns ``pd.DataFrame`` with columns defined in
    :data:`~cp_bg_bench.segmentation.postprocess.SEG_COLUMNS`.
    """
    import zarr

    output_root = Path(output_root)
    nuc_diam, cell_diam = _get_diameter(seg_config, source)
    model = _make_cellpose_model(seg_config.model, use_gpu)

    channels_for_nucleus: list[int] = list(seg_config.channels_for_nucleus)
    channels_for_cell: list[int] = list(seg_config.channels_for_cell)

    input_path = plate_store_path(output_root, source, batch, plate)
    output_path = seg_store_path(output_root, source, batch, plate)

    input_group = zarr.open_group(store=str(input_path), mode="r", zarr_format=3)
    output_group = open_plate_group(output_path)

    fov_ids = sorted(fov_ids) if fov_ids is not None else sorted(input_group.keys())
    store_keys = set(input_group.keys())
    missing = [f for f in fov_ids if f not in store_keys]
    if missing:
        logger.warning(
            "segment_plate: %d FOV(s) in metadata absent from Zarr store (skipped): %s",
            len(missing),
            missing,
        )
        fov_ids = [f for f in fov_ids if f in store_keys]
    logger.info(
        "segment_plate: source=%s batch=%s plate=%s → %d FOVs, nuc_diam=%.1f cell_diam=%.1f",
        source,
        batch,
        plate,
        len(fov_ids),
        nuc_diam,
        cell_diam,
    )

    # Load all FOV stacks once; build per-FOV nuc/cell image lists.
    nuc_imgs: list[np.ndarray] = []
    cell_imgs: list[np.ndarray] = []
    for fov_id in fov_ids:
        nuc_img, cell_img = _build_inference_inputs(
            input_group[fov_id][:], channels_for_nucleus, channels_for_cell
        )
        nuc_imgs.append(nuc_img)
        cell_imgs.append(cell_img)

    # Two model.eval() calls for the whole plate — tiles from all FOVs are
    # packed into the same GPU batch, saturating the accelerator.
    nuc_results, _, _ = model.eval(nuc_imgs, diameter=nuc_diam)
    cell_results, _, _ = model.eval(cell_imgs, diameter=cell_diam)

    all_stats: list[pd.DataFrame] = []
    for fov_id, nuc_raw, cell_raw in zip(fov_ids, nuc_results, cell_results, strict=True):
        nuc_raw = np.asarray(nuc_raw, dtype=np.uint32)
        cell_raw = np.asarray(cell_raw, dtype=np.uint32)

        nuc_clean = drop_border_cells(nuc_raw)
        cell_clean = drop_border_cells(cell_raw)
        nuc_final, cell_final, nuc_orig_to_new, cell_orig_to_new = match_and_renumber(
            nuc_clean, cell_clean
        )

        label_image = np.stack([nuc_final, cell_final], axis=0)
        write_label_array(output_group, fov_id, label_image)

        stats = compute_fov_stats(nuc_final, cell_final, fov_id, nuc_orig_to_new, cell_orig_to_new)
        all_stats.append(stats)
        logger.debug("FOV %s: %d matched cells", fov_id, len(nuc_orig_to_new))

    if not all_stats:
        return pd.DataFrame(columns=SEG_COLUMNS)

    result = pd.concat(all_stats, ignore_index=True)
    logger.info(
        "segment_plate: %s__%s__%s complete — %d total cells", source, batch, plate, len(result)
    )
    return result
