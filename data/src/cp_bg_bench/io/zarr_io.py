"""Zarr v3 helpers for cp-bg-bench pipeline artifacts.

Central place for the store conventions the plan commits to (§7.3):
``zarr_format=3`` explicit on every open, ``BloscCodec(zstd, clevel=3,
shuffle)`` compression, ``dimension_names=("channel", "y", "x")`` on every
array, and one chunk per FOV.

Consumers pass a per-plate store path and a ``(C, H, W)`` uint16 stack;
the module handles store creation, idempotent FOV writes, and consistent
metadata. Zarr is imported lazily so this module stays importable without
the optional ``io`` feature env active (useful for collectors / test
harnesses that never touch zarr).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "LABEL_DIMENSION_NAMES",
    "ZARR_DIMENSION_NAMES",
    "default_codec",
    "open_plate_group",
    "plate_store_path",
    "seg_store_path",
    "write_fov_array",
    "write_label_array",
]

# Dimension names stamped on every per-FOV array. Kept as a module-level
# constant so downstream rules assert against the same tuple.
ZARR_DIMENSION_NAMES: tuple[str, str, str] = ("channel", "y", "x")

# Dimension names for per-FOV label images (nucleus + cell masks).
LABEL_DIMENSION_NAMES: tuple[str, str, str] = ("label", "y", "x")

# Lazily-built module-level singleton: codec specs are immutable value
# objects, so rebuilding one per-array write (2k+ times per smoke plate,
# millions at prod scale) is pure waste.
_DEFAULT_CODEC: Any | None = None


def default_codec() -> Any:
    """``BloscCodec(zstd, clevel=3, shuffle)`` — pipeline-wide default."""
    global _DEFAULT_CODEC
    if _DEFAULT_CODEC is None:
        from zarr.codecs import BloscCodec

        _DEFAULT_CODEC = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    return _DEFAULT_CODEC


def plate_store_path(output_root: Path | str, source: str, batch: str, plate: str) -> Path:
    """Resolve the per-plate Zarr store path under ``output_root/full_images``."""
    return Path(output_root) / "full_images" / f"{source}__{batch}__{plate}.zarr"


def seg_store_path(output_root: Path | str, source: str, batch: str, plate: str) -> Path:
    """Resolve the per-plate segmentation Zarr store path under ``output_root/segmentation``."""
    return Path(output_root) / "segmentation" / f"{source}__{batch}__{plate}.zarr"


def open_plate_group(path: Path | str) -> Any:
    """Open-or-create a Zarr v3 group at ``path`` in append mode.

    Append mode (``mode="a"``) lets multiple Snakemake jobs converge on
    the same plate store without racing — each job creates its own
    per-FOV array in a fresh subdirectory. Callers must still guarantee
    FOV ids are partitioned across concurrent writers (the pipeline's
    ``assign_snakemake_batches`` enforces this by construction).
    """
    import zarr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(store=str(path), mode="a", zarr_format=3)


def write_fov_array(group: Any, fov_id: str, stack: np.ndarray) -> Any:
    """Write a single ``(C, H, W)`` uint16 FOV array under ``group[fov_id]``.

    One chunk per FOV; the default codec and ``dimension_names`` are
    applied. If an array with the same name and shape already exists
    the call is a no-op (idempotent rerun). A shape mismatch on an
    existing array raises ``ValueError`` rather than silently
    overwriting — downstream rules rely on raw images being immutable
    once written.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (C, H, W), got shape {stack.shape}")
    if stack.dtype != np.uint16:
        raise ValueError(f"stack must be uint16, got dtype {stack.dtype}")

    if fov_id in group:
        existing = group[fov_id]
        if tuple(existing.shape) != tuple(stack.shape):
            raise ValueError(
                f"FOV {fov_id!r} already exists with shape {tuple(existing.shape)}, "
                f"refusing to overwrite with shape {tuple(stack.shape)}"
            )
        return existing

    array = group.create_array(
        name=fov_id,
        shape=stack.shape,
        chunks=stack.shape,
        dtype=stack.dtype,
        compressors=default_codec(),
        dimension_names=ZARR_DIMENSION_NAMES,
    )
    array[:] = stack
    return array


def write_label_array(group: Any, fov_id: str, label_img: np.ndarray) -> Any:
    """Write a ``(2, H, W)`` uint32 label image under ``group[fov_id]``.

    Channel 0 = nucleus labels, channel 1 = cell labels; both renumbered
    so nucleus i and cell i are a matched pair. Idempotent on existing
    arrays with matching shape; raises ``ValueError`` on shape mismatch.
    """
    if label_img.ndim != 3 or label_img.shape[0] != 2:
        raise ValueError(f"label_img must be (2, H, W), got shape {label_img.shape}")
    if label_img.dtype != np.uint32:
        raise ValueError(f"label_img must be uint32, got dtype {label_img.dtype}")

    if fov_id in group:
        existing = group[fov_id]
        if tuple(existing.shape) != tuple(label_img.shape):
            raise ValueError(
                f"FOV {fov_id!r} label already exists with shape {tuple(existing.shape)}, "
                f"refusing to overwrite with shape {tuple(label_img.shape)}"
            )
        return existing

    array = group.create_array(
        name=fov_id,
        shape=label_img.shape,
        chunks=label_img.shape,
        dtype=label_img.dtype,
        compressors=default_codec(),
        dimension_names=LABEL_DIMENSION_NAMES,
    )
    array[:] = label_img
    return array
