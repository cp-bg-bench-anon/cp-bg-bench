"""Segmentation module: Cellpose-SAM driver and postprocessing for rule D."""

from cp_bg_bench.segmentation.cpsam import segment_plate
from cp_bg_bench.segmentation.postprocess import (
    compute_fov_stats,
    drop_border_cells,
    match_and_renumber,
)

__all__ = [
    "compute_fov_stats",
    "drop_border_cells",
    "match_and_renumber",
    "segment_plate",
]
