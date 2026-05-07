"""Tests for cp_bg_bench.segmentation (rule D)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import zarr

from cp_bg_bench.io.zarr_io import open_plate_group, plate_store_path, write_fov_array
from cp_bg_bench.segmentation.postprocess import (
    SEG_COLUMNS,
    compute_fov_stats,
    drop_border_cells,
    match_and_renumber,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_seg_config(
    nucleus_diam: float = 21.0,
    cell_diam: float = 53.0,
    source_overrides: dict | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.model = "cpsam"
    cfg.channels_for_nucleus = [0]
    cfg.channels_for_cell = [3, 0]
    cfg.default_diameters.nucleus = nucleus_diam
    cfg.default_diameters.cytosol = cell_diam
    cfg.per_source_diameters = source_overrides or {}
    return cfg


def _make_fov_stack(c: int = 5, h: int = 20, w: int = 20) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 1000, size=(c, h, w), dtype=np.uint16)


def _single_cell_masks(
    h: int = 20,
    w: int = 20,
    nuc_box: tuple = (5, 5, 9, 9),
    cell_box: tuple = (4, 4, 11, 11),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nuc_mask, cell_mask) uint32 with one interior cell."""
    nuc_mask = np.zeros((h, w), dtype=np.uint32)
    nuc_mask[nuc_box[0] : nuc_box[2], nuc_box[1] : nuc_box[3]] = 1
    cell_mask = np.zeros((h, w), dtype=np.uint32)
    cell_mask[cell_box[0] : cell_box[2], cell_box[1] : cell_box[3]] = 1
    return nuc_mask, cell_mask


def _mock_model(nuc_mask: np.ndarray, cell_mask: np.ndarray) -> MagicMock:
    """Return a mock CellposeModel whose eval() returns batched nuc/cell masks.

    Matches the cellpose ≥4 API: eval() returns (masks, flows, styles) — 3 values,
    not 4. The first call is treated as the nucleus pass, the second as cell.
    Masks list length matches the number of input images.
    """
    model = MagicMock()
    call_count = [0]

    def eval_side_effect(imgs, **kwargs):
        call_count[0] += 1
        n = len(imgs)
        if call_count[0] % 2 == 1:  # odd → nucleus call
            return ([nuc_mask.copy() for _ in range(n)], [[]] * n, [[]] * n)
        return ([cell_mask.copy() for _ in range(n)], [[]] * n, [[]] * n)

    model.eval.side_effect = eval_side_effect
    return model


# ---------------------------------------------------------------------------
# TestDropBorderCells
# ---------------------------------------------------------------------------


class TestDropBorderCells:
    def test_top_row_label_dropped(self) -> None:
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[0, 2] = 1
        mask[2:4, 2:4] = 1  # same label in interior — all pixels of label 1 zeroed
        result = drop_border_cells(mask)
        assert not np.any(result == 1)

    def test_bottom_row_label_dropped(self) -> None:
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[5, 3] = 2
        result = drop_border_cells(mask)
        assert not np.any(result == 2)

    def test_left_col_label_dropped(self) -> None:
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[3, 0] = 3
        result = drop_border_cells(mask)
        assert not np.any(result == 3)

    def test_right_col_label_dropped(self) -> None:
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[3, 5] = 4
        result = drop_border_cells(mask)
        assert not np.any(result == 4)

    def test_interior_label_preserved(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint32)
        mask[2:5, 2:5] = 7
        result = drop_border_cells(mask)
        assert np.sum(result == 7) == np.sum(mask == 7)

    def test_empty_mask(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint32)
        result = drop_border_cells(mask)
        assert not np.any(result > 0)

    def test_returns_copy_not_inplace(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint32)
        mask[0, 0] = 5
        result = drop_border_cells(mask)
        assert mask[0, 0] == 5  # original untouched
        assert result[0, 0] == 0

    def test_multiple_border_labels_dropped(self) -> None:
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[0, :] = 1  # label 1 on top border
        mask[5, :] = 2  # label 2 on bottom border
        mask[2:4, 2:4] = 3  # label 3 interior
        result = drop_border_cells(mask)
        assert not np.any(result == 1)
        assert not np.any(result == 2)
        assert np.any(result == 3)


# ---------------------------------------------------------------------------
# TestMatchAndRenumber
# ---------------------------------------------------------------------------


class TestMatchAndRenumber:
    def test_single_matched_pair(self) -> None:
        nuc_mask, cell_mask = _single_cell_masks()
        new_nuc, new_cell, nuc_map, cell_map = match_and_renumber(nuc_mask, cell_mask)
        assert nuc_map == {1: 1}
        assert cell_map == {1: 1}
        assert np.any(new_nuc == 1)
        assert np.any(new_cell == 1)

    def test_nucleus_outside_cell_no_match(self) -> None:
        # Nucleus centroid at (2,2), cell far at (15,15)
        nuc_mask = np.zeros((20, 20), dtype=np.uint32)
        nuc_mask[1:4, 1:4] = 1
        cell_mask = np.zeros((20, 20), dtype=np.uint32)
        cell_mask[14:18, 14:18] = 1
        _, _, nuc_map, cell_map = match_and_renumber(nuc_mask, cell_mask)
        assert nuc_map == {}
        assert cell_map == {}

    def test_two_nuclei_same_cell_keeps_first(self) -> None:
        # Both nucleus centroids inside one big cell; first nucleus wins
        cell_mask = np.zeros((20, 20), dtype=np.uint32)
        cell_mask[2:18, 2:18] = 1
        nuc_mask = np.zeros((20, 20), dtype=np.uint32)
        nuc_mask[4:7, 4:7] = 1  # first nucleus
        nuc_mask[11:14, 11:14] = 2  # second nucleus, same cell
        _, _, nuc_map, cell_map = match_and_renumber(nuc_mask, cell_mask)
        assert len(nuc_map) == 1
        assert len(cell_map) == 1

    def test_renumbering_consecutive(self) -> None:
        # Two independent nucleus-cell pairs
        nuc_mask = np.zeros((30, 30), dtype=np.uint32)
        nuc_mask[2:6, 2:6] = 5  # nuc label 5
        nuc_mask[18:22, 18:22] = 9  # nuc label 9
        cell_mask = np.zeros((30, 30), dtype=np.uint32)
        cell_mask[1:8, 1:8] = 3  # cell label 3 (covers nuc 5)
        cell_mask[17:24, 17:24] = 7  # cell label 7 (covers nuc 9)
        new_nuc, new_cell, nuc_map, cell_map = match_and_renumber(nuc_mask, cell_mask)
        assert set(nuc_map.values()) == {1, 2}
        assert set(cell_map.values()) == {1, 2}
        assert np.max(new_nuc) == 2
        assert np.max(new_cell) == 2

    def test_empty_nuc_mask_no_pairs(self) -> None:
        nuc_mask = np.zeros((10, 10), dtype=np.uint32)
        cell_mask = np.zeros((10, 10), dtype=np.uint32)
        cell_mask[2:7, 2:7] = 1
        _, _, nuc_map, cell_map = match_and_renumber(nuc_mask, cell_mask)
        assert nuc_map == {}
        assert cell_map == {}

    def test_many_cells_performance(self) -> None:
        """256-cell FOV must finish in <0.1 s — guards O(n_pixels) LUT path."""
        import time

        n = 256  # grid side
        cell_side = 16  # 16×16 cells → 16×16 grid = 256 cells
        cells_per_side = n // cell_side
        n_cells = cells_per_side * cells_per_side  # 256

        nuc_mask = np.zeros((n, n), dtype=np.uint32)
        cell_mask = np.zeros((n, n), dtype=np.uint32)
        label = 1
        for r in range(cells_per_side):
            for c in range(cells_per_side):
                r0, c0 = r * cell_side, c * cell_side
                # nucleus: inner 4×4 block
                nuc_mask[r0 + 6 : r0 + 10, c0 + 6 : c0 + 10] = label
                cell_mask[r0 + 1 : r0 + 15, c0 + 1 : c0 + 15] = label
                label += 1

        t0 = time.perf_counter()
        new_nuc, new_cell, nuc_map, cell_map = match_and_renumber(nuc_mask, cell_mask)
        elapsed = time.perf_counter() - t0

        assert len(nuc_map) == n_cells
        assert len(cell_map) == n_cells
        assert set(nuc_map.values()) == set(range(1, n_cells + 1))
        assert elapsed < 0.1, f"match_and_renumber: {elapsed:.3f}s for {n_cells} cells (limit 0.1s)"


# ---------------------------------------------------------------------------
# TestComputeFovStats
# ---------------------------------------------------------------------------


class TestComputeFovStats:
    def test_empty_masks_returns_empty_df(self) -> None:
        nuc_mask = np.zeros((10, 10), dtype=np.uint32)
        cell_mask = np.zeros((10, 10), dtype=np.uint32)
        df = compute_fov_stats(nuc_mask, cell_mask, "fov_0", {}, {})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == SEG_COLUMNS

    def test_single_cell_areas(self) -> None:
        nuc_mask = np.zeros((20, 20), dtype=np.uint32)
        nuc_mask[5:8, 5:8] = 1  # 3×3 = 9 pixels
        cell_mask = np.zeros((20, 20), dtype=np.uint32)
        cell_mask[4:10, 4:10] = 1  # 6×6 = 36 pixels
        df = compute_fov_stats(nuc_mask, cell_mask, "fov_1", {1: 1}, {1: 1})
        assert len(df) == 1
        assert df.iloc[0]["nuc_area"] == 9
        assert df.iloc[0]["cyto_area"] == 36
        assert abs(df.iloc[0]["nuc_cyto_ratio"] - 9 / 36) < 1e-9

    def test_columns_match_seg_columns(self) -> None:
        nuc_mask = np.zeros((10, 10), dtype=np.uint32)
        nuc_mask[2:5, 2:5] = 1
        cell_mask = np.zeros((10, 10), dtype=np.uint32)
        cell_mask[1:7, 1:7] = 1
        df = compute_fov_stats(nuc_mask, cell_mask, "fov_x", {1: 1}, {1: 1})
        assert list(df.columns) == SEG_COLUMNS

    def test_fov_id_preserved(self) -> None:
        nuc_mask = np.zeros((10, 10), dtype=np.uint32)
        nuc_mask[2:5, 2:5] = 1
        cell_mask = np.zeros((10, 10), dtype=np.uint32)
        cell_mask[1:7, 1:7] = 1
        df = compute_fov_stats(nuc_mask, cell_mask, "my_fov_id", {1: 1}, {1: 1})
        assert df.iloc[0]["fov_id"] == "my_fov_id"

    def test_n_cells_in_fov(self) -> None:
        nuc_mask = np.zeros((30, 10), dtype=np.uint32)
        nuc_mask[2:5, 2:5] = 1
        nuc_mask[20:23, 2:5] = 2
        cell_mask = np.zeros((30, 10), dtype=np.uint32)
        cell_mask[1:7, 1:7] = 1
        cell_mask[19:25, 1:7] = 2
        df = compute_fov_stats(nuc_mask, cell_mask, "fov_2", {1: 1, 2: 2}, {1: 1, 2: 2})
        assert (df["n_cells_in_fov"] == 2).all()

    def test_cyto_zero_area_ratio_is_zero(self) -> None:
        nuc_mask = np.zeros((10, 10), dtype=np.uint32)
        nuc_mask[2:5, 2:5] = 1
        cell_mask = np.zeros((10, 10), dtype=np.uint32)
        # cell_mask has no label 1 pixels → cyto_area = 0
        df = compute_fov_stats(nuc_mask, cell_mask, "fov_zero", {1: 1}, {1: 1})
        assert df.iloc[0]["nuc_cyto_ratio"] == 0.0


# ---------------------------------------------------------------------------
# TestSegmentPlate (mocked Cellpose)
# ---------------------------------------------------------------------------


class TestSegmentPlate:
    def _write_input_store(
        self, tmp_path: Path, source: str, batch: str, plate: str, n_fovs: int = 2
    ) -> dict[str, np.ndarray]:
        store_path = plate_store_path(tmp_path, source, batch, plate)
        group = open_plate_group(store_path)
        stacks: dict[str, np.ndarray] = {}
        for i in range(n_fovs):
            fov_id = f"fov_{i:03d}"
            stack = _make_fov_stack()
            write_fov_array(group, fov_id, stack)
            stacks[fov_id] = stack
        return stacks

    def test_basic_writes_zarr_and_parquet(self, tmp_path: Path) -> None:
        from cp_bg_bench.segmentation.cpsam import segment_plate

        source, batch, plate = "src", "b1", "p1"
        self._write_input_store(tmp_path, source, batch, plate, n_fovs=2)

        nuc_mask, cell_mask = _single_cell_masks()
        mock_model = _mock_model(nuc_mask, cell_mask)

        with patch("cp_bg_bench.segmentation.cpsam._make_cellpose_model", return_value=mock_model):
            df = segment_plate(
                source=source,
                batch=batch,
                plate=plate,
                output_root=tmp_path,
                seg_config=_make_seg_config(),
            )

        # One cell per FOV × 2 FOVs
        assert len(df) == 2
        assert list(df.columns) == SEG_COLUMNS

        seg_store = tmp_path / "segmentation" / f"{source}__{batch}__{plate}.zarr"
        assert seg_store.exists()
        seg_group = zarr.open_group(str(seg_store), mode="r")
        for fov_id in ("fov_000", "fov_001"):
            arr = seg_group[fov_id]
            assert arr.shape == (2, 20, 20)
            assert arr.dtype == np.uint32

    def test_empty_fov_produces_zero_cells(self, tmp_path: Path) -> None:
        from cp_bg_bench.segmentation.cpsam import segment_plate

        source, batch, plate = "src", "b1", "p1"
        self._write_input_store(tmp_path, source, batch, plate, n_fovs=1)

        empty_mask = np.zeros((20, 20), dtype=np.uint32)
        mock_model = _mock_model(empty_mask, empty_mask)

        with patch("cp_bg_bench.segmentation.cpsam._make_cellpose_model", return_value=mock_model):
            df = segment_plate(
                source=source,
                batch=batch,
                plate=plate,
                output_root=tmp_path,
                seg_config=_make_seg_config(),
            )

        assert len(df) == 0
        assert list(df.columns) == SEG_COLUMNS

    def test_per_source_diameters_used(self, tmp_path: Path) -> None:
        from cp_bg_bench.segmentation.cpsam import _get_diameter

        override = MagicMock()
        override.nucleus = 30.0
        override.cytosol = 75.0
        cfg = _make_seg_config(source_overrides={"my_source": override})

        nuc_d, cell_d = _get_diameter(cfg, "my_source")
        assert nuc_d == 30.0
        assert cell_d == 75.0

    def test_default_diameters_fallback(self, tmp_path: Path) -> None:
        from cp_bg_bench.segmentation.cpsam import _get_diameter

        cfg = _make_seg_config(nucleus_diam=18.0, cell_diam=45.0)

        nuc_d, cell_d = _get_diameter(cfg, "unknown_source")
        assert nuc_d == 18.0
        assert cell_d == 45.0

    def test_label_image_dtype_and_channel_count(self, tmp_path: Path) -> None:
        from cp_bg_bench.segmentation.cpsam import segment_plate

        source, batch, plate = "src", "b1", "p1"
        self._write_input_store(tmp_path, source, batch, plate, n_fovs=1)

        nuc_mask, cell_mask = _single_cell_masks()
        mock_model = _mock_model(nuc_mask, cell_mask)

        with patch("cp_bg_bench.segmentation.cpsam._make_cellpose_model", return_value=mock_model):
            segment_plate(
                source=source,
                batch=batch,
                plate=plate,
                output_root=tmp_path,
                seg_config=_make_seg_config(),
            )

        seg_store = tmp_path / "segmentation" / f"{source}__{batch}__{plate}.zarr"
        seg_group = zarr.open_group(str(seg_store), mode="r")
        arr = seg_group["fov_000"]
        assert arr.shape[0] == 2, "label image must have 2 channels (nuc + cell)"
        assert arr.dtype == np.uint32


# ---------------------------------------------------------------------------
# Module importability without cellpose
# ---------------------------------------------------------------------------


def test_segmentation_module_imports_without_cellpose() -> None:
    """postprocess is pure numpy; cpsam uses lazy imports — both must be
    importable in the default env (no cellpose installed)."""
    import importlib

    assert importlib.util.find_spec("cp_bg_bench.segmentation.postprocess") is not None
    assert importlib.util.find_spec("cp_bg_bench.segmentation.cpsam") is not None
