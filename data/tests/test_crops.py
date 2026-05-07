"""Tests for cp_bg_bench.crops (rule F): normalize + extract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cp_bg_bench.crops.normalize import naive_u16_to_u8, per_fov_percentile_norm

# ── TestNormalize ─────────────────────────────────────────────────────────────


class TestPerFovPercentileNorm:
    def test_output_dtype_is_uint8(self) -> None:
        stack = np.random.default_rng(0).integers(0, 65535, (5, 20, 20), dtype=np.uint16)
        result = per_fov_percentile_norm(stack)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self) -> None:
        stack = np.zeros((3, 10, 10), dtype=np.uint16)
        result = per_fov_percentile_norm(stack)
        assert result.shape == stack.shape

    def test_values_in_0_255(self) -> None:
        rng = np.random.default_rng(42)
        stack = rng.integers(0, 65535, (5, 32, 32), dtype=np.uint16)
        result = per_fov_percentile_norm(stack, low=0.01, high=0.99)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_channel_maps_to_zero(self) -> None:
        stack = np.full((2, 8, 8), 1000, dtype=np.uint16)
        result = per_fov_percentile_norm(stack)
        assert (result == 0).all()

    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="3-D"):
            per_fov_percentile_norm(np.zeros((5, 8), dtype=np.uint16))

    def test_per_channel_independent(self) -> None:
        stack = np.zeros((2, 8, 8), dtype=np.uint16)
        stack[0] = 100  # constant → all 0
        stack[1] = np.arange(64).reshape(8, 8)  # varying → non-trivial
        result = per_fov_percentile_norm(stack)
        assert (result[0] == 0).all()
        assert result[1].max() > 0


class TestNaiveU16ToU8:
    def test_output_dtype_is_uint8(self) -> None:
        stack = np.full((3, 4, 4), 257, dtype=np.uint16)
        result = naive_u16_to_u8(stack)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 1  # 257 // 257 = 1

    def test_max_value_maps_near_255(self) -> None:
        stack = np.full((1, 4, 4), 65535, dtype=np.uint16)
        result = naive_u16_to_u8(stack)
        assert int(result[0, 0, 0]) == 255

    def test_zero_maps_to_zero(self) -> None:
        stack = np.zeros((1, 4, 4), dtype=np.uint16)
        result = naive_u16_to_u8(stack)
        assert (result == 0).all()


# ── TestExtractCrop ───────────────────────────────────────────────────────────


class TestExtractCrop:
    """Test via the internal helper indirectly through extract_plate_crops."""

    def _make_zarr_stores(self, tmp_path: Path, plate_key: str = "src__bat__plt"):

        from cp_bg_bench.io.zarr_io import (
            open_plate_group,
            plate_store_path,
            seg_store_path,
            write_fov_array,
            write_label_array,
        )

        source, batch, plate = plate_key.split("__")
        fov_id = f"{plate_key}__A01__1"

        # Write full image (5, 50, 50) uint16
        full_grp = open_plate_group(plate_store_path(tmp_path, source, batch, plate))
        img = np.random.default_rng(0).integers(0, 65535, (5, 50, 50), dtype=np.uint16)
        write_fov_array(full_grp, fov_id, img)

        # Write label image (2, 50, 50) uint32 — one cell at id_local=1
        label = np.zeros((2, 50, 50), dtype=np.uint32)
        label[0, 20:28, 20:28] = 1  # nucleus
        label[1, 18:30, 18:30] = 1  # cell
        seg_grp = open_plate_group(seg_store_path(tmp_path, source, batch, plate))
        write_label_array(seg_grp, fov_id, label)

        return fov_id, img, label

    def test_extract_returns_pyarrow_table(self, tmp_path: Path) -> None:
        import pandas as pd
        import pyarrow as pa

        from cp_bg_bench.crops.extract import extract_plate_crops

        plate_key = "src__bat__plt"
        fov_id, img, label = self._make_zarr_stores(tmp_path, plate_key)

        sel_df = pd.DataFrame(
            [
                {
                    "fov_id": fov_id,
                    "id_local": 1,
                    "cyto_cent_row": 24,
                    "cyto_cent_col": 24,
                    "nuc_area": 64,
                    "cyto_area": 144,
                    "nuc_cyto_ratio": 64 / 144,
                    "n_cells_in_fov": 1,
                    "n_cells_scaled": 128.0,
                    "Metadata_JCP2022": "JCP",
                    "Metadata_InChIKey": "IK",
                    "Metadata_InChI": "IC",
                    "Metadata_PlateType": "COMPOUND",
                    "perturbation": "IK",
                    "batch": "bat",
                    "treatment": "",
                }
            ]
        )

        table = extract_plate_crops(plate_key, sel_df, tmp_path, patch_size=30)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1

    def test_crop_has_correct_channels(self, tmp_path: Path) -> None:
        import pandas as pd

        from cp_bg_bench.crops.extract import N_MASK_CHANNELS, extract_plate_crops

        N_IMG_CHANNELS = 5  # test image has 5 fluorescence channels

        plate_key = "src__bat__plt2"
        fov_id, _, _ = self._make_zarr_stores(tmp_path, plate_key)

        sel_df = pd.DataFrame(
            [
                {
                    "fov_id": fov_id,
                    "id_local": 1,
                    "cyto_cent_row": 24,
                    "cyto_cent_col": 24,
                    "nuc_area": 64,
                    "cyto_area": 144,
                    "nuc_cyto_ratio": 0.44,
                    "n_cells_in_fov": 1,
                    "n_cells_scaled": 128.0,
                    "Metadata_JCP2022": "",
                    "Metadata_InChIKey": "",
                    "Metadata_InChI": "",
                    "Metadata_PlateType": "",
                    "perturbation": "gene_A",
                    "batch": "plate_p",
                    "treatment": "",
                }
            ]
        )

        table = extract_plate_crops(plate_key, sel_df, tmp_path, patch_size=30)

        # cell: fluorescence channels only
        cell_bytes = table["cell"][0].as_py()
        cell_arr = np.frombuffer(cell_bytes, dtype=np.uint8).reshape(N_IMG_CHANNELS, 30, 30)
        assert cell_arr.shape == (N_IMG_CHANNELS, 30, 30)

        # mask: segmentation mask channels
        mask_bytes = table["mask"][0].as_py()
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(N_MASK_CHANNELS, 30, 30)
        assert mask_arr.shape == (N_MASK_CHANNELS, 30, 30)

    def test_empty_selection_returns_empty_table(self, tmp_path: Path) -> None:
        import pandas as pd

        from cp_bg_bench.crops.extract import CROPS_SCHEMA, extract_plate_crops

        plate_key = "src__bat__plt3"
        self._make_zarr_stores(tmp_path, plate_key)
        empty_df = pd.DataFrame(
            columns=[
                "fov_id",
                "id_local",
                "cyto_cent_row",
                "cyto_cent_col",
                "nuc_area",
                "cyto_area",
                "nuc_cyto_ratio",
                "n_cells_in_fov",
                "n_cells_scaled",
                "Metadata_JCP2022",
                "Metadata_InChIKey",
                "Metadata_InChI",
                "Metadata_PlateType",
                "perturbation",
                "batch",
                "treatment",
            ]
        )
        table = extract_plate_crops(plate_key, empty_df, tmp_path, patch_size=30)
        assert table.num_rows == 0
        assert table.schema.equals(CROPS_SCHEMA)

    def test_near_border_cell_excluded(self, tmp_path: Path) -> None:
        """Cells whose centroid is within patch_size//2 of any border are dropped."""
        import pandas as pd

        from cp_bg_bench.crops.extract import extract_plate_crops

        plate_key = "src__bat__plt5"
        fov_id, _, _ = self._make_zarr_stores(tmp_path, plate_key)  # 50×50 image

        # patch_size=30 → half=15; centroid at row=5 → 5 < 15 → out of bounds
        near_border_df = pd.DataFrame(
            [
                {
                    "fov_id": fov_id,
                    "id_local": 1,
                    "cyto_cent_row": 5,
                    "cyto_cent_col": 24,
                    "nuc_area": 64,
                    "cyto_area": 144,
                    "nuc_cyto_ratio": 0.44,
                    "n_cells_in_fov": 1,
                    "n_cells_scaled": 128.0,
                    "Metadata_JCP2022": "",
                    "Metadata_InChIKey": "",
                    "Metadata_InChI": "",
                    "Metadata_PlateType": "",
                    "perturbation": "gene_A",
                    "batch": "plate_p",
                    "treatment": "",
                }
            ]
        )
        table = extract_plate_crops(plate_key, near_border_df, tmp_path, patch_size=30)
        assert table.num_rows == 0

    def test_in_bounds_cell_not_excluded(self, tmp_path: Path) -> None:
        """Cells whose centroid is exactly at patch_size//2 from the border are kept."""
        import pandas as pd

        from cp_bg_bench.crops.extract import extract_plate_crops

        plate_key = "src__bat__plt6"
        fov_id, _, _ = self._make_zarr_stores(tmp_path, plate_key)  # 50×50 image

        # half=15; centroid at row=15 → exactly on the boundary → kept
        boundary_df = pd.DataFrame(
            [
                {
                    "fov_id": fov_id,
                    "id_local": 1,
                    "cyto_cent_row": 15,
                    "cyto_cent_col": 24,
                    "nuc_area": 64,
                    "cyto_area": 144,
                    "nuc_cyto_ratio": 0.44,
                    "n_cells_in_fov": 1,
                    "n_cells_scaled": 128.0,
                    "Metadata_JCP2022": "",
                    "Metadata_InChIKey": "",
                    "Metadata_InChI": "",
                    "Metadata_PlateType": "",
                    "perturbation": "gene_A",
                    "batch": "plate_p",
                    "treatment": "",
                }
            ]
        )
        table = extract_plate_crops(plate_key, boundary_df, tmp_path, patch_size=30)
        assert table.num_rows == 1

    def test_row_key_format(self, tmp_path: Path) -> None:
        import pandas as pd

        from cp_bg_bench.crops.extract import extract_plate_crops

        plate_key = "src__bat__plt4"
        fov_id, _, _ = self._make_zarr_stores(tmp_path, plate_key)

        sel_df = pd.DataFrame(
            [
                {
                    "fov_id": fov_id,
                    "id_local": 3,
                    "cyto_cent_row": 24,
                    "cyto_cent_col": 24,
                    "nuc_area": 64,
                    "cyto_area": 144,
                    "nuc_cyto_ratio": 0.44,
                    "n_cells_in_fov": 1,
                    "n_cells_scaled": 0.0,
                    "Metadata_JCP2022": "",
                    "Metadata_InChIKey": "",
                    "Metadata_InChI": "",
                    "Metadata_PlateType": "",
                    "perturbation": "gene_A",
                    "batch": "plate_p",
                    "treatment": "",
                }
            ]
        )
        table = extract_plate_crops(plate_key, sel_df, tmp_path, patch_size=30)
        row_key = table["row_key"][0].as_py()
        assert row_key == f"{fov_id}__3"
