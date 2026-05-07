"""Tests for cp_bg_bench.transforms (masking, density_patch, resize)."""

from __future__ import annotations

import numpy as np
import pytest

from cp_bg_bench.transforms.density_patch import draw_corner_patches, draw_corner_patches_batch
from cp_bg_bench.transforms.masking import apply_masks


def _has_scipy_or_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import scipy  # noqa: F401

        return True
    except ImportError:
        return False


# ── TestMasking ───────────────────────────────────────────────────────────────


class TestApplyMasks:
    def _make_cell(self, h: int = 10, w: int = 10) -> np.ndarray:
        """Fluorescence-only (5, H, W) array; all channels non-zero."""
        return np.ones((5, h, w), dtype=np.uint8) * 100

    def _make_seg(self, h: int = 10, w: int = 10) -> np.ndarray:
        """(2, H, W) mask array: NucMask (3:7, 3:7), CellMask (2:8, 2:8)."""
        seg = np.zeros((2, h, w), dtype=np.uint8)
        seg[0, 3:7, 3:7] = 255  # nucleus region
        seg[1, 2:8, 2:8] = 255  # cell region
        return seg

    def test_dna_zeroed_outside_nucleus(self) -> None:
        cell = self._make_cell()
        seg = self._make_seg()
        out = apply_masks(cell, seg)
        nuc_bool = seg[0] != 0
        assert (out[0][~nuc_bool] == 0).all()

    def test_agp_zeroed_outside_cell(self) -> None:
        cell = self._make_cell()
        seg = self._make_seg()
        out = apply_masks(cell, seg)
        cell_bool = seg[1] != 0
        assert (out[1][~cell_bool] == 0).all()

    def test_returns_copy_not_inplace(self) -> None:
        cell = self._make_cell()
        original_ch0 = cell[0].copy()
        _ = apply_masks(cell, self._make_seg())
        np.testing.assert_array_equal(cell[0], original_ch0)

    def test_rejects_non_3d_cell(self) -> None:
        with pytest.raises(ValueError):
            apply_masks(np.zeros((5, 10), dtype=np.uint8), self._make_seg())

    def test_rejects_wrong_seg_channels(self) -> None:
        with pytest.raises(ValueError):
            apply_masks(self._make_cell(), np.zeros((3, 10, 10), dtype=np.uint8))


# ── TestDensityPatch ──────────────────────────────────────────────────────────


class TestDrawCornerPatches:
    def _make_arr(self, h: int = 50, w: int = 50) -> np.ndarray:
        return np.zeros((7, h, w), dtype=np.uint8)

    def test_four_corners_set(self) -> None:
        arr = self._make_arr()
        patch_size, pad = 5, 3
        out = draw_corner_patches(arr, intensity=128, patch_size=patch_size, pad=pad)
        # All four corners should have value 128
        assert out[0, pad, pad] == 128
        assert out[0, pad, 50 - pad - patch_size] == 128
        assert out[0, 50 - pad - patch_size, pad] == 128
        assert out[0, 50 - pad - patch_size, 50 - pad - patch_size] == 128

    def test_centre_unchanged(self) -> None:
        arr = self._make_arr()
        out = draw_corner_patches(arr, intensity=200, patch_size=5, pad=3)
        centre = out[:, 25, 25]
        assert (centre == 0).all()

    def test_intensity_clipped_to_uint8(self) -> None:
        arr = self._make_arr()
        out = draw_corner_patches(arr, intensity=300, patch_size=5, pad=3)
        assert out[0, 3, 3] == 255

    def test_returns_copy(self) -> None:
        arr = self._make_arr()
        draw_corner_patches(arr, intensity=128, patch_size=5, pad=3)
        assert (arr == 0).all()

    def test_patch_too_large_raises(self) -> None:
        with pytest.raises(ValueError):
            draw_corner_patches(np.zeros((7, 10, 10), dtype=np.uint8), 128, patch_size=8, pad=5)


class TestDrawCornerPatchesBatch:
    def test_batch_applies_to_all(self) -> None:
        cell_shape = (7, 50, 50)
        cells = [np.zeros(cell_shape, dtype=np.uint8).tobytes() for _ in range(3)]
        intensities = [100.0, 200.0, 50.0]
        result = draw_corner_patches_batch(cells, intensities, cell_shape, patch_size=5, pad=3)
        assert len(result) == 3
        for blob, expected_i in zip(result, [100, 200, 50], strict=False):
            arr = np.frombuffer(blob, dtype=np.uint8).reshape(cell_shape)
            assert arr[0, 3, 3] == np.uint8(np.clip(expected_i, 0, 255))


# ── TestResize ────────────────────────────────────────────────────────────────


class TestResize:
    @pytest.mark.skipif(not _has_scipy_or_cupy(), reason="requires scipy or cupy for resize")
    def test_output_shape(self) -> None:
        from cp_bg_bench.transforms.resize import resize_cell

        arr = np.random.default_rng(0).integers(0, 255, (7, 30, 30), dtype=np.uint8)
        out = resize_cell(arr, out_hw=(60, 60))
        assert out.shape == (7, 60, 60)
        assert out.dtype == np.uint8

    @pytest.mark.skipif(not _has_scipy_or_cupy(), reason="requires scipy or cupy for resize")
    def test_noop_when_same_size(self) -> None:
        from cp_bg_bench.transforms.resize import resize_cell

        arr = np.random.default_rng(0).integers(0, 255, (7, 30, 30), dtype=np.uint8)
        out = resize_cell(arr, out_hw=(30, 30))
        np.testing.assert_array_equal(out, arr)
