"""Row-key alignment invariant across the four dataset variants (§1 invariant 1).

Asserts that crops, seg, crops_density, and seg_density are row-for-row aligned
on a shared ``row_key`` in identical order — the "datasets truly matched"
guarantee.  Uses the same transform functions called by rules J, K, L so any
logic regression here is also caught.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

# Separate cell (fluorescence) and seg (mask) shapes for unit tests.
_C_IMG, _H, _W = 5, 16, 16
_C_SEG = 2
_CELL_SHAPE = (_C_IMG, _H, _W)
_SEG_SHAPE = (_C_SEG, _H, _W)
_N_ROWS = 20

# Density-patch params that fit inside (16, 16): patch_size=2, pad=1
_PATCH_SIZE = 2
_PAD = 1


def _make_cell_blob(rng: np.random.Generator) -> bytes:
    """Synthetic (5, 16, 16) uint8 fluorescence-only blob."""
    arr = rng.integers(1, 255, size=_CELL_SHAPE, dtype=np.uint8)
    return arr.tobytes(order="C")


def _make_seg_blob(rng: np.random.Generator) -> bytes:
    """Synthetic (2, 16, 16) uint8 mask blob with non-trivial binary masks."""
    arr = np.zeros(_SEG_SHAPE, dtype=np.uint8)
    arr[0] = (rng.integers(0, 255, size=(_H, _W), dtype=np.uint8) > 128).astype(np.uint8) * 255
    arr[1] = (rng.integers(0, 255, size=(_H, _W), dtype=np.uint8) > 100).astype(np.uint8) * 255
    return arr.tobytes(order="C")


def _make_crops_hf(tmp_path):
    """Build a minimal crops HF Dataset with _N_ROWS synthetic rows."""
    from datasets import Dataset

    from cp_bg_bench.datasets.hf import build_hf_features

    rng = np.random.default_rng(0)
    rows = []
    for i in range(_N_ROWS):
        fov_id = f"src__bat__plate__A01__{i // 5 + 1}"
        row_key = f"{fov_id}__{i + 1}"
        rows.append(
            {
                "row_key": row_key,
                "source": "src",
                "batch": "bat",
                "plate": "plate",
                "well": "A01",
                "tile": str(i // 5 + 1),
                "id_local": i + 1,
                "nuc_area": int(rng.integers(50, 200)),
                "cyto_area": int(rng.integers(200, 800)),
                "nuc_cyto_ratio": float(rng.uniform(0.1, 0.5)),
                "n_cells_in_fov": 5,
                "n_cells_scaled": float(rng.uniform(0, 255)),
                "cell": _make_cell_blob(rng),
                "mask": _make_seg_blob(rng),
                "Metadata_JCP2022": "JCP2022_abc",
                "Metadata_InChIKey": "AAAA",
                "Metadata_PlateType": "COMPOUND",
                "perturbation": "gene_A",
                "treatment": "",
            }
        )
    features = build_hf_features()
    return Dataset.from_list(rows, features=features)


def _derive_seg(ds):
    from cp_bg_bench.transforms.masking import _apply_masks_batch as _mask_batch

    def _batch_fn(batch):
        return {"cell": _mask_batch(batch["cell"], batch["mask"], _CELL_SHAPE, _SEG_SHAPE)}

    return ds.map(_batch_fn, batched=True, batch_size=_N_ROWS, num_proc=1, features=ds.features)


def _derive_density(ds):
    from cp_bg_bench.transforms.density_patch import draw_corner_patches_batch

    def _batch_fn(batch):
        return {
            "cell": draw_corner_patches_batch(
                cell_list=batch["cell"],
                intensities=batch["n_cells_scaled"],
                cell_shape=_CELL_SHAPE,
                patch_size=_PATCH_SIZE,
                pad=_PAD,
            )
        }

    return ds.map(_batch_fn, batched=True, batch_size=_N_ROWS, num_proc=1, features=ds.features)


# ── TestAlignment ─────────────────────────────────────────────────────────────


class TestAlignment:
    @pytest.fixture(scope="class")
    def four_variants(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("align")
        crops = _make_crops_hf(tmp)
        seg = _derive_seg(crops)
        crops_density = _derive_density(crops)
        seg_density = _derive_density(seg)
        return {
            "crops": crops,
            "seg": seg,
            "crops_density": crops_density,
            "seg_density": seg_density,
        }

    def test_row_counts_equal(self, four_variants):
        sizes = {k: ds.num_rows for k, ds in four_variants.items()}
        assert len(set(sizes.values())) == 1, f"row counts differ: {sizes}"

    def test_row_keys_identical_across_variants(self, four_variants):
        keys = {k: ds["row_key"] for k, ds in four_variants.items()}
        ref = keys["crops"]
        for name, rk in keys.items():
            assert rk == ref, f"{name} row_key differs from crops"

    def test_row_keys_unique(self, four_variants):
        rk = four_variants["crops"]["row_key"]
        assert len(set(rk)) == len(rk), "row_key values are not unique in crops"

    def test_seg_cell_differs_from_crops(self, four_variants):
        crops_cells = four_variants["crops"]["cell"]
        seg_cells = four_variants["seg"]["cell"]
        # At least one cell must change (masks zero out out-of-mask signal)
        diffs = sum(c != s for c, s in zip(crops_cells, seg_cells, strict=True))
        assert diffs > 0, "derive_seg produced no change — masking may be broken"

    def test_seg_column_unchanged_by_derive_seg(self, four_variants):
        crops_masks = four_variants["crops"]["mask"]
        seg_masks = four_variants["seg"]["mask"]
        assert crops_masks == seg_masks, "mask column must not change across derive_seg"

    def test_crops_density_differs_from_crops(self, four_variants):
        crops_cells = four_variants["crops"]["cell"]
        cd_cells = four_variants["crops_density"]["cell"]
        diffs = sum(c != d for c, d in zip(crops_cells, cd_cells, strict=True))
        assert diffs > 0, "derive_crops_density produced no change — patch drawing may be broken"

    def test_seg_density_differs_from_seg(self, four_variants):
        seg_cells = four_variants["seg"]["cell"]
        sd_cells = four_variants["seg_density"]["cell"]
        diffs = sum(c != d for c, d in zip(seg_cells, sd_cells, strict=True))
        assert diffs > 0, "derive_seg_density produced no change"

    def test_non_cell_columns_identical(self, four_variants):
        # Structural columns must be unchanged across all derived variants
        for col in ("row_key", "source", "batch", "plate", "well", "tile", "id_local"):
            ref = four_variants["crops"][col]
            for name, ds in four_variants.items():
                assert ds[col] == ref, f"column {col!r} differs in {name}"

    def test_schema_preserved_across_variants(self, four_variants):
        ref_features = four_variants["crops"].features
        for name, ds in four_variants.items():
            assert ds.features == ref_features, f"{name} features schema differs from crops"

    def test_derive_chain_order_preserved(self, four_variants):
        """seg_density must equal apply(density, apply(mask, crops)) elementwise."""
        from cp_bg_bench.transforms.density_patch import draw_corner_patches_batch
        from cp_bg_bench.transforms.masking import _apply_masks_batch as _mask_batch

        crops_cells = four_variants["crops"]["cell"]
        crops_masks = four_variants["crops"]["mask"]
        intensities = four_variants["crops"]["n_cells_scaled"]

        masked = _mask_batch(crops_cells, crops_masks, _CELL_SHAPE, _SEG_SHAPE)
        masked_and_patched = draw_corner_patches_batch(
            masked, intensities, _CELL_SHAPE, _PATCH_SIZE, _PAD
        )

        sd_cells = four_variants["seg_density"]["cell"]
        for i, (expected, actual) in enumerate(zip(masked_and_patched, sd_cells, strict=True)):
            assert expected == actual, f"seg_density mismatch at row {i}"
