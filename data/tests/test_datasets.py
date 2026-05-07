"""Tests for cp_bg_bench.datasets: quality_filter and hf helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── TestQualityFilter ─────────────────────────────────────────────────────────


class TestComputeThresholds:
    def test_returns_bounds_per_field(self) -> None:
        from cp_bg_bench.datasets.quality_filter import compute_thresholds

        df = pd.DataFrame({"nuc_area": np.arange(100), "cyto_area": np.arange(100) * 2})
        thresholds = compute_thresholds(df, ["nuc_area", "cyto_area"], (0.1, 0.9))
        assert "nuc_area" in thresholds
        assert "cyto_area" in thresholds
        lo, hi = thresholds["nuc_area"]
        assert lo < hi

    def test_skips_missing_field(self) -> None:
        from cp_bg_bench.datasets.quality_filter import compute_thresholds

        df = pd.DataFrame({"nuc_area": np.arange(10)})
        thresholds = compute_thresholds(df, ["nuc_area", "missing_col"], (0.1, 0.9))
        assert "missing_col" not in thresholds

    def test_bounds_match_quantiles(self) -> None:
        from cp_bg_bench.datasets.quality_filter import compute_thresholds

        vals = np.arange(100, dtype=float)
        df = pd.DataFrame({"x": vals})
        thresholds = compute_thresholds(df, ["x"], (0.1, 0.9))
        expected_lo = float(np.quantile(vals, 0.1))
        expected_hi = float(np.quantile(vals, 0.9))
        assert abs(thresholds["x"][0] - expected_lo) < 1e-9
        assert abs(thresholds["x"][1] - expected_hi) < 1e-9


class TestFilterDataframe:
    def test_drops_out_of_range_rows(self) -> None:
        from cp_bg_bench.datasets.quality_filter import filter_dataframe

        df = pd.DataFrame({"nuc_area": [1, 5, 10, 50, 100]})
        thresholds = {"nuc_area": (5.0, 50.0)}
        result = filter_dataframe(df, thresholds)
        assert (result["nuc_area"] >= 5).all()
        assert (result["nuc_area"] <= 50).all()

    def test_keeps_all_when_no_thresholds(self) -> None:
        from cp_bg_bench.datasets.quality_filter import filter_dataframe

        df = pd.DataFrame({"x": [1, 2, 3]})
        result = filter_dataframe(df, {})
        assert len(result) == 3

    def test_multiple_fields_and_conjunction(self) -> None:
        from cp_bg_bench.datasets.quality_filter import filter_dataframe

        df = pd.DataFrame({"a": [1, 5, 10], "b": [10, 5, 1]})
        thresholds = {"a": (2.0, 8.0), "b": (2.0, 8.0)}
        result = filter_dataframe(df, thresholds)
        # only row with a=5, b=5 survives
        assert len(result) == 1
        assert result.iloc[0]["a"] == 5


# ── TestHFFeatures ────────────────────────────────────────────────────────────


class TestBuildHFFeatures:
    def test_cell_and_mask_are_large_binary(self) -> None:
        from datasets import Value

        from cp_bg_bench.datasets.hf import build_hf_features

        features = build_hf_features()
        assert features["cell"] == Value("large_binary")
        assert features["mask"] == Value("large_binary")

    def test_row_key_is_string(self) -> None:
        from datasets import Value

        from cp_bg_bench.datasets.hf import build_hf_features

        features = build_hf_features()
        assert features["row_key"] == Value("string")

    def test_extra_meta_cols_added(self) -> None:
        from datasets import Value

        from cp_bg_bench.datasets.hf import build_hf_features

        features = build_hf_features(extra_meta_cols=["my_col"])
        assert "my_col" in features
        assert features["my_col"] == Value("string")

    def test_empty_meta_cols(self) -> None:
        from cp_bg_bench.datasets.hf import build_hf_features

        features = build_hf_features(extra_meta_cols=[])
        assert "Metadata_JCP2022" not in features

    def test_metadata_perturbation_not_in_default_features(self) -> None:
        from cp_bg_bench.datasets.hf import build_hf_features

        features = build_hf_features()
        assert "Metadata_Perturbation" not in features


class TestParquetDirToHF:
    def _make_parquet(self, tmp_path, rows: list[dict]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from cp_bg_bench.crops.extract import CROPS_SCHEMA

        table = pa.Table.from_pylist(rows, schema=CROPS_SCHEMA)
        pq.write_table(table, tmp_path / "shard_0.parquet")

    def test_row_transform_adds_column(self, tmp_path) -> None:
        import numpy as np

        from cp_bg_bench.crops.extract import _META_COLS as CROP_META_COLS
        from cp_bg_bench.datasets.hf import _META_COLS, build_hf_features, parquet_dir_to_hf

        rng = np.random.default_rng(0)
        cell_blob = rng.integers(0, 255, 5 * 16 * 16, dtype=np.uint8).tobytes()
        seg_blob = rng.integers(0, 255, 2 * 16 * 16, dtype=np.uint8).tobytes()
        rows = [
            {
                "row_key": "k1",
                "source": "s",
                "batch": "b",
                "plate": "p",
                "well": "A01",
                "tile": "1",
                "id_local": 1,
                "nuc_area": 100,
                "cyto_area": 400,
                "nuc_cyto_ratio": 0.25,
                "n_cells_in_fov": 1,
                "n_cells_scaled": 128.0,
                "cell": cell_blob,
                "seg": seg_blob,
                **{c: "" for c in CROP_META_COLS},
            }
        ]
        (tmp_path / "parquets").mkdir()
        self._make_parquet(tmp_path / "parquets", rows)

        def _add_tag(df):
            df["Metadata_Perturbation"] = "test_value"
            return df

        features = build_hf_features(extra_meta_cols=[*_META_COLS, "Metadata_Perturbation"])
        parquet_dir_to_hf(
            parquet_dir=tmp_path / "parquets",
            output_hf_dir=tmp_path / "hf_out",
            features=features,
            row_transform=_add_tag,
        )
        from datasets import load_from_disk

        ds = load_from_disk(str(tmp_path / "hf_out"))
        assert "Metadata_Perturbation" in ds.features
        assert ds[0]["Metadata_Perturbation"] == "test_value"
