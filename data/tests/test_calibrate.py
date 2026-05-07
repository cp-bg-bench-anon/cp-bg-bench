"""Tests for cp_bg_bench.calibrate.diameters (rule C)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import zarr

from cp_bg_bench.calibrate.diameters import (
    CalibrationQualityError,
    _build_md,
    _build_png,
    _build_yml,
    _estimate_diameter,
    compute_config_hash,
    run_calibration,
    sample_fovs,
)
from cp_bg_bench.io.zarr_io import plate_store_path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_meta_df(
    sources: list[str] | None = None,
    n_per_source: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    if sources is None:
        sources = ["source_2", "source_4"]
    rows = []
    for src in sources:
        for i in range(n_per_source):
            rows.append(
                {
                    "id": f"{src}__batch_1__plate_1__well_A01__site_{i + 1}",
                    "Metadata_Source": src,
                    "Metadata_Batch": "batch_1",
                    "Metadata_Plate": "plate_1",
                    "Metadata_Well": "A01",
                    "Metadata_Site": str(i + 1),
                }
            )
    return pd.DataFrame(rows)


def _make_zarr_store(tmp_path: Path, meta_df: pd.DataFrame, shape: tuple = (5, 64, 64)) -> None:
    """Populate per-plate Zarr v3 stores from a metadata DataFrame."""
    from zarr.codecs import BloscCodec

    codec = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    for (src, batch, plate), group_df in meta_df.groupby(
        ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]
    ):
        store_path = plate_store_path(tmp_path, src, batch, plate)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(store=str(store_path), mode="a", zarr_format=3)
        rng = np.random.default_rng(42)
        for _, row in group_df.iterrows():
            fov_id = row["id"]
            if fov_id not in group:
                arr = group.create_array(
                    name=fov_id,
                    shape=shape,
                    chunks=shape,
                    dtype=np.uint16,
                    compressors=codec,
                    dimension_names=("channel", "y", "x"),
                )
                arr[:] = rng.integers(0, 4096, shape, dtype=np.uint16)


def _make_seg_config(channels_for_nucleus=(0,), channels_for_cell=(3, 0)):
    cfg = MagicMock()
    cfg.model = "cpsam"
    cfg.channels_for_nucleus = list(channels_for_nucleus)
    cfg.channels_for_cell = list(channels_for_cell)
    cfg.default_diameters.nucleus = 21.0
    cfg.default_diameters.cytosol = 53.0
    return cfg


def _make_cal_config(fovs_per_source=3, seed=42, min_ok=0.7):
    cfg = MagicMock()
    cfg.fovs_per_source = fovs_per_source
    cfg.random_seed = seed
    cfg.min_success_fraction = min_ok
    return cfg


def _mock_png(per_source_stats, output_path):  # noqa: ARG001
    """Stub for _build_png that just touches the output file."""
    Path(output_path).touch()


def _mock_model_factory(masks_val: int = 1, diam: float = 22.0):
    """Return a callable that patches _make_cellpose_model."""

    def factory(model_type, use_gpu):  # noqa: ARG001
        model = MagicMock()

        def eval_fn(imgs, diameter, channels):  # noqa: ARG001
            masks = [np.full(imgs[0].shape[:2], masks_val, dtype=np.int32)]
            flows = [None]
            styles = [None]
            diams = [diam]
            return masks, flows, styles, diams

        model.eval.side_effect = eval_fn
        return model

    return factory


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    def test_length(self):
        h = compute_config_hash("smoke", ["a", "b"], [0], [3, 0], 12, 42)
        assert len(h) == 12

    def test_hex_chars(self):
        h = compute_config_hash("smoke", ["a", "b"], [0], [3, 0], 12, 42)
        int(h, 16)  # raises ValueError if not valid hex

    def test_stability(self):
        kwargs = dict(
            data_source_stem="smoke",
            fov_ids=["b", "a"],
            channels_for_nucleus=[0],
            channels_for_cell=[3, 0],
            fovs_per_source=12,
            random_seed=42,
        )
        assert compute_config_hash(**kwargs) == compute_config_hash(**kwargs)

    def test_fov_order_independent(self):
        h1 = compute_config_hash("smoke", ["a", "b"], [0], [3, 0], 12, 42)
        h2 = compute_config_hash("smoke", ["b", "a"], [0], [3, 0], 12, 42)
        assert h1 == h2

    def test_changes_on_seed(self):
        h1 = compute_config_hash("smoke", ["a"], [0], [3, 0], 12, 42)
        h2 = compute_config_hash("smoke", ["a"], [0], [3, 0], 12, 99)
        assert h1 != h2

    def test_changes_on_fov_ids(self):
        h1 = compute_config_hash("smoke", ["a"], [0], [3, 0], 12, 42)
        h2 = compute_config_hash("smoke", ["b"], [0], [3, 0], 12, 42)
        assert h1 != h2

    def test_changes_on_channels(self):
        h1 = compute_config_hash("smoke", ["a"], [0], [3, 0], 12, 42)
        h2 = compute_config_hash("smoke", ["a"], [1], [3, 0], 12, 42)
        assert h1 != h2


# ---------------------------------------------------------------------------
# sample_fovs
# ---------------------------------------------------------------------------


class TestSampleFovs:
    def test_returns_one_df_per_source(self):
        meta = _make_meta_df(["source_2", "source_4"], n_per_source=8)
        result = sample_fovs(meta, fovs_per_source=4, seed=42)
        assert set(result) == {"source_2", "source_4"}

    def test_exact_sample_size(self):
        meta = _make_meta_df(["source_2"], n_per_source=10)
        result = sample_fovs(meta, fovs_per_source=4, seed=42)
        assert len(result["source_2"]) == 4

    def test_deterministic(self):
        meta = _make_meta_df(["source_2"], n_per_source=20)
        r1 = sample_fovs(meta, fovs_per_source=5, seed=7)
        r2 = sample_fovs(meta, fovs_per_source=5, seed=7)
        pd.testing.assert_frame_equal(r1["source_2"], r2["source_2"])

    def test_different_seeds_different_samples(self):
        meta = _make_meta_df(["source_2"], n_per_source=20)
        r1 = sample_fovs(meta, fovs_per_source=5, seed=1)
        r2 = sample_fovs(meta, fovs_per_source=5, seed=2)
        assert not r1["source_2"]["id"].equals(r2["source_2"]["id"])

    def test_fewer_available_uses_all(self):
        meta = _make_meta_df(["source_2"], n_per_source=2)
        result = sample_fovs(meta, fovs_per_source=10, seed=42)
        assert len(result["source_2"]) == 2

    def test_fewer_available_emits_warning(self, caplog):
        import logging

        meta = _make_meta_df(["source_2"], n_per_source=2)
        with caplog.at_level(logging.WARNING, logger="cp_bg_bench.calibrate.diameters"):
            sample_fovs(meta, fovs_per_source=10, seed=42)
        assert "only 2 FOVs" in caplog.text

    def test_missing_columns_raises(self):
        meta = pd.DataFrame({"id": ["x"], "Metadata_Source": ["s"]})
        with pytest.raises(KeyError, match="missing required columns"):
            sample_fovs(meta, fovs_per_source=1, seed=0)

    def test_per_source_seeds_are_independent(self):
        """Adding a second source must not shift the first source's sample."""
        meta_one = _make_meta_df(["source_2"], n_per_source=20)
        meta_two = _make_meta_df(["source_2", "source_4"], n_per_source=20)
        r1 = sample_fovs(meta_one, fovs_per_source=5, seed=42)
        r2 = sample_fovs(meta_two, fovs_per_source=5, seed=42)
        # source_2 sample should be the same in both calls
        assert list(r1["source_2"]["id"]) == list(r2["source_2"]["id"])


# ---------------------------------------------------------------------------
# _estimate_diameter
# ---------------------------------------------------------------------------


class TestEstimateDiameter:
    def _make_model(self, mask_val=1, diam=22.0):
        model = MagicMock()

        def eval_fn(imgs, diameter, channels):  # noqa: ARG001
            return (
                [np.full(imgs[0].shape[:2], mask_val, dtype=np.int32)],
                [None],
                [None],
                [diam],
            )

        model.eval.side_effect = eval_fn
        return model

    def test_returns_diameter_and_cells_found(self):
        model = self._make_model(mask_val=1, diam=25.0)
        diam, ok = _estimate_diameter(model, np.zeros((64, 64), dtype=np.uint16), [0, 0])
        assert diam == pytest.approx(25.0)
        assert ok is True

    def test_no_cells_returns_false(self):
        model = self._make_model(mask_val=0, diam=21.0)
        _, ok = _estimate_diameter(model, np.zeros((64, 64), dtype=np.uint16), [0, 0])
        assert ok is False


# ---------------------------------------------------------------------------
# Failure gate
# ---------------------------------------------------------------------------


class TestFailureGate:
    def test_raises_when_too_many_empty_fovs(self, tmp_path):
        meta = _make_meta_df(["source_2"], n_per_source=10)
        _make_zarr_store(tmp_path, meta)
        seg = _make_seg_config()
        cal = _make_cal_config(fovs_per_source=10, min_ok=0.7)

        # All FOVs return empty masks → 0% success < 70% threshold
        with (
            patch(
                "cp_bg_bench.calibrate.diameters._make_cellpose_model",
                side_effect=_mock_model_factory(masks_val=0, diam=21.0),
            ),
            pytest.raises(CalibrationQualityError, match="source_2"),
        ):
            run_calibration(meta, tmp_path, seg, cal, "smoke", use_gpu=False)

    def test_no_error_when_above_threshold(self, tmp_path):
        meta = _make_meta_df(["source_2"], n_per_source=3)
        _make_zarr_store(tmp_path, meta)
        seg = _make_seg_config()
        cal = _make_cal_config(fovs_per_source=3, min_ok=0.7)

        with (
            patch(
                "cp_bg_bench.calibrate.diameters._make_cellpose_model",
                side_effect=_mock_model_factory(masks_val=1, diam=22.0),
            ),
            patch("cp_bg_bench.calibrate.diameters._build_png", side_effect=_mock_png),
        ):
            config_hash, output_dir = run_calibration(
                meta, tmp_path, seg, cal, "smoke", use_gpu=False
            )
        assert len(config_hash) == 12

    def test_error_message_contains_failed_fov_ids(self, tmp_path):
        meta = _make_meta_df(["source_2"], n_per_source=3)
        _make_zarr_store(tmp_path, meta)
        seg = _make_seg_config()
        cal = _make_cal_config(fovs_per_source=3, min_ok=1.0)  # require 100% → any failure raises

        with (
            patch(
                "cp_bg_bench.calibrate.diameters._make_cellpose_model",
                side_effect=_mock_model_factory(masks_val=0, diam=21.0),
            ),
            pytest.raises(CalibrationQualityError) as exc_info,
        ):
            run_calibration(meta, tmp_path, seg, cal, "smoke", use_gpu=False)
        assert "source_2" in str(exc_info.value)
        assert "Failed FOVs" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


class TestBuildYml:
    def test_round_trips_through_yaml(self):
        import yaml

        per_source = {"source_2": {"nucleus": 21.0, "cytosol": 53.0}}
        text = _build_yml(per_source)
        loaded = yaml.safe_load(text)
        assert loaded["per_source_diameters"] == per_source

    def test_multiple_sources(self):
        import yaml

        per_source = {
            "source_2": {"nucleus": 21.0, "cytosol": 53.0},
            "source_4": {"nucleus": 25.0, "cytosol": 69.0},
        }
        text = _build_yml(per_source)
        loaded = yaml.safe_load(text)
        assert set(loaded["per_source_diameters"]) == {"source_2", "source_4"}


class TestBuildMd:
    def _make_stats(self, source="source_2"):
        return {
            source: {
                "nuc_diams": [20.0, 22.0],
                "cell_diams": [50.0, 55.0],
                "nuc_median": 21.0,
                "cell_median": 52.5,
                "nuc_iqr": 2.0,
                "cell_iqr": 5.0,
                "n_ok": 2,
                "n_total": 2,
            }
        }

    def test_has_per_source_heading(self):
        md = _build_md(self._make_stats(), default_nucleus=21.0, default_cytosol=53.0)
        assert "## source_2" in md

    def test_contains_default_diameters(self):
        md = _build_md(self._make_stats(), default_nucleus=21.0, default_cytosol=53.0)
        assert "21.0" in md
        assert "53.0" in md

    def test_contains_fov_counts(self):
        md = _build_md(self._make_stats(), default_nucleus=21.0, default_cytosol=53.0)
        assert "2" in md


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("matplotlib") is None,
    reason="matplotlib not installed in this env",
)
class TestBuildPng:
    def test_file_created(self, tmp_path):
        stats = {
            "source_2": {
                "nuc_diams": [20.0, 22.0, 21.0],
                "cell_diams": [50.0, 55.0, 52.0],
                "nuc_median": 21.0,
                "cell_median": 52.0,
                "nuc_iqr": 1.0,
                "cell_iqr": 2.5,
                "n_ok": 3,
                "n_total": 3,
            }
        }
        out = tmp_path / "test.png"
        _build_png(stats, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# run_calibration end-to-end (mocked cellpose)
# ---------------------------------------------------------------------------


class TestRunCalibration:
    def test_writes_all_three_report_files(self, tmp_path):
        meta = _make_meta_df(["source_2"], n_per_source=3)
        _make_zarr_store(tmp_path, meta)
        seg = _make_seg_config()
        cal = _make_cal_config(fovs_per_source=3)

        with (
            patch(
                "cp_bg_bench.calibrate.diameters._make_cellpose_model",
                side_effect=_mock_model_factory(masks_val=1, diam=22.0),
            ),
            patch("cp_bg_bench.calibrate.diameters._build_png", side_effect=_mock_png),
        ):
            config_hash, output_dir = run_calibration(
                meta, tmp_path, seg, cal, "smoke", use_gpu=False
            )

        assert (output_dir / f"{config_hash}.yml").exists()
        assert (output_dir / f"{config_hash}.md").exists()
        assert (output_dir / f"{config_hash}.png").exists()

    def test_config_hash_is_stable_across_reruns(self, tmp_path):
        meta = _make_meta_df(["source_2"], n_per_source=3)
        _make_zarr_store(tmp_path, meta)
        seg = _make_seg_config()
        cal = _make_cal_config(fovs_per_source=3, seed=42)

        factory = _mock_model_factory(masks_val=1, diam=22.0)
        with (
            patch("cp_bg_bench.calibrate.diameters._make_cellpose_model", side_effect=factory),
            patch("cp_bg_bench.calibrate.diameters._build_png", side_effect=_mock_png),
        ):
            h1, _ = run_calibration(meta, tmp_path, seg, cal, "smoke", use_gpu=False)
        with (
            patch("cp_bg_bench.calibrate.diameters._make_cellpose_model", side_effect=factory),
            patch("cp_bg_bench.calibrate.diameters._build_png", side_effect=_mock_png),
        ):
            h2, _ = run_calibration(meta, tmp_path, seg, cal, "smoke", use_gpu=False)

        assert h1 == h2

    def test_multichannel_cytosol_stack_shape(self, tmp_path):
        """Ensure (H, W, 2) is passed to cellpose for two-channel cytosol."""
        meta = _make_meta_df(["source_2"], n_per_source=2)
        _make_zarr_store(tmp_path, meta, shape=(5, 32, 32))
        seg = _make_seg_config(channels_for_nucleus=[0], channels_for_cell=[3, 0])
        cal = _make_cal_config(fovs_per_source=2)

        call_shapes: list[tuple] = []

        def factory(model_type, use_gpu):  # noqa: ARG001
            model = MagicMock()

            def eval_fn(imgs, diameter, channels):  # noqa: ARG001
                call_shapes.append(imgs[0].shape)
                return (
                    [np.ones(imgs[0].shape[:2], dtype=np.int32)],
                    [None],
                    [None],
                    [22.0],
                )

            model.eval.side_effect = eval_fn
            return model

        with (
            patch("cp_bg_bench.calibrate.diameters._make_cellpose_model", side_effect=factory),
            patch("cp_bg_bench.calibrate.diameters._build_png", side_effect=_mock_png),
        ):
            run_calibration(meta, tmp_path, seg, cal, "smoke", use_gpu=False)

        # For nucleus calls: (32, 32); for cytosol calls: (32, 32, 2)
        nuc_shapes = [s for s in call_shapes if len(s) == 2]
        cell_shapes = [s for s in call_shapes if len(s) == 3]
        assert all(s == (32, 32) for s in nuc_shapes)
        assert all(s == (32, 32, 2) for s in cell_shapes)

    def test_single_channel_cytosol_stays_2d(self, tmp_path):
        """channels_for_cell=[0] → grayscale (H, W), not (H, W, 1)."""
        meta = _make_meta_df(["source_2"], n_per_source=2)
        _make_zarr_store(tmp_path, meta, shape=(3, 32, 32))
        seg = _make_seg_config(channels_for_nucleus=[0], channels_for_cell=[0])
        cal = _make_cal_config(fovs_per_source=2)

        call_shapes: list[tuple] = []

        def factory(model_type, use_gpu):  # noqa: ARG001
            model = MagicMock()

            def eval_fn(imgs, diameter, channels):  # noqa: ARG001
                call_shapes.append(imgs[0].shape)
                return (
                    [np.ones(imgs[0].shape[:2], dtype=np.int32)],
                    [None],
                    [None],
                    [22.0],
                )

            model.eval.side_effect = eval_fn
            return model

        with (
            patch("cp_bg_bench.calibrate.diameters._make_cellpose_model", side_effect=factory),
            patch("cp_bg_bench.calibrate.diameters._build_png", side_effect=_mock_png),
        ):
            run_calibration(meta, tmp_path, seg, cal, "smoke", use_gpu=False)

        assert all(len(s) == 2 for s in call_shapes)


# ---------------------------------------------------------------------------
# Lazy import: module importable without cellpose
# ---------------------------------------------------------------------------


def test_module_imports_without_cellpose():
    """Importing calibrate.diameters must not require cellpose to be installed."""
    import importlib
    import sys

    # Temporarily shadow cellpose in sys.modules
    orig = sys.modules.get("cellpose")
    sys.modules["cellpose"] = None  # type: ignore[assignment]
    try:
        import cp_bg_bench.calibrate.diameters as mod

        importlib.reload(mod)  # force re-execution of module-level code
    finally:
        if orig is None:
            sys.modules.pop("cellpose", None)
        else:
            sys.modules["cellpose"] = orig


# ---------------------------------------------------------------------------
# CalibrateConfig in the schema
# ---------------------------------------------------------------------------


def test_calibrate_config_defaults():
    from cp_bg_bench.config import CalibrateConfig

    cfg = CalibrateConfig()
    assert cfg.fovs_per_source == 12
    assert cfg.random_seed == 42
    assert cfg.min_success_fraction == pytest.approx(0.7)


def test_pipeline_config_loads_calibrate_block(tmp_path):
    """GlobalConfig picks up the calibrate: block from YAML."""
    from cp_bg_bench.config import load

    # Write minimal smoke config files
    ds_yml = tmp_path / "ds.yml"
    ds_yml.write_text(
        textwrap.dedent("""\
        samples:
          - metadata_source: source_2
        metadata_tables:
          plate: http://example.com/plate.csv.gz
          well: http://example.com/well.csv.gz
          compound: http://example.com/compound.csv.gz
        channel_s3_keys: [s3_OrigDNA]
        segmentation:
          model: cpsam
          channels_for_nucleus: [0]
          channels_for_cell: [0]
          default_diameters:
            nucleus: 21
            cytosol: 53
        """)
    )
    global_yml = tmp_path / "config.yml"
    global_yml.write_text(
        textwrap.dedent("""\
        data_source: jump
        data_source_config: ds.yml
        paths:
          root: /tmp/bench
        selection:
          strategy: uniform_per_well
          cells_per_well: 50
          random_seed: 42
        quality_filter:
          enabled: true
          quantiles: [0.025, 0.975]
          fields: [nuc_area]
        crops:
          patch_size: 150
          image_size: 224
          normalization:
            scheme: per_fov_percentile
            low: 0.001
            high: 0.999
        patching:
          patch_size_resized: 22
          pad_resized: 7
        sharding:
          parquet_rows_per_shard: 10000
          hf_rows_per_shard: 25000
          hf_max_shards: 64
          snakemake_batch_size: 250
        compute:
          dask:
            n_workers: 4
            threads_per_worker: 1
          gpu:
            cellpose_vram_per_sample_mb: 200
            resize_vram_mb_max: 32000
        calibrate:
          fovs_per_source: 5
          random_seed: 99
          min_success_fraction: 0.8
        """)
    )
    cfg = load(global_yml)
    assert cfg.global_.calibrate.fovs_per_source == 5
    assert cfg.global_.calibrate.random_seed == 99
    assert cfg.global_.calibrate.min_success_fraction == pytest.approx(0.8)
