"""Tests for :mod:`cp_bg_bench.io.rxrx3_core`."""

from __future__ import annotations

import io as _io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from cp_bg_bench.config import Rxrx3CoreConfig, Rxrx3CoreSample
from cp_bg_bench.io.rxrx3_core import (
    _RXRX3_CORE_BATCH,
    _SCALAR_OUTPUT_COLUMNS,
    _build_fov_table,
    _select_samples,
    assign_snakemake_batches,
    resolve_metadata,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

_MINIMAL_CFG = Rxrx3CoreConfig(
    samples=[Rxrx3CoreSample(experiment="gene-001")],
    hf_repo="recursionpharma/rxrx3-core",
    channel_names=["Hoechst", "ConA"],
    segmentation={
        "model": "cpsam",
        "channels_for_nucleus": [0],
        "channels_for_cell": [2, 0],
        "default_diameters": {"nucleus": 21, "cytosol": 53},
    },
)

# Minimal metadata CSV: CRISPR + one COMPOUND row (must be excluded).
_SAMPLE_CSV = """\
well_id,experiment_name,plate,address,gene,treatment,SMILES,concentration,perturbation_type,cell_type
w1,gene-001,1,A01,TP53,TP53_guide_1,,1.0,CRISPR,HUVEC
w2,gene-001,1,A02,EGFR,EGFR_guide_2,,1.0,CRISPR,HUVEC
w3,gene-001,1,A03,EMPTY_control,EMPTY_control,,0.0,CRISPR,HUVEC
w4,gene-001,2,B01,TP53,TP53_guide_1,,1.0,CRISPR,HUVEC
w5,gene-002,1,A01,BRCA1,BRCA1_guide_1,,1.0,CRISPR,HUVEC
w6,gene-001,1,C01,MDM2,MDM2_guide_1,,1.0,COMPOUND,HUVEC
"""


def _make_raw() -> pd.DataFrame:
    return pd.read_csv(_io.StringIO(_SAMPLE_CSV))


def _make_shard_index() -> dict[str, int]:
    """Minimal shard index covering all CRISPR rows in _SAMPLE_CSV."""
    return {
        "gene-001/Plate1/A01_s1": 15,
        "gene-001/Plate1/A02_s1": 15,
        "gene-001/Plate1/A03_s1": 15,
        "gene-001/Plate2/B01_s1": 16,
        "gene-002/Plate1/A01_s1": 17,
    }


# ── _select_samples ───────────────────────────────────────────────────────────


def test_select_samples_excludes_compound_rows() -> None:
    """COMPOUND perturbation_type rows must be filtered out."""
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx3CoreSample(experiment="gene-001")])
    assert (result["perturbation_type"] == "CRISPR").all()
    assert "MDM2" not in result["gene"].values


def test_select_samples_by_experiment() -> None:
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx3CoreSample(experiment="gene-001")])
    assert set(result["experiment_name"].unique()) == {"gene-001"}
    assert len(result) == 4  # A01, A02, A03, B01 (all CRISPR, C01 COMPOUND excluded)


def test_select_samples_by_plate() -> None:
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx3CoreSample(experiment="gene-001", plate=1)])
    assert set(result["plate"].unique()) == {1}
    assert len(result) == 3  # A01, A02, A03 on plate 1


def test_select_samples_by_gene() -> None:
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx3CoreSample(experiment="gene-001", gene="TP53")])
    assert set(result["gene"].unique()) == {"TP53"}


def test_select_samples_by_address() -> None:
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx3CoreSample(experiment="gene-001", address="A01")])
    assert set(result["address"].unique()) == {"A01"}
    assert len(result) == 1


def test_select_samples_union_across_entries() -> None:
    raw = _make_raw()
    result = _select_samples(
        raw,
        [Rxrx3CoreSample(experiment="gene-001"), Rxrx3CoreSample(experiment="gene-002")],
    )
    assert set(result["experiment_name"].unique()) == {"gene-001", "gene-002"}


def test_select_samples_no_match_raises() -> None:
    raw = _make_raw()
    with pytest.raises(ValueError, match="empty selection"):
        _select_samples(raw, [Rxrx3CoreSample(experiment="gene-999")])


def test_select_samples_deduplicates() -> None:
    raw = _make_raw()
    result = _select_samples(
        raw,
        [Rxrx3CoreSample(experiment="gene-001"), Rxrx3CoreSample(experiment="gene-001")],
    )
    assert not result.duplicated(subset=["experiment_name", "plate", "address"]).any()


# ── _build_fov_table ──────────────────────────────────────────────────────────


def _crispr_rows() -> pd.DataFrame:
    raw = _make_raw()
    return _select_samples(raw, [Rxrx3CoreSample(experiment="gene-001")])


def test_build_fov_table_column_rename() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    for col in ("Metadata_Source", "Metadata_Plate", "Metadata_InChIKey", "Metadata_InChI"):
        assert col in fov.columns


def test_build_fov_table_batch_is_fixed() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    assert (fov["Metadata_Batch"] == _RXRX3_CORE_BATCH).all()


def test_build_fov_table_site_is_one() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    # Metadata_Site is not in the scalar output columns but must exist internally.
    # Check that the id always ends with __1.
    assert fov["id"].str.endswith("__1").all()


def test_build_fov_table_id_format() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    assert fov["id"].str.contains("__rxrx3_core__").all()
    assert (fov["id"].str.count("__") == 4).all()


def test_build_fov_table_parquet_key_prefix() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    # Prefix must be "{experiment}/Plate{plate}/{address}_s1"
    assert fov["parquet_key_prefix"].str.startswith("gene-001/Plate").all()
    assert fov["parquet_key_prefix"].str.endswith("_s1").all()


def test_build_fov_table_shard_assigned_from_index() -> None:
    idx = _make_shard_index()
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, idx)
    for _, row in fov.iterrows():
        assert row["parquet_shard"] == idx[row["parquet_key_prefix"]]


def test_build_fov_table_missing_shard_raises() -> None:
    """A key prefix absent from the shard index raises ValueError."""
    fov_rows = _crispr_rows()
    with pytest.raises(ValueError, match="not found in any parquet shard"):
        _build_fov_table(fov_rows, _MINIMAL_CFG, {})


# ── assign_snakemake_batches ──────────────────────────────────────────────────


def test_assign_snakemake_batches_column_added() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    bucketed = assign_snakemake_batches(fov, batch_size=2)
    assert "snakemake_batch" in bucketed.columns


def test_assign_snakemake_batches_uses_source_prefix() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    bucketed = assign_snakemake_batches(fov, batch_size=10)
    assert bucketed["snakemake_batch"].str.startswith("gene-001").all()


def test_assign_snakemake_batches_resets_per_source() -> None:
    raw = _make_raw()
    selected = _select_samples(
        raw,
        [Rxrx3CoreSample(experiment="gene-001"), Rxrx3CoreSample(experiment="gene-002")],
    )
    # Need shard index for both experiments.
    idx = _make_shard_index()
    fov = _build_fov_table(selected, _MINIMAL_CFG, idx)
    bucketed = assign_snakemake_batches(fov, batch_size=10)
    batch_ids = set(bucketed["snakemake_batch"])
    assert "gene-001_snakemake_batch_0" in batch_ids
    assert "gene-002_snakemake_batch_0" in batch_ids


def test_assign_snakemake_batches_invalid_size_raises() -> None:
    fov = _build_fov_table(_crispr_rows(), _MINIMAL_CFG, _make_shard_index())
    with pytest.raises(ValueError, match="batch_size must be positive"):
        assign_snakemake_batches(fov, batch_size=0)


# ── resolve_metadata ──────────────────────────────────────────────────────────


def test_resolve_metadata_output_columns(tmp_path: Path) -> None:
    raw = _make_raw()
    idx = _make_shard_index()

    with (
        patch("cp_bg_bench.io.rxrx3_core._download_metadata_csv", return_value=raw),
        patch("cp_bg_bench.io.rxrx3_core._build_shard_index", return_value=idx),
    ):
        df = resolve_metadata(_MINIMAL_CFG, cache_dir=tmp_path, batch_size=10)

    assert list(df.columns) == list(_SCALAR_OUTPUT_COLUMNS)


def test_resolve_metadata_row_count_matches_selection(tmp_path: Path) -> None:
    raw = _make_raw()
    idx = _make_shard_index()

    with (
        patch("cp_bg_bench.io.rxrx3_core._download_metadata_csv", return_value=raw),
        patch("cp_bg_bench.io.rxrx3_core._build_shard_index", return_value=idx),
    ):
        df = resolve_metadata(_MINIMAL_CFG, cache_dir=tmp_path, batch_size=10)

    # gene-001: 4 CRISPR rows (A01, A02, A03 on plate 1 + B01 on plate 2);
    # gene-002 excluded by config; COMPOUND row excluded by perturbation_type filter.
    assert len(df) == 4


def test_resolve_metadata_no_duplicate_ids(tmp_path: Path) -> None:
    raw = _make_raw()
    idx = _make_shard_index()

    with (
        patch("cp_bg_bench.io.rxrx3_core._download_metadata_csv", return_value=raw),
        patch("cp_bg_bench.io.rxrx3_core._build_shard_index", return_value=idx),
    ):
        df = resolve_metadata(_MINIMAL_CFG, cache_dir=tmp_path, batch_size=10)

    assert not df["id"].duplicated().any()
