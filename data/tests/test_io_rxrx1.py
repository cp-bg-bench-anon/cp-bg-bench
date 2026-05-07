"""Tests for :mod:`cp_bg_bench.io.rxrx1` and :class:`Rxrx1Config.channel_zip_keys`."""

from __future__ import annotations

import io as _io
import re
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cp_bg_bench.config import Rxrx1Config, Rxrx1Sample
from cp_bg_bench.io.rxrx1 import (
    _RXRX1_BATCH,
    _SCALAR_OUTPUT_COLUMNS,
    _build_fov_table,
    _select_samples,
    assign_snakemake_batches,
    resolve_metadata,
)

# ── fixtures ─────────────────────────────────────────────────────────────────

_MINIMAL_CFG = Rxrx1Config(
    samples=[Rxrx1Sample(experiment="EXP-1")],
    metadata_url="https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip",
    images_zip_url="https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip",
    channel_names=["Hoechst", "ConA"],
    segmentation={
        "model": "cpsam",
        "channels_for_nucleus": [0],
        "channels_for_cell": [1, 0],
        "default_diameters": {"nucleus": 21, "cytosol": 53},
    },
)

# Matches actual Rxrx1 metadata CSV column schema (no sirna_seq).
_SAMPLE_CSV = """\
site_id,well_id,cell_type,dataset,experiment,plate,well,site,well_type,sirna,sirna_id
EXP-1_1_A01_1,EXP-1_1_A01,HUVEC,train,EXP-1,1,A01,1,positive,si001,1
EXP-1_1_A01_2,EXP-1_1_A01,HUVEC,train,EXP-1,1,A01,2,positive,si001,1
EXP-1_1_B02_1,EXP-1_1_B02,HUVEC,test,EXP-1,1,B02,1,negative,EMPTY,1138
EXP-2_1_A01_1,EXP-2_1_A01,HUVEC,train,EXP-2,1,A01,1,positive,si003,3
"""


def _make_raw() -> pd.DataFrame:
    return pd.read_csv(_io.StringIO(_SAMPLE_CSV))


def _make_zip_bytes() -> bytes:
    """Build a minimal metadata zip matching the live GCS archive structure."""
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rxrx1/metadata.csv", _SAMPLE_CSV)
    return buf.getvalue()


# ── channel_zip_keys ─────────────────────────────────────────────────────────


def test_channel_zip_keys_matches_channel_names() -> None:
    assert _MINIMAL_CFG.channel_zip_keys == ["zip_Hoechst", "zip_ConA"]


def test_channel_zip_keys_length() -> None:
    assert len(_MINIMAL_CFG.channel_zip_keys) == len(_MINIMAL_CFG.channel_names)


# ── _select_samples ───────────────────────────────────────────────────────────


def test_select_samples_by_experiment() -> None:
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    assert set(result["experiment"].unique()) == {"EXP-1"}
    assert len(result) == 3


def test_select_samples_by_experiment_and_plate() -> None:
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1", plate="1")])
    assert set(result["plate"].astype(str).unique()) == {"1"}
    assert set(result["experiment"].unique()) == {"EXP-1"}


def test_select_samples_union_across_entries() -> None:
    raw = _make_raw()
    result = _select_samples(
        raw,
        [Rxrx1Sample(experiment="EXP-1"), Rxrx1Sample(experiment="EXP-2")],
    )
    assert set(result["experiment"].unique()) == {"EXP-1", "EXP-2"}


def test_select_samples_no_match_raises() -> None:
    raw = _make_raw()
    with pytest.raises(ValueError, match="empty selection"):
        _select_samples(raw, [Rxrx1Sample(experiment="DOES-NOT-EXIST")])


def test_select_samples_deduplicates() -> None:
    raw = _make_raw()
    result = _select_samples(
        raw,
        [Rxrx1Sample(experiment="EXP-1"), Rxrx1Sample(experiment="EXP-1")],
    )
    assert not result.duplicated(subset=["experiment", "plate", "well", "site"]).any()


def test_select_samples_by_well() -> None:
    raw = _make_raw()
    result = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1", plate="1", well="A01")])
    assert set(result["well"].unique()) == {"A01"}
    assert len(result) == 2  # two sites for A01


def test_select_samples_well_or_across_entries() -> None:
    """Two entries selecting individual wells from different experiments."""
    raw = _make_raw()
    result = _select_samples(
        raw,
        [
            Rxrx1Sample(experiment="EXP-1", plate="1", well="B02"),
            Rxrx1Sample(experiment="EXP-2", plate="1", well="A01"),
        ],
    )
    assert len(result) == 2
    assert set(zip(result["experiment"], result["well"], strict=True)) == {
        ("EXP-1", "B02"),
        ("EXP-2", "A01"),
    }


def test_select_samples_by_sirna() -> None:
    raw = _make_raw()
    # si001 appears in EXP-1 plate 1 well A01 (2 sites)
    result = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1", sirna="si001")])
    assert set(result["sirna"].unique()) == {"si001"}
    assert len(result) == 2


def test_select_samples_sirna_or_across_entries() -> None:
    """Two entries selecting by different sirna values across experiments."""
    raw = _make_raw()
    result = _select_samples(
        raw,
        [
            Rxrx1Sample(experiment="EXP-1", sirna="si001"),
            Rxrx1Sample(experiment="EXP-2", sirna="si003"),
        ],
    )
    assert set(result["sirna"].unique()) == {"si001", "si003"}
    assert set(result["experiment"].unique()) == {"EXP-1", "EXP-2"}


def test_select_samples_sirna_and_well_filter() -> None:
    """Combining sirna + well selects the intersection."""
    raw = _make_raw()
    # si001 at A01, EMPTY at B02 in EXP-1; requesting si001 + A01 gives 2 sites
    result = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1", sirna="si001", well="A01")])
    assert len(result) == 2
    assert set(result["well"].unique()) == {"A01"}
    assert set(result["sirna"].unique()) == {"si001"}


# ── _build_fov_table ──────────────────────────────────────────────────────────


def test_build_fov_table_column_rename() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    for col in ("Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_Site"):
        assert col in fov.columns
    assert "Metadata_Batch" in fov.columns


def test_build_fov_table_sirna_mapped_to_inchikey() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    assert "Metadata_InChIKey" in fov.columns
    assert fov["Metadata_InChIKey"].iloc[0] == "si001"


def test_build_fov_table_batch_is_fixed() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    assert (fov["Metadata_Batch"] == _RXRX1_BATCH).all()


def test_build_fov_table_id_format() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    assert fov["id"].str.contains("__rxrx1__").all()
    assert (fov["id"].str.count("__") == 4).all()


def test_build_fov_table_zip_path_columns_present() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    for key in _MINIMAL_CFG.channel_zip_keys:
        assert key in fov.columns


def test_build_fov_table_zip_paths_correct_format() -> None:
    """Paths match rxrx1/images/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png."""
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    # Channel 1 ends with _w1.png, channel 2 with _w2.png.
    assert fov["zip_Hoechst"].str.endswith("_w1.png").all()
    assert fov["zip_ConA"].str.endswith("_w2.png").all()
    # Path contains Plate prefix and the rxrx1/images/ root.
    assert fov["zip_Hoechst"].str.contains("/Plate").all()
    assert fov["zip_Hoechst"].str.startswith("rxrx1/images/EXP-1/").all()


def test_build_fov_table_zip_path_contains_well_and_site() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    # Every path must match: rxrx1/images/{experiment}/Plate<n>/<well>_s<site>_w<ch>.png
    pattern = re.compile(r"^rxrx1/images/EXP-1/Plate\d+/[A-Z]\d+_s\d+_w\d+\.png$")
    assert fov["zip_Hoechst"].apply(lambda p: bool(pattern.match(p))).all()


# ── assign_snakemake_batches ──────────────────────────────────────────────────


def test_assign_snakemake_batches_column_added() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    bucketed = assign_snakemake_batches(fov, batch_size=2)
    assert "snakemake_batch" in bucketed.columns


def test_assign_snakemake_batches_uses_source_prefix() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    bucketed = assign_snakemake_batches(fov, batch_size=10)
    assert bucketed["snakemake_batch"].str.startswith("EXP-1").all()


def test_assign_snakemake_batches_resets_per_source() -> None:
    raw = _make_raw()
    selected = _select_samples(
        raw, [Rxrx1Sample(experiment="EXP-1"), Rxrx1Sample(experiment="EXP-2")]
    )
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    bucketed = assign_snakemake_batches(fov, batch_size=2)
    batch_ids = set(bucketed["snakemake_batch"])
    assert "EXP-1_snakemake_batch_0" in batch_ids
    assert "EXP-2_snakemake_batch_0" in batch_ids


def test_assign_snakemake_batches_invalid_batch_size() -> None:
    raw = _make_raw()
    selected = _select_samples(raw, [Rxrx1Sample(experiment="EXP-1")])
    fov = _build_fov_table(selected, _MINIMAL_CFG)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        assign_snakemake_batches(fov, batch_size=0)


# ── resolve_metadata (mocked zip download) ───────────────────────────────────


def _fake_urlopen(zip_bytes: bytes) -> MagicMock:
    """Return a urlopen mock whose context manager yields zip_bytes on .read()."""
    resp = MagicMock()
    resp.read.return_value = zip_bytes
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_resolve_metadata_output_columns(tmp_path: Path) -> None:
    """resolve_metadata returns all scalar columns + channel zip keys."""
    zip_bytes = _make_zip_bytes()

    with patch(
        "cp_bg_bench.io.rxrx1.urllib.request.urlopen", return_value=_fake_urlopen(zip_bytes)
    ):
        df = resolve_metadata(_MINIMAL_CFG, cache_dir=tmp_path, batch_size=10)

    expected_cols = list(_SCALAR_OUTPUT_COLUMNS) + _MINIMAL_CFG.channel_zip_keys
    assert list(df.columns) == expected_cols


def test_resolve_metadata_row_count_matches_selection(tmp_path: Path) -> None:
    """Only rows from configured experiments appear in the output."""
    zip_bytes = _make_zip_bytes()

    with patch(
        "cp_bg_bench.io.rxrx1.urllib.request.urlopen", return_value=_fake_urlopen(zip_bytes)
    ):
        df = resolve_metadata(_MINIMAL_CFG, cache_dir=tmp_path, batch_size=10)

    # EXP-1 has 3 rows; EXP-2 is excluded by the config.
    assert len(df) == 3


def test_resolve_metadata_cache_reused(tmp_path: Path) -> None:
    """Second call reuses the extracted CSV (urlopen called once)."""
    zip_bytes = _make_zip_bytes()

    with patch(
        "cp_bg_bench.io.rxrx1.urllib.request.urlopen", return_value=_fake_urlopen(zip_bytes)
    ) as mock_open:
        resolve_metadata(_MINIMAL_CFG, cache_dir=tmp_path, batch_size=10)
        resolve_metadata(_MINIMAL_CFG, cache_dir=tmp_path, batch_size=10)

    assert mock_open.call_count == 1
