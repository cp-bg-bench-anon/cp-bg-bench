"""Tests for :mod:`cp_bg_bench.download.rxrx3_core_jp2`."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import zarr
from PIL import Image

from cp_bg_bench.download.rxrx3_core_jp2 import (
    MAX_DECODE_WORKERS,
    PLATE_GROUP_COLUMNS,
    decode_jp2,
    download_batch_to_zarr,
    load_shard,
)
from cp_bg_bench.io.zarr_io import plate_store_path

_DEFAULT_SHAPE: tuple[int, int] = (8, 12)
_CHANNEL_NAMES = ["Hoechst", "ConA", "Phalloidin", "Syto14", "MitoTracker", "WGA"]
_HF_REPO = "fake/repo"


# ── helpers ───────────────────────────────────────────────────────────────────


def _encode_jp2(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG2000")
    return buf.getvalue()


def _build_batch(
    n_fovs: int = 3,
    *,
    channel_names: list[str] = _CHANNEL_NAMES,
    shape: tuple[int, int] = _DEFAULT_SHAPE,
    source: str = "gene-001",
    batch: str = "rxrx3_core",
    plate: str = "1",
    shard_idx: int = 15,
) -> tuple[dict[str, bytes], pd.DataFrame, dict[str, np.ndarray]]:
    """Return (key_to_bytes, meta_batch, expected_stacks) for download tests."""
    rng = np.random.default_rng(42)
    key_to_bytes: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    expected: dict[str, np.ndarray] = {}

    for i in range(n_fovs):
        address = f"A{i + 1:02d}"
        fov_id = f"{source}__{batch}__{plate}__{address}__1"
        key_prefix = f"{source}/Plate{plate}/{address}_s1"
        planes: list[np.ndarray] = []

        for ch in range(1, len(channel_names) + 1):
            plane = rng.integers(0, 255, size=shape, dtype=np.uint8)
            key_to_bytes[f"{key_prefix}_{ch}"] = _encode_jp2(plane)
            planes.append(plane)

        rows.append(
            {
                "id": fov_id,
                "Metadata_Source": source,
                "Metadata_Batch": batch,
                "Metadata_Plate": plate,
                "parquet_shard": shard_idx,
                "parquet_key_prefix": key_prefix,
            }
        )
        expected[fov_id] = np.stack(planes, axis=0).astype(np.uint16)

    return key_to_bytes, pd.DataFrame(rows), expected


# ── decode_jp2 ────────────────────────────────────────────────────────────────


def test_decode_jp2_shape_and_dtype() -> None:
    arr = np.zeros(_DEFAULT_SHAPE, dtype=np.uint8)
    result = decode_jp2(_encode_jp2(arr))
    assert result.shape == _DEFAULT_SHAPE
    assert result.dtype == np.uint8


def test_decode_jp2_content_roundtrip() -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(8, 12), dtype=np.uint8)
    result = decode_jp2(_encode_jp2(arr))
    np.testing.assert_array_equal(result, arr)


# ── load_shard ────────────────────────────────────────────────────────────────


def _make_fake_table(
    keys: list[str],
    blobs: list[bytes | dict[str, Any]],
) -> pa.Table:
    """Build a minimal pyarrow table matching the HF parquet shard schema."""
    return pa.table({"__key__": keys, "jp2": blobs})


def _mock_fs_open(table: pa.Table) -> MagicMock:
    """Return a mock HfFileSystem whose open() context manager yields a fake filehandle."""
    fh = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = lambda s: fh
    ctx.__exit__ = MagicMock(return_value=False)
    fs = MagicMock()
    fs.open.return_value = ctx
    return fs, fh, table


def _patch_load_shard_internals(table: pa.Table):
    """Context manager patching the lazy HF + parquet imports inside load_shard."""
    fs, _, _ = _mock_fs_open(table)
    return (
        patch("huggingface_hub.HfFileSystem", return_value=fs),
        patch("pyarrow.parquet.read_table", return_value=table),
    )


def test_load_shard_returns_key_bytes_dict() -> None:
    jp2 = b"fakebytes"
    table = _make_fake_table(["gene-001/Plate1/A01_s1_1"], [jp2])
    p1, p2 = _patch_load_shard_internals(table)

    with p1, p2:
        result = load_shard(_HF_REPO, 15)

    assert result == {"gene-001/Plate1/A01_s1_1": jp2}


def test_load_shard_unwraps_hf_image_feature_dict() -> None:
    """HF Image feature stores bytes as {"bytes": ..., "path": ...} — must be unwrapped."""
    jp2 = b"rawjp2bytes"
    hf_dict = {"bytes": jp2, "path": None}
    table = _make_fake_table(["gene-001/Plate1/A01_s1_1"], [hf_dict])
    p1, p2 = _patch_load_shard_internals(table)

    with p1, p2:
        result = load_shard(_HF_REPO, 15)

    assert result["gene-001/Plate1/A01_s1_1"] == jp2


def test_load_shard_passes_raw_bytes_through() -> None:
    """Plain bytes (non-dict) must pass through unchanged."""
    jp2 = b"plainbytes"
    table = _make_fake_table(["key_1"], [jp2])
    p1, p2 = _patch_load_shard_internals(table)

    with p1, p2:
        result = load_shard(_HF_REPO, 15)

    assert result["key_1"] == jp2


# ── download_batch_to_zarr ────────────────────────────────────────────────────


def test_download_batch_to_zarr_writes_fovs(tmp_path: Path) -> None:
    key_to_bytes, meta, expected = _build_batch(n_fovs=2)

    with patch("cp_bg_bench.download.rxrx3_core_jp2.load_shard", return_value=key_to_bytes):
        summary = download_batch_to_zarr(meta, tmp_path, _HF_REPO, _CHANNEL_NAMES)

    store = plate_store_path(tmp_path, "gene-001", "rxrx3_core", "1")
    root = zarr.open_group(str(store), mode="r")
    for fov_id, exp_stack in expected.items():
        arr = root[fov_id][:]
        assert arr.shape == exp_stack.shape
        assert arr.dtype == np.uint16
        np.testing.assert_array_equal(arr, exp_stack)

    assert sum(summary.values()) == 2


def test_download_batch_to_zarr_channel_shape(tmp_path: Path) -> None:
    """Output array shape is (C, H, W) with C = len(channel_names)."""
    key_to_bytes, meta, _ = _build_batch(n_fovs=1, channel_names=["A", "B", "C"])

    with patch("cp_bg_bench.download.rxrx3_core_jp2.load_shard", return_value=key_to_bytes):
        download_batch_to_zarr(meta, tmp_path, _HF_REPO, ["A", "B", "C"])

    store = plate_store_path(tmp_path, "gene-001", "rxrx3_core", "1")
    root = zarr.open_group(str(store), mode="r")
    fov_id = meta["id"].iloc[0]
    assert root[fov_id].shape[0] == 3


def test_download_batch_to_zarr_missing_key_raises(tmp_path: Path) -> None:
    """A key prefix absent from the shard dict raises KeyError."""
    key_to_bytes, meta, _ = _build_batch(n_fovs=1)
    # Remove the first channel key.
    first_key = next(iter(key_to_bytes))
    del key_to_bytes[first_key]

    with (
        patch("cp_bg_bench.download.rxrx3_core_jp2.load_shard", return_value=key_to_bytes),
        pytest.raises(KeyError, match="not found in shard"),
    ):
        download_batch_to_zarr(meta, tmp_path, _HF_REPO, _CHANNEL_NAMES)


def test_download_batch_to_zarr_empty_batch_raises(tmp_path: Path) -> None:
    empty = pd.DataFrame(
        columns=["id", "parquet_shard", "parquet_key_prefix", *PLATE_GROUP_COLUMNS]
    )
    with pytest.raises(ValueError, match="empty"):
        download_batch_to_zarr(empty, tmp_path, _HF_REPO, _CHANNEL_NAMES)


def test_download_batch_to_zarr_missing_columns_raises(tmp_path: Path) -> None:
    _, meta, _ = _build_batch(n_fovs=1)
    meta = meta.drop(columns=["parquet_shard"])
    with pytest.raises(KeyError, match="parquet_shard"):
        download_batch_to_zarr(meta, tmp_path, _HF_REPO, _CHANNEL_NAMES)


def test_download_batch_to_zarr_duplicate_ids_raises(tmp_path: Path) -> None:
    key_to_bytes, meta, _ = _build_batch(n_fovs=1)
    meta = pd.concat([meta, meta], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate FOV ids"):
        download_batch_to_zarr(meta, tmp_path, _HF_REPO, _CHANNEL_NAMES)


def test_max_decode_workers_positive() -> None:
    assert MAX_DECODE_WORKERS >= 1
