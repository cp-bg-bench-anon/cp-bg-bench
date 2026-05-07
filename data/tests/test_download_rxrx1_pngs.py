"""Tests for :mod:`cp_bg_bench.download.rxrx1_pngs`."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import zarr
from PIL import Image

from cp_bg_bench.download.rxrx1_pngs import (
    MAX_DOWNLOAD_WORKERS,
    download_batch_to_zarr,
    fetch_png,
    stack_fov,
)
from cp_bg_bench.io.zarr_io import plate_store_path

CHANNEL_ZIP_KEYS = [
    "zip_Hoechst",
    "zip_ConA",
    "zip_Phalloidin",
    "zip_Syto14",
    "zip_MitoTracker",
    "zip_WGA",
]
IMAGES_ZIP_URL = "https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip"
_DEFAULT_SHAPE: tuple[int, int] = (8, 12)


# ── helpers ──────────────────────────────────────────────────────────────────


def _encode_png(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _make_fake_rz_class(
    blobs: dict[str, bytes],
) -> tuple[type, list[int]]:
    """Return (FakeRemoteZip class, open_count list) for patching remotezip.RemoteZip.

    Each instantiation increments open_count[0], allowing tests to assert that
    RemoteZip is opened at most once per worker thread, not once per FOV.
    """
    open_count = [0]

    class FakeRemoteZip:
        def __init__(self, url: str) -> None:
            open_count[0] += 1

        def read(self, path: str) -> bytes:
            if path not in blobs:
                raise KeyError(f"FakeRemoteZip: unregistered path {path!r}")
            return blobs[path]

        def __enter__(self) -> FakeRemoteZip:
            return self

        def __exit__(self, *_: Any) -> None:
            pass

    return FakeRemoteZip, open_count


def _build_batch(
    n_fovs_per_plate: dict[str, int],
    *,
    channel_keys: list[str] = CHANNEL_ZIP_KEYS,
    shape: tuple[int, int] = _DEFAULT_SHAPE,
    source: str = "EXP-TEST",
    batch: str = "rxrx1",
) -> tuple[dict[str, bytes], pd.DataFrame, dict[str, np.ndarray]]:
    """Build (blobs, metadata_df, expected_stacks) for download tests.

    ``expected[fov_id]`` is the ``(C, H, W)`` uint16 stack that the download
    should write into the Zarr store, enabling content round-trip assertions.
    """
    blobs: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    expected: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)

    for plate, n_fovs in n_fovs_per_plate.items():
        for i in range(n_fovs):
            well = f"A{i + 1:02d}"
            site = "1"
            planes: list[np.ndarray] = []
            row: dict[str, Any] = {
                "id": f"{source}__{batch}__{plate}__{well}__{site}",
                "Metadata_Source": source,
                "Metadata_Batch": batch,
                "Metadata_Plate": plate,
            }
            for ch_idx, key in enumerate(channel_keys, start=1):
                path = f"rxrx1/images/{source}/Plate{plate}/{well}_s{site}_w{ch_idx}.png"
                plane = rng.integers(0, 255, size=shape, dtype=np.uint8)
                blobs[path] = _encode_png(plane)
                row[key] = path
                planes.append(plane)
            rows.append(row)
            expected[row["id"]] = np.stack(planes, axis=0).astype(np.uint16)

    return blobs, pd.DataFrame(rows), expected


# ── fetch_png ─────────────────────────────────────────────────────────────────


def test_fetch_png_decodes_grayscale_uint8() -> None:
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    fake_rz = SimpleNamespace(read=lambda p: _encode_png(arr))
    result = fetch_png(fake_rz, "some/path.png")
    np.testing.assert_array_equal(result, arr)
    assert result.dtype == np.uint8


# ── stack_fov ─────────────────────────────────────────────────────────────────


def test_stack_fov_preserves_channel_order() -> None:
    planes = [np.full((4, 4), i, dtype=np.uint8) for i in range(3)]
    blobs = {f"ch{i}.png": _encode_png(p) for i, p in enumerate(planes)}
    fake_rz = SimpleNamespace(read=lambda p: blobs[p])

    stack = stack_fov(fake_rz, ["ch0.png", "ch1.png", "ch2.png"])

    assert stack.shape == (3, 4, 4)
    assert stack.dtype == np.uint16
    for i, plane in enumerate(planes):
        np.testing.assert_array_equal(stack[i], plane.astype(np.uint16))


def test_stack_fov_shape_mismatch_raises() -> None:
    blobs = {
        "a.png": _encode_png(np.zeros((4, 4), dtype=np.uint8)),
        "b.png": _encode_png(np.zeros((5, 4), dtype=np.uint8)),
    }
    fake_rz = SimpleNamespace(read=lambda p: blobs[p])
    with pytest.raises(ValueError, match="channel shape mismatch"):
        stack_fov(fake_rz, ["a.png", "b.png"])


def test_stack_fov_empty_paths_raises() -> None:
    fake_rz = SimpleNamespace(read=lambda p: b"")
    with pytest.raises(ValueError, match="at least one channel"):
        stack_fov(fake_rz, [])


# ── download_batch_to_zarr ────────────────────────────────────────────────────


def test_download_batch_to_zarr_writes_per_plate_store(tmp_path: Path) -> None:
    blobs, meta, expected = _build_batch({"P1": 2, "P2": 1})
    FakeRZ, _ = _make_fake_rz_class(blobs)

    with patch("remotezip.RemoteZip", FakeRZ):
        summary = download_batch_to_zarr(
            meta, tmp_path, IMAGES_ZIP_URL, CHANNEL_ZIP_KEYS, n_threads=2
        )

    source, batch = "EXP-TEST", "rxrx1"
    p1_store = plate_store_path(tmp_path, source, batch, "P1")
    p2_store = plate_store_path(tmp_path, source, batch, "P2")
    assert summary[str(p1_store)] == 2
    assert summary[str(p2_store)] == 1

    for store in (p1_store, p2_store):
        group = zarr.open_group(store=str(store), mode="r")
        for fov_id in group.array_keys():
            arr = group[fov_id]
            assert arr.shape == (len(CHANNEL_ZIP_KEYS), *_DEFAULT_SHAPE)
            assert arr.dtype == np.uint16
            # Content round-trip: channel ordering and pixel values must match.
            np.testing.assert_array_equal(arr[:], expected[fov_id])


def test_download_batch_to_zarr_is_idempotent(tmp_path: Path) -> None:
    blobs, meta, expected = _build_batch({"P1": 2})
    FakeRZ, _ = _make_fake_rz_class(blobs)

    with patch("remotezip.RemoteZip", FakeRZ):
        download_batch_to_zarr(meta, tmp_path, IMAGES_ZIP_URL, CHANNEL_ZIP_KEYS, n_threads=1)
        download_batch_to_zarr(meta, tmp_path, IMAGES_ZIP_URL, CHANNEL_ZIP_KEYS, n_threads=1)

    group = zarr.open_group(
        store=str(plate_store_path(tmp_path, "EXP-TEST", "rxrx1", "P1")), mode="r"
    )
    assert len(list(group.array_keys())) == 2
    for fov_id in group.array_keys():
        np.testing.assert_array_equal(group[fov_id][:], expected[fov_id])


def test_remote_zip_opened_at_most_once_per_worker(tmp_path: Path) -> None:
    """Central-directory cost must be O(workers), not O(FOVs)."""
    n_fovs = 8
    n_threads = 2
    blobs, meta, _ = _build_batch({"P1": n_fovs})
    FakeRZ, open_count = _make_fake_rz_class(blobs)

    with patch("remotezip.RemoteZip", FakeRZ):
        download_batch_to_zarr(
            meta, tmp_path, IMAGES_ZIP_URL, CHANNEL_ZIP_KEYS, n_threads=n_threads
        )

    assert open_count[0] <= n_threads, (
        f"RemoteZip opened {open_count[0]} times for {n_fovs} FOVs "
        f"with {n_threads} workers; expected ≤ {n_threads}"
    )


def test_download_batch_missing_columns_raises(tmp_path: Path) -> None:
    blobs, meta, _ = _build_batch({"P1": 1})
    broken = meta.drop(columns=["zip_Hoechst"])
    FakeRZ, _ = _make_fake_rz_class(blobs)

    with (
        patch("remotezip.RemoteZip", FakeRZ),
        pytest.raises(KeyError, match="forceall resolve_metadata"),
    ):
        download_batch_to_zarr(broken, tmp_path, IMAGES_ZIP_URL, CHANNEL_ZIP_KEYS)


def test_download_batch_empty_raises(tmp_path: Path) -> None:
    plate_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]
    with pytest.raises(ValueError, match="empty"):
        download_batch_to_zarr(
            pd.DataFrame(columns=["id", *plate_cols, *CHANNEL_ZIP_KEYS]),
            tmp_path,
            IMAGES_ZIP_URL,
            CHANNEL_ZIP_KEYS,
        )


def test_download_batch_duplicate_ids_raise(tmp_path: Path) -> None:
    blobs, meta, _ = _build_batch({"P1": 2})
    dup = pd.concat([meta, meta.iloc[[0]]], ignore_index=True)
    FakeRZ, _ = _make_fake_rz_class(blobs)

    with patch("remotezip.RemoteZip", FakeRZ), pytest.raises(ValueError, match="duplicate FOV ids"):
        download_batch_to_zarr(dup, tmp_path, IMAGES_ZIP_URL, CHANNEL_ZIP_KEYS)


def test_download_batch_clips_thread_count(tmp_path: Path) -> None:
    blobs, meta, _ = _build_batch({"P1": 3})
    FakeRZ, _ = _make_fake_rz_class(blobs)

    with patch("remotezip.RemoteZip", FakeRZ):
        summary = download_batch_to_zarr(
            meta,
            tmp_path,
            IMAGES_ZIP_URL,
            CHANNEL_ZIP_KEYS,
            n_threads=MAX_DOWNLOAD_WORKERS * 10,
        )

    assert summary[str(plate_store_path(tmp_path, "EXP-TEST", "rxrx1", "P1"))] == 3


def test_download_batch_parametric_channel_count(tmp_path: Path) -> None:
    """download_batch_to_zarr works with != 6 channels (e.g. a 3-channel subset)."""
    keys = ["zip_Hoechst", "zip_ConA", "zip_Phalloidin"]
    blobs, meta, expected = _build_batch({"P1": 2}, channel_keys=keys, shape=(6, 10))
    FakeRZ, _ = _make_fake_rz_class(blobs)

    with patch("remotezip.RemoteZip", FakeRZ):
        summary = download_batch_to_zarr(meta, tmp_path, IMAGES_ZIP_URL, keys, n_threads=2)

    store = plate_store_path(tmp_path, "EXP-TEST", "rxrx1", "P1")
    assert summary[str(store)] == 2
    group = zarr.open_group(store=str(store), mode="r")
    for fov_id, expected_stack in expected.items():
        arr = group[fov_id]
        assert arr.shape == (3, 6, 10)
        np.testing.assert_array_equal(arr[:], expected_stack)
