"""Tests for :mod:`cp_bg_bench.download.tiffs` (rule B building blocks)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr
from botocore.exceptions import ClientError, EndpointConnectionError
from tifffile import tifffile

from cp_bg_bench.download.tiffs import (
    PLATE_GROUP_COLUMNS,
    download_batch_to_zarr,
    fetch_tiff,
    plate_store_path,
    stack_fov,
)
from cp_bg_bench.io.jump import JumpS3Error

CHANNEL_KEYS = ["s3_OrigDNA", "s3_OrigAGP", "s3_OrigER", "s3_OrigMito", "s3_OrigRNA"]
# Derive short channel names from CHANNEL_KEYS so the two lists can't drift.
CHANNEL_NAMES: tuple[str, ...] = tuple(k.removeprefix("s3_Orig") for k in CHANNEL_KEYS)


# --------- synthetic S3 client ------------------------------------------------


def _encode_tiff(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


class FakeS3Client:
    """Minimal boto3 stand-in: returns pre-registered TIFF blobs per URL."""

    def __init__(self) -> None:
        self._blobs: dict[tuple[str, str], bytes] = {}
        self._status: dict[tuple[str, str], int] = {}
        self._errors: dict[tuple[str, str], Exception] = {}
        self.call_log: list[tuple[str, str]] = []

    def register(
        self,
        url: str,
        arr: np.ndarray | None,
        *,
        status: int = 200,
        error: Exception | None = None,
    ) -> None:
        assert url.startswith("s3://")
        bucket, _, key = url[5:].partition("/")
        k = (bucket, key)
        if error is not None:
            self._errors[k] = error
            return
        assert arr is not None
        self._blobs[k] = _encode_tiff(arr)
        self._status[k] = status

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self.call_log.append((Bucket, Key))
        k = (Bucket, Key)
        if k in self._errors:
            raise self._errors[k]
        body = io.BytesIO(self._blobs[k])
        return {
            "ResponseMetadata": {"HTTPStatusCode": self._status[k]},
            "Body": body,
        }


def _make_client_with_fov(
    plate: str, well: str, site: str, shape: tuple[int, int] = (8, 12)
) -> tuple[FakeS3Client, list[str]]:
    """Build a FakeS3Client with all five channel TIFFs for one FOV."""
    client = FakeS3Client()
    urls: list[str] = []
    rng = np.random.default_rng(abs(hash((plate, well, site))) % (2**32))
    for ch in CHANNEL_NAMES:
        url = f"s3://bucket/{plate}/{well}/{ch}_{site}.tiff"
        arr = rng.integers(0, 65535, size=shape, dtype=np.uint16)
        client.register(url, arr)
        urls.append(url)
    return client, urls


# --------- fetch_tiff ---------------------------------------------------------


def test_fetch_tiff_decodes_2d_uint16() -> None:
    client = FakeS3Client()
    payload = np.arange(48, dtype=np.uint16).reshape(6, 8)
    client.register("s3://b/k.tiff", payload)

    arr = fetch_tiff(client, "s3://b/k.tiff")
    np.testing.assert_array_equal(arr, payload)
    assert arr.dtype == np.uint16


def test_fetch_tiff_rejects_non_200() -> None:
    client = FakeS3Client()
    client.register("s3://b/k.tiff", np.zeros((4, 4), dtype=np.uint16), status=404)
    with pytest.raises(JumpS3Error, match="unexpected HTTP 404"):
        fetch_tiff(client, "s3://b/k.tiff")


def test_fetch_tiff_wraps_botocore_client_error() -> None:
    client = FakeS3Client()
    client.register(
        "s3://b/missing.tiff",
        None,
        error=ClientError(
            error_response={"Error": {"Code": "NoSuchKey", "Message": "nope"}},
            operation_name="GetObject",
        ),
    )
    with pytest.raises(JumpS3Error, match="failed to fetch"):
        fetch_tiff(client, "s3://b/missing.tiff")


def test_fetch_tiff_wraps_botocore_core_error() -> None:
    """BotoCoreError subclasses (network/transport layer) wrap the same way."""
    client = FakeS3Client()
    client.register(
        "s3://b/unreachable.tiff",
        None,
        error=EndpointConnectionError(endpoint_url="https://example.invalid"),
    )
    with pytest.raises(JumpS3Error, match="failed to fetch"):
        fetch_tiff(client, "s3://b/unreachable.tiff")


def test_fetch_tiff_rejects_3d_tiff() -> None:
    client = FakeS3Client()
    # multi-page TIFF → 3-D when decoded.
    blob = _encode_tiff(np.zeros((3, 4, 4), dtype=np.uint16))
    client._blobs[("b", "multi.tiff")] = blob  # noqa: SLF001
    client._status[("b", "multi.tiff")] = 200  # noqa: SLF001

    with pytest.raises(ValueError, match="expected 2-D TIFF"):
        fetch_tiff(client, "s3://b/multi.tiff")


# --------- stack_fov ----------------------------------------------------------


def test_stack_fov_preserves_channel_order() -> None:
    client, urls = _make_client_with_fov("P1", "A01", "1")
    stack = stack_fov(client, urls)

    assert stack.shape == (5, 8, 12)
    assert stack.dtype == np.uint16
    # Each plane equals what fetch_tiff returns for that url.
    for idx, url in enumerate(urls):
        expected = fetch_tiff(client, url)
        np.testing.assert_array_equal(stack[idx], expected)


def test_stack_fov_shape_mismatch_raises() -> None:
    client = FakeS3Client()
    client.register("s3://b/a.tiff", np.zeros((8, 8), dtype=np.uint16))
    client.register("s3://b/b.tiff", np.zeros((9, 8), dtype=np.uint16))

    with pytest.raises(ValueError, match="channel shape mismatch"):
        stack_fov(client, ["s3://b/a.tiff", "s3://b/b.tiff"])


def test_stack_fov_rejects_empty_urls() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        stack_fov(FakeS3Client(), [])


# --------- download_batch_to_zarr --------------------------------------------


def _metadata_row(
    plate: str,
    well: str,
    site: str,
    channel_urls: list[str],
    *,
    source: str = "source_X",
    batch: str = "B1",
    channel_keys: list[str] = CHANNEL_KEYS,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"{source}__{batch}__{plate}__{well}__{site}",
        "Metadata_Source": source,
        "Metadata_Batch": batch,
        "Metadata_Plate": plate,
    }
    for key, url in zip(channel_keys, channel_urls, strict=True):
        row[key] = url
    return row


def _build_batch(
    n_fovs_per_plate: dict[str, int],
    *,
    channels: tuple[str, ...] = CHANNEL_NAMES,
    channel_keys: list[str] = CHANNEL_KEYS,
    shape: tuple[int, int] = (8, 12),
) -> tuple[FakeS3Client, pd.DataFrame, dict[str, np.ndarray]]:
    """Construct a metadata DataFrame + a client + expected per-FOV stacks.

    Random pixel data is deterministic (seeded); ``expected[fov_id]`` is
    the ``(C, H, W)`` uint16 stack the download should land in the
    per-plate Zarr group so round-trip asserts can verify content, not
    just shape.
    """
    assert len(channels) == len(channel_keys), "channels and channel_keys must align"
    client = FakeS3Client()
    rows: list[dict[str, Any]] = []
    expected: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for plate, n_fovs in n_fovs_per_plate.items():
        for i in range(n_fovs):
            well = f"A{i + 1:02d}"
            urls: list[str] = []
            planes: list[np.ndarray] = []
            for ch in channels:
                url = f"s3://bucket/{plate}/{well}/{ch}.tiff"
                plane = rng.integers(0, 65535, size=shape, dtype=np.uint16)
                client.register(url, plane)
                urls.append(url)
                planes.append(plane)
            row = _metadata_row(plate, well, "1", urls, channel_keys=channel_keys)
            rows.append(row)
            expected[row["id"]] = np.stack(planes, axis=0).astype(np.uint16)
    return client, pd.DataFrame(rows), expected


def test_download_batch_to_zarr_writes_per_plate_store(tmp_path: Path) -> None:
    client, meta, expected = _build_batch({"P1": 2, "P2": 1})
    summary = download_batch_to_zarr(meta, tmp_path, CHANNEL_KEYS, n_threads=2, s3_client=client)

    assert len(summary) == 2
    p1_store = plate_store_path(tmp_path, "source_X", "B1", "P1")
    p2_store = plate_store_path(tmp_path, "source_X", "B1", "P2")
    assert summary[str(p1_store)] == 2
    assert summary[str(p2_store)] == 1

    p1 = zarr.open_group(store=str(p1_store), mode="r")
    p2 = zarr.open_group(store=str(p2_store), mode="r")
    assert set(p1.array_keys()) == {
        "source_X__B1__P1__A01__1",
        "source_X__B1__P1__A02__1",
    }
    assert set(p2.array_keys()) == {"source_X__B1__P2__A01__1"}

    # Content round-trip: every FOV must materialise byte-for-byte as
    # the planes the fake S3 client served, in the declared channel
    # order. This is the "datasets truly matched" invariant for rule B.
    for group in (p1, p2):
        for fov_id in group.array_keys():
            arr = group[fov_id]
            assert arr.shape == (5, 8, 12)
            assert arr.dtype == np.uint16
            np.testing.assert_array_equal(arr[:], expected[fov_id])


def test_download_batch_to_zarr_is_idempotent(tmp_path: Path) -> None:
    client, meta, expected = _build_batch({"P1": 2})
    download_batch_to_zarr(meta, tmp_path, CHANNEL_KEYS, n_threads=1, s3_client=client)
    calls_after_first_run = len(client.call_log)

    # Second call: same metadata, same output. Existing arrays should
    # short-circuit in write_fov_array; S3 gets re-read (we don't try
    # to be clever about skipping the fetch) but nothing crashes.
    download_batch_to_zarr(meta, tmp_path, CHANNEL_KEYS, n_threads=1, s3_client=client)
    assert len(client.call_log) == 2 * calls_after_first_run

    group = zarr.open_group(store=str(plate_store_path(tmp_path, "source_X", "B1", "P1")), mode="r")
    assert len(list(group.array_keys())) == 2
    for fov_id in group.array_keys():
        np.testing.assert_array_equal(group[fov_id][:], expected[fov_id])


def test_download_batch_parametric_channel_count(tmp_path: Path) -> None:
    """Rule B must work with != 5 channels (rxrx1 will ship 6, etc.)."""
    ch_keys = ["s3_Hoechst", "s3_ConA", "s3_Phalloidin"]
    client, meta, expected = _build_batch(
        {"P1": 2},
        channels=("Hoechst", "ConA", "Phalloidin"),
        channel_keys=ch_keys,
        shape=(6, 10),
    )
    summary = download_batch_to_zarr(meta, tmp_path, ch_keys, n_threads=2, s3_client=client)

    store = plate_store_path(tmp_path, "source_X", "B1", "P1")
    assert summary[str(store)] == 2
    group = zarr.open_group(store=str(store), mode="r")
    for fov_id, expected_stack in expected.items():
        arr = group[fov_id]
        assert arr.shape == (3, 6, 10)
        np.testing.assert_array_equal(arr[:], expected_stack)


def test_download_batch_missing_columns_raises(tmp_path: Path) -> None:
    client, meta, _ = _build_batch({"P1": 1})
    broken = meta.drop(columns=["s3_OrigDNA"])
    with pytest.raises(KeyError, match="forceall resolve_metadata"):
        download_batch_to_zarr(broken, tmp_path, CHANNEL_KEYS, n_threads=1, s3_client=client)


def test_download_batch_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        download_batch_to_zarr(
            pd.DataFrame(columns=["id", *PLATE_GROUP_COLUMNS, *CHANNEL_KEYS]),
            tmp_path,
            CHANNEL_KEYS,
            s3_client=FakeS3Client(),
        )


def test_download_batch_duplicate_ids_raise(tmp_path: Path) -> None:
    client, meta, _ = _build_batch({"P1": 2})
    dup = pd.concat([meta, meta.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate FOV ids"):
        download_batch_to_zarr(dup, tmp_path, CHANNEL_KEYS, n_threads=1, s3_client=client)


def test_download_batch_s3_failure_propagates(tmp_path: Path) -> None:
    client, meta, _ = _build_batch({"P1": 2})
    # Break the second FOV's DNA channel — expect JumpS3Error bubbles up.
    url = meta.iloc[1]["s3_OrigDNA"]
    client.register(
        url,
        None,
        error=ClientError(
            error_response={"Error": {"Code": "NoSuchKey", "Message": "gone"}},
            operation_name="GetObject",
        ),
    )
    with pytest.raises(JumpS3Error):
        download_batch_to_zarr(meta, tmp_path, CHANNEL_KEYS, n_threads=1, s3_client=client)


def test_download_batch_clips_thread_count(tmp_path: Path) -> None:
    """`n_threads` is clamped to MAX_DOWNLOAD_WORKERS regardless of caller intent."""
    from cp_bg_bench.download.tiffs import MAX_DOWNLOAD_WORKERS

    client, meta, expected = _build_batch({"P1": 3})
    summary = download_batch_to_zarr(
        meta, tmp_path, CHANNEL_KEYS, n_threads=MAX_DOWNLOAD_WORKERS * 10, s3_client=client
    )
    assert summary[str(plate_store_path(tmp_path, "source_X", "B1", "P1"))] == 3


def test_plate_store_path_layout(tmp_path: Path) -> None:
    p = plate_store_path(tmp_path, "source_2", "B1", "P1")
    assert p == tmp_path / "full_images" / "source_2__B1__P1.zarr"
