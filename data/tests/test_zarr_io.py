"""Tests for :mod:`cp_bg_bench.io.zarr_io` — Zarr v3 store conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec

from cp_bg_bench.io.zarr_io import (
    LABEL_DIMENSION_NAMES,
    ZARR_DIMENSION_NAMES,
    default_codec,
    open_plate_group,
    plate_store_path,
    seg_store_path,
    write_fov_array,
    write_label_array,
)


def _synthetic_stack(c: int = 5, h: int = 8, w: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, np.iinfo(np.uint16).max, size=(c, h, w), dtype=np.uint16)


def test_default_codec_is_zstd_shuffle() -> None:
    codec = default_codec()
    assert isinstance(codec, BloscCodec)
    assert codec.cname.value == "zstd"
    assert codec.clevel == 3
    # shuffle is a BloscShuffle enum; compare by value rather than str().
    assert codec.shuffle.value == "shuffle"


def test_plate_store_path_and_seg_store_path(tmp_path: Path) -> None:
    p = plate_store_path(tmp_path, "src", "b1", "plate1")
    assert p == tmp_path / "full_images" / "src__b1__plate1.zarr"
    s = seg_store_path(tmp_path, "src", "b1", "plate1")
    assert s == tmp_path / "segmentation" / "src__b1__plate1.zarr"


def test_open_plate_group_is_zarr_format_3(tmp_path: Path) -> None:
    store = tmp_path / "plate.zarr"
    group = open_plate_group(store)
    assert group.metadata.zarr_format == 3


def test_write_fov_array_round_trip(tmp_path: Path) -> None:
    stack = _synthetic_stack()
    group = open_plate_group(tmp_path / "plate.zarr")

    write_fov_array(group, "fov_a", stack)

    reopened = zarr.open_group(store=str(tmp_path / "plate.zarr"), mode="r")
    arr = reopened["fov_a"]
    assert arr.metadata.zarr_format == 3
    assert tuple(arr.shape) == stack.shape
    assert arr.dtype == np.uint16
    assert arr.chunks == stack.shape  # one chunk per FOV
    assert arr.metadata.dimension_names == ZARR_DIMENSION_NAMES
    np.testing.assert_array_equal(arr[:], stack)


def test_write_fov_array_idempotent(tmp_path: Path) -> None:
    """Second write with same shape returns without re-compressing."""
    stack = _synthetic_stack(seed=1)
    group = open_plate_group(tmp_path / "plate.zarr")

    write_fov_array(group, "fov_a", stack)
    write_fov_array(group, "fov_a", stack)  # should not raise

    arr = group["fov_a"]
    np.testing.assert_array_equal(arr[:], stack)


def test_write_fov_array_rejects_shape_change(tmp_path: Path) -> None:
    group = open_plate_group(tmp_path / "plate.zarr")
    write_fov_array(group, "fov_a", _synthetic_stack())
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_fov_array(group, "fov_a", _synthetic_stack(c=5, h=9, w=12))


def test_write_fov_array_requires_3d_uint16(tmp_path: Path) -> None:
    group = open_plate_group(tmp_path / "plate.zarr")

    with pytest.raises(ValueError, match="3-D"):
        write_fov_array(group, "fov_a", np.zeros((5, 8), dtype=np.uint16))

    with pytest.raises(ValueError, match="uint16"):
        write_fov_array(group, "fov_a", np.zeros((5, 8, 8), dtype=np.uint8))


def test_two_fovs_in_same_store_share_metadata(tmp_path: Path) -> None:
    group = open_plate_group(tmp_path / "plate.zarr")
    write_fov_array(group, "fov_a", _synthetic_stack(seed=1))
    write_fov_array(group, "fov_b", _synthetic_stack(seed=2))

    reopened = zarr.open_group(store=str(tmp_path / "plate.zarr"), mode="r")
    assert {"fov_a", "fov_b"} <= set(reopened.array_keys())
    for fov_id in ("fov_a", "fov_b"):
        arr = reopened[fov_id]
        assert arr.metadata.dimension_names == ZARR_DIMENSION_NAMES
        assert arr.dtype == np.uint16


def test_reopen_same_store_appends_not_overwrites(tmp_path: Path) -> None:
    """Simulates two Snakemake jobs writing FOVs into the same plate store."""
    store = tmp_path / "plate.zarr"

    g1 = open_plate_group(store)
    write_fov_array(g1, "fov_a", _synthetic_stack(seed=1))

    g2 = open_plate_group(store)  # second opener, as if from another job
    write_fov_array(g2, "fov_b", _synthetic_stack(seed=2))

    reopened = zarr.open_group(store=str(store), mode="r")
    assert {"fov_a", "fov_b"} <= set(reopened.array_keys())


def test_open_plate_group_concurrent_writers(tmp_path: Path) -> None:
    """Multiple threads each writing distinct FOVs into the same store
    must converge to a store where every FOV is present with correct
    content. This is the intra-rule invariant for `download_batch_to_zarr`
    (fans FOV fetches out via ThreadPoolExecutor then writes via the
    caller thread), and the inter-rule invariant for multiple Snakemake
    jobs touching the same plate store — we simulate the former directly.
    """
    import threading

    store = tmp_path / "plate.zarr"
    n_threads = 8
    fovs_per_thread = 4
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            group = open_plate_group(store)
            for j in range(fovs_per_thread):
                stack = np.full((5, 8, 8), i * 100 + j, dtype=np.uint16)
                write_fov_array(group, f"fov_{i}_{j}", stack)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"worker threads raised: {errors}"
    group = zarr.open_group(store=str(store), mode="r")
    arr_keys = set(group.array_keys())
    expected_keys = {f"fov_{i}_{j}" for i in range(n_threads) for j in range(fovs_per_thread)}
    assert arr_keys == expected_keys
    for i in range(n_threads):
        for j in range(fovs_per_thread):
            arr = group[f"fov_{i}_{j}"]
            assert int(arr[0, 0, 0]) == i * 100 + j


# ---------------------------------------------------------------------------
# write_label_array
# ---------------------------------------------------------------------------


def _synthetic_label(h: int = 8, w: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 50, size=(2, h, w), dtype=np.uint32)


def test_label_dimension_names() -> None:
    assert LABEL_DIMENSION_NAMES == ("label", "y", "x")


def test_write_label_array_round_trip(tmp_path: Path) -> None:
    label = _synthetic_label()
    group = open_plate_group(tmp_path / "seg.zarr")

    write_label_array(group, "fov_a", label)

    reopened = zarr.open_group(store=str(tmp_path / "seg.zarr"), mode="r")
    arr = reopened["fov_a"]
    assert arr.metadata.zarr_format == 3
    assert tuple(arr.shape) == label.shape
    assert arr.dtype == np.uint32
    assert arr.chunks == label.shape
    assert arr.metadata.dimension_names == LABEL_DIMENSION_NAMES
    np.testing.assert_array_equal(arr[:], label)


def test_write_label_array_idempotent(tmp_path: Path) -> None:
    label = _synthetic_label(seed=1)
    group = open_plate_group(tmp_path / "seg.zarr")
    write_label_array(group, "fov_a", label)
    write_label_array(group, "fov_a", label)  # should not raise
    np.testing.assert_array_equal(group["fov_a"][:], label)


def test_write_label_array_rejects_shape_change(tmp_path: Path) -> None:
    group = open_plate_group(tmp_path / "seg.zarr")
    write_label_array(group, "fov_a", _synthetic_label(h=8, w=12))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_label_array(group, "fov_a", _synthetic_label(h=9, w=12))


def test_write_label_array_requires_2channel_uint32(tmp_path: Path) -> None:
    group = open_plate_group(tmp_path / "seg.zarr")

    with pytest.raises(ValueError, match=r"\(2, H, W\)"):
        write_label_array(group, "fov_a", np.zeros((3, 8, 8), dtype=np.uint32))

    with pytest.raises(ValueError, match="uint32"):
        write_label_array(group, "fov_a", np.zeros((2, 8, 8), dtype=np.uint16))
