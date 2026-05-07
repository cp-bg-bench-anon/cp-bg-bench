"""GPU (CuPy) + CPU (scipy) resize for crop tensors (rule G)."""

from __future__ import annotations

import numpy as np

__all__ = ["resize_cell", "resize_batch"]

N_MASK_CHANNELS = 2  # nearest-neighbour for masks, bilinear for fluorescent channels


def resize_cell(
    arr: np.ndarray,
    out_hw: tuple[int, int],
    n_mask_channels: int = N_MASK_CHANNELS,
) -> np.ndarray:
    """(C, H, W) uint8 → (C, OH, OW) uint8.

    Uses CuPy + cupyx.scipy.ndimage when available; falls back to
    ``scipy.ndimage.zoom`` otherwise (CPU). Mask channels use nearest-neighbour
    interpolation; fluorescent channels use bilinear (order=1).
    """
    if arr.shape[1:] == out_hw:
        return arr.copy()

    try:
        return _resize_cupy(arr, out_hw, n_mask_channels)
    except ImportError:
        return _resize_scipy(arr, out_hw, n_mask_channels)


def _resize_cupy(
    arr: np.ndarray,
    out_hw: tuple[int, int],
    n_mask_channels: int,
) -> np.ndarray:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi

    oh, ow = out_hw
    h, w = arr.shape[1], arr.shape[2]
    zoom = (1.0, 1.0, oh / h, ow / w)

    x = cp.asarray(arr[None])  # (1, C, H, W)
    out = cp.empty((1, arr.shape[0], oh, ow), dtype=cp.uint8)

    m = min(n_mask_channels, arr.shape[0])
    if m > 0:
        out[:, :m] = cndi.zoom(x[:, :m], zoom, order=0, mode="nearest", prefilter=False)
    if arr.shape[0] > m:
        out[:, m:] = cndi.zoom(x[:, m:], zoom, order=1, mode="reflect", prefilter=True)

    result = out[0].get()
    del x, out
    cp.get_default_memory_pool().free_all_blocks()
    return result


def _resize_scipy(
    arr: np.ndarray,
    out_hw: tuple[int, int],
    n_mask_channels: int,
) -> np.ndarray:
    from scipy.ndimage import zoom

    oh, ow = out_hw
    h, w = arr.shape[1], arr.shape[2]
    zy, zx = oh / h, ow / w

    result = np.empty((arr.shape[0], oh, ow), dtype=np.uint8)
    for c in range(arr.shape[0]):
        order = 0 if c < n_mask_channels else 1
        zoomed = zoom(arr[c].astype(np.float32), (zy, zx), order=order)
        result[c] = np.clip(zoomed, 0, 255).astype(np.uint8)
    return result


def resize_batch(
    cell_list: list[bytes],
    in_shape: tuple[int, int, int],
    out_hw: tuple[int, int],
    n_mask_channels: int = N_MASK_CHANNELS,
) -> list[bytes]:
    """Resize a list of serialised ``(C, H, W)`` tensors.

    Uses batched CuPy when available for efficiency; falls back to
    per-cell scipy on CPU.
    """
    expected = int(np.prod(in_shape))

    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi

        C, H, W = in_shape
        oh, ow = out_hw
        zoom = (1.0, 1.0, oh / H, ow / W)

        batch = np.frombuffer(b"".join(cell_list), dtype=np.uint8).reshape(
            len(cell_list), *in_shape
        )
        x = cp.asarray(batch)
        out = cp.empty((len(cell_list), C, oh, ow), dtype=cp.uint8)

        m = min(n_mask_channels, C)
        if m > 0:
            out[:, :m] = cndi.zoom(x[:, :m], zoom, order=0, mode="nearest", prefilter=False)
        if m < C:
            out[:, m:] = cndi.zoom(x[:, m:], zoom, order=1, mode="reflect", prefilter=True)

        result_np = out.get()
        del x, out
        cp.get_default_memory_pool().free_all_blocks()
        return [result_np[i].tobytes(order="C") for i in range(len(cell_list))]

    except (ImportError, MemoryError, RuntimeError):
        # ImportError  → cupy not installed.
        # MemoryError / RuntimeError → GPU OOM (cupy.cuda.memory.OutOfMemoryError
        # inherits from both depending on the cupy version).
        # Free the pool if cupy was already imported before the failure.
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        result = []
        for blob in cell_list:
            arr = np.frombuffer(blob, dtype=np.uint8)
            if arr.size != expected:
                raise ValueError(f"blob size {arr.size} != expected {expected}") from None
            resized = resize_cell(arr.reshape(in_shape), out_hw, n_mask_channels)
            result.append(resized.tobytes(order="C"))
        return result
