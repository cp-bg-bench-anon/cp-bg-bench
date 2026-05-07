"""Embed a HuggingFace image dataset with a saved ``Cp_bg_benchModelPredictor``.

Reads a ``datasets.Dataset`` saved to disk (with at minimum a ``cell``
column of raw uint8 image buffers in ``C*H*W`` layout), runs the image
encoder over it in batches, and writes a parquet file of
``(sample_id, embedding[float16, D])`` rows. Float16 keeps the file
small without measurable loss in cosine-retrieval evals.

``--channels`` and ``--image-size`` are optional when the checkpoint was
saved with the current ``save_checkpoint`` (which bakes them in as
metadata). Supply them explicitly only for older checkpoints.

Example::

    pixi run python scripts/embed_dataset.py \\
        --checkpoint runs/foo/predictor.pt \\
        --dataset /path/to/hf_ds \\
        --output embeddings.parquet \\
        --batch-size 256 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm

from cp_bg_bench_model import Cp_bg_benchModelPredictor
from cp_bg_bench_model._constants import DatasetEnum


def _decode_batch(bufs: list, channels: int, hw: int) -> torch.Tensor:
    """Decode raw uint8 image buffers to a ``(B, C, H, W)`` tensor."""
    expected = channels * hw * hw
    out = np.empty((len(bufs), channels, hw, hw), dtype=np.uint8)
    for i, buf in enumerate(bufs):
        if isinstance(buf, (list, tuple)):
            if len(buf) != 1:
                raise ValueError(f"Expected 1 image per row, got len={len(buf)}")
            buf = buf[0]
        if isinstance(buf, memoryview):
            buf = buf.tobytes()
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size != expected:
            raise ValueError(f"Bad image buffer: got {arr.size} bytes, expected {expected}.")
        out[i] = arr.reshape(channels, hw, hw)
    return torch.from_numpy(out)


def _resolve_ids(dataset, id_col: str | None) -> list:
    if id_col is None:
        return list(range(len(dataset)))
    if id_col not in dataset.column_names:
        raise KeyError(f"--id-col {id_col!r} not in dataset. Available: {dataset.column_names}")
    return list(dataset[id_col])


def _require(value: int | None, name: str, meta_key: str, meta: dict) -> int:
    if value is not None:
        return value
    if meta.get(meta_key) is not None:
        return int(meta[meta_key])
    raise ValueError(
        f"{name} not found in checkpoint metadata and not supplied on the command line. "
        f"Pass --{name.replace('_', '-')} explicitly."
    )


def embed_dataset(
    checkpoint: Path,
    dataset_path: Path,
    output: Path,
    *,
    channels: int | None = None,
    image_size: int | None = None,
    batch_size: int = 256,
    device: str = "cpu",
    id_col: str | None = None,
    image_col: str = DatasetEnum.IMG.value,
) -> Path:
    """Embed every row of a HF dataset and write a parquet of ids+embeddings.

    ``channels`` and ``image_size`` default to values baked into the
    checkpoint metadata; explicit arguments override those values.
    """
    predictor = Cp_bg_benchModelPredictor.load(checkpoint, device=device)
    meta = predictor.metadata

    channels = _require(channels, "channels", "in_channels", meta)
    image_size = _require(image_size, "image_size", "image_size", meta)

    dataset = load_from_disk(str(dataset_path))
    if image_col not in dataset.column_names:
        raise KeyError(
            f"Image column {image_col!r} not in dataset. Available: {dataset.column_names}"
        )

    ids = _resolve_ids(dataset, id_col)
    n = len(dataset)

    if n == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        empty = pa.table({"sample_id": pa.array([], type=pa.int64()), "embedding": pa.array([], type=pa.list_(pa.float16()))})
        pq.write_table(empty, str(output))
        return output

    embed_dim_meta = meta.get("embed_dim")
    if embed_dim_meta is not None:
        embed_dim = int(embed_dim_meta)
    else:
        probe = _decode_batch(dataset[0:1][image_col], channels, image_size)
        embed_dim = predictor.predict_batch(probe).shape[1]

    schema = pa.schema([
        ("sample_id", pa.int64()),
        ("embedding", pa.list_(pa.float16(), embed_dim)),
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(output), schema)

    def _write(batch_ids: list, emb: np.ndarray) -> None:
        flat = pa.array(emb.ravel(), type=pa.float16())
        embedding_col = pa.FixedSizeListArray.from_arrays(flat, embed_dim)
        table = pa.table(
            {"sample_id": pa.array(batch_ids, type=pa.int64()), "embedding": embedding_col},
            schema=schema,
        )
        writer.write_table(table)

    bar = tqdm(total=n, unit="cell", desc="embedding")
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = dataset[start:end]
            crops = _decode_batch(batch[image_col], channels, image_size)
            emb = predictor.predict_batch(crops).astype(np.float16)
            _write(ids[start:end], emb)
            bar.update(end - start)
    finally:
        bar.close()
        writer.close()

    return output


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True, help="Path to HF dataset on disk")
    p.add_argument("--output", type=Path, required=True, help="Output parquet path")
    p.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Image channels (inferred from checkpoint metadata if omitted)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Image spatial size (inferred from checkpoint metadata if omitted)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--id-col", default=None, help="Column to use as sample_id; default = row index")
    p.add_argument(
        "--image-col", default=DatasetEnum.IMG.value, help="Image column name (default: cell)"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = embed_dataset(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        output=args.output,
        channels=args.channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        id_col=args.id_col,
        image_col=args.image_col,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
