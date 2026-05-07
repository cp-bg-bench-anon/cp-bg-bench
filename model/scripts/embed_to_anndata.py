"""Embed a HF image dataset with a saved ImageEncoderWithHead and write the
results as an AnnData h5ad file.

    X   = (n_cells, embed_dim) float32, L2-normalised image embeddings
    obs = all dataset metadata columns + config provenance columns

Config provenance columns added to every obs row:

    config_id               full experiment identifier, e.g. 13_Rxrx1_DINO_ESM2_C
    config_num              numeric part, e.g. 13
    config_dataset          dataset name, e.g. Rxrx1
    config_image_encoder    image encoder name, e.g. DINO
    config_perturbation_encoder  perturbation encoder name, e.g. ESM2
    config_view             view / channel set, e.g. C, CD, S, SD

Output is named ``<output_dir>/<config_id>_singlecell.h5ad``.

Stratification: for each unique value of ``--stratify-col`` (default:
``plate``), sample ``fraction`` of cells.  Plate is the canonical batch unit
in cell-painting data (each plate is seeded, stained, and imaged
independently).  Stratifying by plate ensures every plate contributes
proportionally to the sample, which is essential for downstream batch-effect
analysis and UMAP visualisation.

Pass the ``imgenc_finetuned`` export from the training script to
``--checkpoint``; the loader resolves the right artifact automatically.
Raw uint8 image bytes are converted to float32 in [0, 1] before being
fed to the encoder; per-channel affine normalisation (if any) is applied
inside the encoder itself.

Example (RxRx1 crops, config 13)::

    pixi run python scripts/embed_to_anndata.py \\
        --config-id 13_Rxrx1_DINO_ESM2_C \\
        --checkpoint results/pretrain/13_Rxrx1_DINO_ESM2_C_202604201316/exports/imgenc_finetuned \\
        --dataset ${DATA_ROOT}/rxrx1_training/datasets/crops_resharded \\
        --output-dir ${DATA_ROOT}/results/embeddings \\
        --channels 6 --image-size 224 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm.auto import tqdm

from cp_bg_bench_model.models._export import load_image_encoder_with_head

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Seed for stratified sampling and any other stochastic op invoked from this
# script. The CLI ``--seed`` flag still allows overrides for ad-hoc subsamples;
# this constant is the default when none is passed.
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_config_components(config_id: str) -> dict[str, str]:
    """Parse '13_Rxrx1_DINO_ESM2_C' into its five named components.

    Expected pattern: {num}_{Dataset}_{ImageEncoder}_{PertEncoder}_{View}
    Returns an empty dict if the config_id doesn't match the pattern.
    """
    parts = config_id.split("_")
    if len(parts) == 5:
        return {
            "config_num": parts[0],
            "config_dataset": parts[1],
            "config_image_encoder": parts[2],
            "config_perturbation_encoder": parts[3],
            "config_view": parts[4],
        }
    log.warning(
        f"config_id {config_id!r} doesn't match expected pattern "
        "'num_Dataset_ImgEncoder_PertEncoder_View' — storing as-is only."
    )
    return {}


def _stratified_indices(
    keys: list[str],
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Return sorted row indices keeping ``fraction`` of cells per unique key."""
    rng = np.random.default_rng(seed)
    series = pd.Series(keys)
    parts: list[np.ndarray] = []
    for _, grp in series.groupby(series, sort=False):
        n = max(1, round(len(grp) * fraction))
        chosen = rng.choice(grp.index.to_numpy(), size=n, replace=False)
        parts.append(chosen)
    return np.sort(np.concatenate(parts))


def _decode_batch(
    bufs: list,
    channels: int,
    hw: int,
) -> torch.Tensor:
    """Decode raw uint8 image buffers to a ``(B, C, H, W)`` tensor."""
    expected = channels * hw * hw
    out = np.empty((len(bufs), channels, hw, hw), dtype=np.uint8)
    for i, buf in enumerate(bufs):
        if isinstance(buf, (list, tuple)):
            buf = buf[0]
        if isinstance(buf, memoryview):
            buf = buf.tobytes()
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size != expected:
            raise ValueError(
                f"Row {i}: image buffer has {arr.size} bytes, "
                f"expected {expected} ({channels}ch × {hw}×{hw})."
            )
        out[i] = arr.reshape(channels, hw, hw)
    return torch.from_numpy(out)


# ── Main ──────────────────────────────────────────────────────────────────────


def embed_to_anndata(
    config_id: str,
    checkpoint: Path,
    dataset_path: Path,
    output_dir: Path,
    *,
    fraction: float = 1.0,
    stratify_col: str = "plate",
    channels: int = 6,
    image_size: int = 224,
    model_image_size: int | None = None,
    batch_size: int = 256,
    device: str = "cuda",
    seed: int = RANDOM_SEED,
    image_col: str = "cell",
) -> Path:
    output = output_dir / f"{config_id}_singlecell.h5ad"
    _model_sz = model_image_size or image_size
    _needs_resize = _model_sz != image_size
    if _needs_resize:
        from torchvision.transforms import Resize
        _resize = Resize((_model_sz, _model_sz), antialias=True)
        log.info(f"Will resize {image_size}→{_model_sz} to match model input")

    # ── Load encoder ─────────────────────────────────────────────────────────
    log.info(f"Loading encoder from {checkpoint}")
    encoder = load_image_encoder_with_head(checkpoint)
    encoder.eval().to(device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    log.info(f"Loading dataset from {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    # Exclude image_col and any binary-typed columns (e.g. raw mask bytes)
    # — h5py's VLEN strings reject embedded NULLs, so binary blobs can't go in obs.
    _BINARY_DTYPES = {"binary", "large_binary"}
    meta_cols = [
        c for c in ds.column_names if c != image_col and getattr(ds.features[c], "dtype", None) not in _BINARY_DTYPES
    ]
    log.info(f"  {len(ds):,} cells, {len(meta_cols)} metadata columns")

    if stratify_col not in ds.column_names:
        raise KeyError(
            f"--stratify-col {stratify_col!r} not found in dataset. "
            f"Available: {ds.column_names}"
        )

    # ── Stratified sample ─────────────────────────────────────────────────────
    log.info(f"Sampling {fraction*100:.0f}% of cells stratified by '{stratify_col}' (seed={seed})")
    indices = _stratified_indices(ds[stratify_col], fraction=fraction, seed=seed)
    log.info(f"  Selected {len(indices):,} / {len(ds):,} cells")

    # ── Single-pass: embed + collect metadata ─────────────────────────────────
    emb_parts: list[np.ndarray] = []
    obs_parts: dict[str, list] = {c: [] for c in meta_cols}

    bar = tqdm(total=len(indices), unit="cell", desc="embedding", dynamic_ncols=True)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size].tolist()
            batch = ds[batch_idx]

            for c in meta_cols:
                v = batch[c]
                obs_parts[c].extend(v if isinstance(v, list) else [v])

            imgs = _decode_batch(batch[image_col], channels, image_size)
            imgs = imgs.to(dtype=torch.float32, device=device)
            imgs.div_(255.0)
            if _needs_resize:
                imgs = _resize(imgs)

            emb = encoder(imgs)
            emb = F.normalize(emb, p=2, dim=-1)
            emb_parts.append(emb.cpu().float().numpy())

            bar.update(len(batch_idx))
    bar.close()

    # ── Build AnnData ──────────────────────────────────────────────────────────
    X = np.concatenate(emb_parts, axis=0)
    obs_df = pd.DataFrame(obs_parts)

    index_col = "row_key" if "row_key" in obs_df.columns else None
    if index_col:
        obs_df.index = obs_df[index_col].astype(str)
    else:
        obs_df.index = obs_df.index.astype(str)

    # Config provenance — constant columns so they survive concat / subsetting
    obs_df["config_id"] = config_id
    components = _parse_config_components(config_id)
    for k, v in components.items():
        obs_df[k] = v

    adata = ad.AnnData(X=X, obs=obs_df)
    adata.obs_names_make_unique()

    log.info(f"AnnData shape: {adata.shape[0]:,} cells × {adata.shape[1]} dims")
    log.info(f"obs columns: {list(adata.obs.columns)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    log.info(f"Wrote {output}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config-id",
        required=True,
        help="Experiment identifier, e.g. 13_Rxrx1_DINO_ESM2_C. "
             "Used to name the output file and stored in obs.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the imgenc_finetuned export (any artifact or shared stem).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to HF dataset on disk",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write <config_id>_singlecell.h5ad into",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of cells to embed per stratification group (default: 1.0)",
    )
    p.add_argument(
        "--stratify-col",
        default="plate",
        help="Column to stratify on (default: plate)",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=6,
        help="Number of fluorescence channels after skipping (default: 6)",
    )
    p.add_argument("--image-size", type=int, default=224,
                   help="On-disk crop size (default: 224)")
    p.add_argument("--model-image-size", type=int, default=None,
                   help="Model input size; resize after decode if different from --image-size")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--image-col",
        default="cell",
        help="HF dataset column containing raw image bytes (default: cell)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    embed_to_anndata(
        config_id=args.config_id,
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        fraction=args.fraction,
        stratify_col=args.stratify_col,
        channels=args.channels,
        image_size=args.image_size,
        model_image_size=args.model_image_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        image_col=args.image_col,
    )


if __name__ == "__main__":
    main()
