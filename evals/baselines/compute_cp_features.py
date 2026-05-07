"""Compute per-cell CellProfiler-style features with cp_measure_fast.

For each cell in the DL singlecell h5ad's row_key list, decode the cached
crop + mask from `crops_unfiltered/<plate>.parquet`, run cp_measure on
the Nuclei + Cytoplasm compartments (channels 0 and 1 of the stored
mask blob), and emit per-cell features.

Sharding: one parquet shard per `shard_col` value (held-out grouping
axis) — JUMP shards by `obs.batch` (= source_<N>), RxRx1 by `obs.source`
(= HEPG2-NN), rxrx3c by `obs.batch` (= plate_<N>). Resumable: existing
shard parquets are skipped. Within a dataset, plate-tasks for all
remaining shards are dispatched as one flat list to keep all workers
busy regardless of per-shard plate count.

Final h5ad: concat shards in DL row order, set `obs["batch"]` to the
shard_col value so the eval notebook's
`obs.batch == HELD_OUT_VALUE` holds across all three datasets.

Run sequentially:
    pixi run -e featurisation python compute_cp_features.py --dataset rxrx3c
    pixi run -e featurisation python compute_cp_features.py --dataset jump
    pixi run -e featurisation python compute_cp_features.py --dataset rxrx1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Pin numba to 1 thread per process before cp_measure import.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "${DATA_ROOT}"
EVAL_DATA = "${WORKSPACE}/evals/data"
OUT_BASE = Path("${WORKSPACE}/evals/baselines/_data")

PER_CHANNEL_MEAS = ("radial_distribution", "radial_zernikes", "intensity", "texture", "granularity")
SHAPE_MEAS = ("sizeshape", "zernike", "feret")
CORR_MEAS = ("pearson", "manders_fold", "rwc")  # costes is slow; skip

DATASETS = {
    "jump": {
        "n_channels": 5,
        "channel_names": ("DNA", "ER", "RNA", "AGP", "Mito"),
        "crops_dir": f"{ROOT}/jump_training/crops_unfiltered",
        "dl_singlecell": f"{EVAL_DATA}/01_JUMP_DINO_ECFP4_C_singlecell.h5ad",
        "shard_col": "batch",  # source_<N>
    },
    "rxrx1": {
        "n_channels": 6,
        "channel_names": ("DNA", "ER", "RNA", "AGP", "Mito", "BF"),
        "crops_dir": f"{ROOT}/rxrx1_training/crops_unfiltered",
        "dl_singlecell": f"{EVAL_DATA}/13_Rxrx1_DINO_ESM2_C_singlecell.h5ad",
        "shard_col": "source",  # HEPG2-NN; aggregated h5ad's batch axis
    },
    "rxrx3c": {
        "n_channels": 6,
        "channel_names": ("DNA", "ER", "RNA", "AGP", "Mito", "BF"),
        "crops_dir": f"{ROOT}/rxrx3_training/crops_unfiltered",
        "dl_singlecell": f"{EVAL_DATA}/25_Rxrx3C_DINO_ESM2_C_singlecell.h5ad",
        "shard_col": "batch",  # plate_<N>
    },
}


# --- decoding ---


def decode_cell(blob: bytes, n_channels: int) -> np.ndarray:
    return (
        np.frombuffer(blob, dtype=np.uint8)
        .reshape(n_channels, 224, 224)
        .astype(np.float32)
        / 255.0
    )


def decode_mask(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(2, 224, 224)
    return (raw[0] != 0).astype(np.int32), (raw[1] != 0).astype(np.int32)


def _scalarize(v) -> float:
    arr = np.asarray(v).ravel()
    return float("nan") if arr.size == 0 else float(arr[0])


# --- worker ---


_W_CORE = None
_W_CORR = None
_W_N_CHANNELS = None
_W_CHANNEL_NAMES = None


def _worker_init(n_channels: int, channel_names: tuple[str, ...]):
    global _W_CORE, _W_CORR, _W_N_CHANNELS, _W_CHANNEL_NAMES
    warnings.filterwarnings("ignore")
    import cp_measure.bulk as cpb

    _W_CORE = cpb.get_core_measurements()
    _W_CORR = cpb.get_correlation_measurements()
    _W_N_CHANNELS = n_channels
    _W_CHANNEL_NAMES = channel_names


def _featurize_one(image: np.ndarray, nuc_lbl: np.ndarray, cyto_lbl: np.ndarray) -> dict:
    out: dict[str, float] = {}
    n_ch = len(_W_CHANNEL_NAMES)
    for cname, lbl in (("Nuclei", nuc_lbl), ("Cytoplasm", cyto_lbl)):
        if lbl.sum() == 0:
            continue
        for ci, ch_name in enumerate(_W_CHANNEL_NAMES):
            pix = np.ascontiguousarray(image[ci])
            for meas in PER_CHANNEL_MEAS:
                try:
                    d = _W_CORE[meas](lbl, pix)
                except Exception:
                    d = {}
                for k, v in d.items():
                    out[f"{cname}_{k}_{ch_name}"] = _scalarize(v)
        for meas in SHAPE_MEAS:
            try:
                d = _W_CORE[meas](lbl, None)
            except Exception:
                d = {}
            for k, v in d.items():
                out[f"{cname}_{k}"] = _scalarize(v)
        for ci in range(n_ch):
            for cj in range(ci + 1, n_ch):
                c1, c2 = _W_CHANNEL_NAMES[ci], _W_CHANNEL_NAMES[cj]
                p1 = np.ascontiguousarray(image[ci])
                p2 = np.ascontiguousarray(image[cj])
                for meas in CORR_MEAS:
                    try:
                        d = _W_CORR[meas](p1, p2, lbl)
                    except Exception:
                        d = {}
                    for k, v in d.items():
                        out[f"{cname}_{k}_{c1}_{c2}"] = _scalarize(v)
    return out


def _process_plate(args):
    """Per-task: featurize all needed cells in one parquet plate, write a
    per-plate output parquet directly. Returns small metadata only so the
    main process never accumulates feature dicts in memory.

    Returns: dict with keys plate_out_path, n_cells, n_missing, skipped.
    """
    import gc

    parquet_path, needed_row_keys, plate_out_path = args
    plate_out_path = Path(plate_out_path)
    if plate_out_path.exists():
        try:
            existing = pq.ParquetFile(plate_out_path)
            return {
                "plate_out_path": str(plate_out_path),
                "n_cells": existing.metadata.num_rows,
                "n_missing": 0,
                "skipped": True,
            }
        except Exception:
            plate_out_path.unlink(missing_ok=True)

    pf = pq.ParquetFile(parquet_path)
    tbl = pf.read_row_group(0, columns=["row_key", "cell", "mask"])
    rk_to_idx = {rk: i for i, rk in enumerate(tbl.column("row_key").to_pylist())}
    cell_col = tbl.column("cell")
    mask_col = tbl.column("mask")

    rows: list[dict] = []
    n_missing = 0
    for rk in needed_row_keys:
        i = rk_to_idx.get(rk)
        if i is None:
            n_missing += 1
            rows.append({"row_key": rk})  # all-NaN feature row
            continue
        try:
            image = decode_cell(cell_col[i].as_py(), _W_N_CHANNELS)
            nuc, cyto = decode_mask(mask_col[i].as_py())
            feats = _featurize_one(image, nuc, cyto)
        except Exception as e:
            feats = {"_featurize_error": str(e)[:160]}
            n_missing += 1
        feats["row_key"] = rk
        rows.append(feats)

    # Release table + column refs before the DataFrame build to bound peak.
    del tbl, cell_col, mask_col, rk_to_idx, pf
    gc.collect()

    df = pd.DataFrame(rows)
    plate_out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = plate_out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(plate_out_path)
    n_cells = len(df)
    del df, rows
    gc.collect()

    return {
        "plate_out_path": str(plate_out_path),
        "n_cells": n_cells,
        "n_missing": n_missing,
        "skipped": False,
    }


# --- driver ---


def parquet_path_for_row_key(row_key: str, crops_dir: str) -> str:
    """row_key = '<source>__<batch_name>__<plate>__<well>__<tile>__<id_local>'.

    Parquet filename = '<source>__<batch_name>__<plate>.parquet'.
    """
    parts = row_key.split("__")
    if len(parts) < 6:
        raise ValueError(f"unexpected row_key shape: {row_key}")
    plate_stem = "__".join(parts[:3])
    return f"{crops_dir}/{plate_stem}.parquet"


def run_dataset(name: str, n_workers: int):
    spec = DATASETS[name]
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] loading DL singlecell h5ad...")
    dl = ad.read_h5ad(spec["dl_singlecell"], backed="r")
    obs = dl.obs.copy()
    if "row_key" not in obs.columns:
        obs["row_key"] = obs.index.astype(str)

    shard_col = spec["shard_col"]
    if shard_col not in obs.columns:
        raise SystemExit(f"shard_col={shard_col} missing from obs")

    shard_values = sorted(obs[shard_col].astype(str).unique().tolist())
    print(f"[{name}] {len(obs)} cells; sharding on '{shard_col}' -> {len(shard_values)} shards")
    for sv in shard_values:
        n = (obs[shard_col].astype(str) == sv).sum()
        print(f"   {sv}: {n} cells")

    # Identify which shards are already on disk (resumable). Remaining shards
    # have their plate tasks dispatched as one flat list across the whole
    # dataset so a small-shard's parallelism doesn't bottleneck the pool.
    remaining_shards: list[str] = []
    skipped_shards: list[str] = []
    for sv in shard_values:
        out_path = out_dir / f"{sv}_cp_singlecell.parquet"
        if out_path.exists():
            skipped_shards.append(sv)
        else:
            remaining_shards.append(sv)
    print(f"[{name}] skipped (existing): {skipped_shards}")
    print(f"[{name}] computing: {remaining_shards}")

    summaries: list[dict] = []
    for sv in skipped_shards:
        existing = pd.read_parquet(out_dir / f"{sv}_cp_singlecell.parquet")
        summaries.append(
            {
                "shard": sv,
                "skipped_existing": True,
                "n_cells": len(existing),
                "n_features": existing.shape[1] - len(obs.columns),
                "elapsed_sec": 0.0,
                "cells_per_sec": float("nan"),
            }
        )

    # Per-plate output dir — workers stream feature data here directly so the
    # main process never holds feature dicts in memory.
    plate_out_dir = out_dir / "_plates"
    plate_out_dir.mkdir(parents=True, exist_ok=True)

    if remaining_shards:
        # Build flat plate-task list across all remaining shards
        plate_to_keys: dict[str, list[str]] = {}
        plate_to_shards: dict[str, set[str]] = {}
        for sv in remaining_shards:
            shard_obs = obs[obs[shard_col].astype(str) == sv]
            for rk in shard_obs["row_key"].astype(str):
                p = parquet_path_for_row_key(rk, spec["crops_dir"])
                plate_to_keys.setdefault(p, []).append(rk)
                plate_to_shards.setdefault(p, set()).add(sv)

        # Build (parquet_path, needed_keys, plate_out_path) tasks
        tasks = []
        plate_name_for_path: dict[str, str] = {}
        for parquet_path, keys in plate_to_keys.items():
            plate_name = Path(parquet_path).stem
            plate_name_for_path[parquet_path] = plate_name
            plate_out_path = plate_out_dir / f"{plate_name}.parquet"
            tasks.append((parquet_path, keys, str(plate_out_path)))
        n_total = sum(len(v) for _, v in plate_to_keys.items())
        print(f"[{name}] flat dispatch: {len(tasks)} parquet tasks, {n_total} cells, {n_workers} workers")

        ctx = mp.get_context("spawn")
        print(f"[{name}] starting pool... JIT compile costs once per worker")
        t0 = time.perf_counter()
        n_done_cells = 0
        n_done_tasks = 0
        # maxtasksperchild recycles workers after N tasks to bound per-worker
        # memory growth (cp_measure / numba / arrow caching).
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(spec["n_channels"], spec["channel_names"]),
            maxtasksperchild=int(os.environ.get("CP_MAX_TASKS_PER_CHILD", "3")),
        ) as pool:
            for meta in pool.imap_unordered(_process_plate, tasks, chunksize=1):
                n_done_cells += meta["n_cells"]
                n_done_tasks += 1
                elapsed = time.perf_counter() - t0
                rate = n_done_cells / elapsed if elapsed > 0 else 0
                print(
                    f"  task {n_done_tasks}/{len(tasks)}: +{meta['n_cells']} cells "
                    f"({'skipped' if meta['skipped'] else 'computed'}) "
                    f"-> {n_done_cells}/{n_total} ({rate:.1f}/s, {elapsed:.0f}s)",
                    flush=True,
                )
        flat_elapsed = time.perf_counter() - t0
        flat_rate = n_total / flat_elapsed if flat_elapsed > 0 else 0
        print(f"[{name}] flat dispatch DONE: {n_total} cells in {flat_elapsed:.0f}s ({flat_rate:.1f} cells/sec)", flush=True)

        # Concatenate per-plate parquets into per-shard parquets.
        # Each plate belongs to exactly one shard (shards partition the cells).
        for sv in remaining_shards:
            shard_obs = obs[obs[shard_col].astype(str) == sv].copy()
            keys = shard_obs["row_key"].astype(str).tolist()
            shard_plates = [
                p for p in plate_to_keys if sv in plate_to_shards[p]
            ]
            parts = []
            for parquet_path in shard_plates:
                plate_name = plate_name_for_path[parquet_path]
                parts.append(pd.read_parquet(plate_out_dir / f"{plate_name}.parquet"))
            feats_df = pd.concat(parts, axis=0, ignore_index=True)
            # Reorder to match shard_obs row order via row_key
            feats_df = feats_df.set_index("row_key", drop=False).loc[keys].reset_index(drop=True)
            assert (feats_df["row_key"].astype(str).values == np.array(keys)).all()
            # Drop row_key from features (will come from shard_obs side)
            feats_only = feats_df.drop(columns=["row_key"])
            out = pd.concat([shard_obs.reset_index(drop=True), feats_only], axis=1)
            out_path = out_dir / f"{sv}_cp_singlecell.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(out_path, index=False)
            n_missing = int(feats_only.isna().all(axis=1).sum())
            summaries.append(
                {
                    "shard": sv,
                    "skipped_existing": False,
                    "n_cells": len(keys),
                    "n_features": feats_only.shape[1],
                    "n_missing": n_missing,
                    "n_total_nan_feats": int(feats_only.isna().sum().sum()),
                    "elapsed_sec": round(flat_elapsed, 1),
                    "cells_per_sec": round(flat_rate, 2),
                    "out_path": str(out_path),
                }
            )
            print(f"[{name}] wrote {sv}: {len(keys)} cells, {feats_only.shape[1]} features, {n_missing} missing")

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "shard_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[{name}] shard summary:\n{summary_df.to_string(index=False)}")
    print(f"[{name}] saved {summary_path}")

    # Final concat -> AnnData with homogenised obs.batch
    print(f"[{name}] concatenating shards into final h5ad...")
    parts = []
    for sv in shard_values:
        p = out_dir / f"{sv}_cp_singlecell.parquet"
        parts.append(pd.read_parquet(p))
    full = pd.concat(parts, axis=0, ignore_index=True)

    # Reorder to match DL obs row order (so obs.index aligns 1:1 with DL)
    full = full.set_index("row_key", drop=False).loc[obs["row_key"].astype(str).values].reset_index(drop=True)
    assert (full["row_key"].astype(str).values == obs["row_key"].astype(str).values).all(), "row_key drift after concat"

    obs_cols = list(obs.columns)
    # Drop overlapping columns from full (we use DL obs as source of truth for metadata)
    feat_only = full.drop(columns=[c for c in full.columns if c in obs_cols], errors="ignore")
    feat_cols = list(feat_only.columns)
    X = feat_only.to_numpy(dtype=np.float32)
    var = pd.DataFrame(index=feat_cols)

    obs_out = obs.set_index("row_key", drop=True).copy()
    obs_out.index = obs_out.index.astype(str)
    obs_out.index.name = None
    # Homogenise: obs.batch = shard_col value (so eval HELD_OUT_VALUE == HELD_OUT_COL=batch works)
    obs_out["batch"] = obs[shard_col].values

    a = ad.AnnData(X=X, obs=obs_out, var=var)
    a.uns["cp_measure_provenance"] = {
        "library": "cp_measure_fast (path-installed editable)",
        "compartments": ["Nuclei", "Cytoplasm"],
        "channels": list(spec["channel_names"]),
        "core_measurements": list(PER_CHANNEL_MEAS) + list(SHAPE_MEAS),
        "correlation_measurements": list(CORR_MEAS),
        "shard_col": shard_col,
        "n_workers": n_workers,
        "dl_singlecell": spec["dl_singlecell"],
    }
    out_h5 = OUT_BASE / f"{name}_cp_singlecell.h5ad"
    a.write_h5ad(out_h5)
    nan = int(np.isnan(X).sum())
    print(f"[{name}] WROTE {out_h5} shape={a.shape} dtype={X.dtype} NaN={nan} ({nan/X.size*100:.4f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("CP_WORKERS", "20")))
    args = p.parse_args()
    run_dataset(args.dataset, n_workers=args.workers)


if __name__ == "__main__":
    main()
