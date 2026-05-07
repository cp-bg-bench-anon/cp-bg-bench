"""Diameter estimation for rule C (calibrate).

Samples FOVs per source from the rule-A parquet, runs Cellpose-SAM's
built-in SizeModel on the nucleus channel (DNA) and a two-channel cytosol
stack (channels_for_cell), aggregates per-source medians, and writes a
three-file calibration report:

- ``{config_hash}.yml`` — suggested ``per_source_diameters:`` block ready
  to paste into ``config/jump.yml``.
- ``{config_hash}.md`` — Markdown narrative with per-source medians + IQR.
- ``{config_hash}.png`` — per-source diameter distribution violin plots.

Cellpose, Zarr, and Matplotlib are imported lazily so this module stays
importable in environments without the ``seg-cpu`` / ``seg-gpu`` features.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cp_bg_bench.io.zarr_io import plate_store_path

logger = logging.getLogger(__name__)

__all__ = [
    "CalibrationQualityError",
    "compute_config_hash",
    "run_calibration",
    "sample_fovs",
]


class CalibrationQualityError(RuntimeError):
    """Raised when too many sampled FOVs contain no detectable cells."""


def compute_config_hash(
    data_source_stem: str,
    fov_ids: list[str],
    channels_for_nucleus: list[int],
    channels_for_cell: list[int],
    fovs_per_source: int,
    random_seed: int,
) -> str:
    """12-hex SHA-256 over the deterministic sampling key.

    Reruns with identical inputs hit the same hash (idempotent reports);
    changes in any input produce a new hash without touching old reports.
    """
    payload = {
        "data_source_stem": data_source_stem,
        "fov_ids": sorted(fov_ids),
        "channels_for_nucleus": channels_for_nucleus,
        "channels_for_cell": channels_for_cell,
        "fovs_per_source": fovs_per_source,
        "random_seed": random_seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def sample_fovs(
    meta_df: pd.DataFrame,
    fovs_per_source: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Sample up to ``fovs_per_source`` FOV rows per Metadata_Source.

    Returns ``{source_name: sampled_rows_df}``. If a source has fewer
    rows than ``fovs_per_source``, all rows are used and a warning is
    logged. Deterministic: same ``seed`` always produces the same sample.

    Required columns: ``id``, ``Metadata_Source``, ``Metadata_Batch``,
    ``Metadata_Plate``.
    """
    required = {"id", "Metadata_Source", "Metadata_Batch", "Metadata_Plate"}
    missing = required - set(meta_df.columns)
    if missing:
        raise KeyError(f"meta_df missing required columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    result: dict[str, pd.DataFrame] = {}

    for source, group in meta_df.groupby("Metadata_Source", sort=True):
        n = len(group)
        if n < fovs_per_source:
            logger.warning(
                "source %r: only %d FOVs available, using all (fovs_per_source=%d)",
                source,
                n,
                fovs_per_source,
            )
            sampled = group
        else:
            idx = rng.choice(n, size=fovs_per_source, replace=False)
            sampled = group.iloc[sorted(idx)]
        result[str(source)] = sampled.reset_index(drop=True)

    return result


def _load_fov_stack(store_path: Path, fov_id: str) -> np.ndarray:
    """Load the full ``(C, H, W)`` uint16 FOV array from a per-plate Zarr v3 store."""
    import zarr

    group = zarr.open_group(store=str(store_path), mode="r", zarr_format=3)
    return group[fov_id][:]


def _make_cellpose_model(model_type: str, use_gpu: bool) -> Any:
    """Lazy-import Cellpose high-level model."""
    from cellpose import models

    return models.Cellpose(model_type=model_type, gpu=use_gpu)


def _estimate_diameter(
    model: Any,
    img: np.ndarray,
    cellpose_channels: list[int],
) -> tuple[float, bool]:
    """Run Cellpose SizeModel estimation on a single image.

    ``img`` is either ``(H, W)`` (grayscale) or ``(H, W, C)``
    (multi-channel, 1-indexed channel convention).

    Returns ``(estimated_diameter_px, cells_found)``.
    """
    masks_list, _, _, diams = model.eval([img], diameter=None, channels=cellpose_channels)
    masks = masks_list[0]
    cells_found = bool(np.any(masks > 0))
    return float(diams[0]), cells_found


def _build_yml(per_source_diameters: dict[str, dict[str, float]]) -> str:
    import yaml

    return yaml.safe_dump({"per_source_diameters": per_source_diameters}, sort_keys=True)


def _build_md(
    per_source_stats: dict[str, dict[str, Any]],
    default_nucleus: float,
    default_cytosol: float,
) -> str:
    lines = [
        "# Calibration report",
        "",
        f"Defaults — nucleus: {default_nucleus:.1f} px, cytosol: {default_cytosol:.1f} px",
        "",
    ]
    for source, s in sorted(per_source_stats.items()):
        lines += [
            f"## {source}",
            f"- FOVs sampled: {s['n_total']}, successful: {s['n_ok']}",
            f"- Nucleus  — median {s['nuc_median']:.1f} px, IQR {s['nuc_iqr']:.1f} px",
            f"- Cytosol  — median {s['cell_median']:.1f} px, IQR {s['cell_iqr']:.1f} px",
            "",
        ]
    return "\n".join(lines)


def _build_png(per_source_stats: dict[str, dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    sources = sorted(per_source_stats)
    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
    for ax, source in zip(axes[0], sources, strict=True):
        s = per_source_stats[source]
        nuc_diams, cell_diams = s["nuc_diams"], s["cell_diams"]
        if len(nuc_diams) >= 2 and len(cell_diams) >= 2:
            ax.violinplot([nuc_diams, cell_diams], positions=[1, 2], showmedians=True)
        else:
            ax.scatter([1] * len(nuc_diams), nuc_diams, alpha=0.7)
            ax.scatter([2] * len(cell_diams), cell_diams, alpha=0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["nucleus", "cytosol"])
        ax.set_ylabel("diameter (px)")
        ax.set_title(source)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def run_calibration(
    meta_df: pd.DataFrame,
    output_root: Path,
    seg_config: Any,
    cal_config: Any,
    data_source_stem: str,
    use_gpu: bool = False,
) -> tuple[str, Path]:
    """Run the full calibration pipeline and write the three-file report.

    Parameters
    ----------
    meta_df:
        Rule-A parquet as a DataFrame.
    output_root:
        Pipeline output root (``PipelineConfig.output_root``).
    seg_config:
        ``SegmentationConfig`` from the data-source config.
    cal_config:
        ``CalibrateConfig`` from the global config.
    data_source_stem:
        Filename stem of the data-source config (e.g. ``"smoke"``).
    use_gpu:
        Whether to initialise Cellpose with GPU support.

    Returns
    -------
    (config_hash, output_dir)
        ``config_hash`` is the 12-hex key used in all three output filenames;
        ``output_dir`` is ``output_root / "calibration"``.
    """
    channels_for_nucleus: list[int] = list(seg_config.channels_for_nucleus)
    channels_for_cell: list[int] = list(seg_config.channels_for_cell)

    sampled = sample_fovs(meta_df, cal_config.fovs_per_source, cal_config.random_seed)
    all_fov_ids = [fov_id for df in sampled.values() for fov_id in df["id"].tolist()]

    config_hash = compute_config_hash(
        data_source_stem=data_source_stem,
        fov_ids=all_fov_ids,
        channels_for_nucleus=channels_for_nucleus,
        channels_for_cell=channels_for_cell,
        fovs_per_source=cal_config.fovs_per_source,
        random_seed=cal_config.random_seed,
    )

    output_dir = output_root / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _make_cellpose_model(seg_config.model, use_gpu)

    # Cellpose channel conventions (1-indexed):
    #   [0, 0] = grayscale (nucleus only)
    #   [1, 2] = channel-1 is cytoplasm, channel-2 is nucleus
    nuc_cp_channels = [0, 0]
    cell_cp_channels = [1, 2] if len(channels_for_cell) >= 2 else [0, 0]

    per_source_stats: dict[str, dict[str, Any]] = {}
    per_source_diameters: dict[str, dict[str, float]] = {}

    for source, source_df in sampled.items():
        nuc_diams: list[float] = []
        cell_diams: list[float] = []
        failed_fovs: list[str] = []

        for _, row in source_df.iterrows():
            fov_id = row["id"]
            store_path = plate_store_path(
                output_root,
                source=str(row["Metadata_Source"]),
                batch=str(row["Metadata_Batch"]),
                plate=str(row["Metadata_Plate"]),
            )

            full_stack = _load_fov_stack(store_path, fov_id)  # (C, H, W) uint16
            nuc_img = full_stack[channels_for_nucleus[0]]  # (H, W)

            if len(channels_for_cell) >= 2:
                # (H, W, 2): cellpose expects [cyto, nuc] as last axis
                cell_img = np.stack([full_stack[c] for c in channels_for_cell], axis=-1)
            else:
                cell_img = full_stack[channels_for_cell[0]]  # (H, W)

            nuc_diam, nuc_ok = _estimate_diameter(model, nuc_img, nuc_cp_channels)
            cell_diam, cell_ok = _estimate_diameter(model, cell_img, cell_cp_channels)

            if not (nuc_ok and cell_ok):
                failed_fovs.append(fov_id)
                logger.warning("source %r: no cells detected in FOV %r", source, fov_id)
                continue

            nuc_diams.append(nuc_diam)
            cell_diams.append(cell_diam)

        n_total = len(source_df)
        n_ok = len(nuc_diams)
        success_rate = n_ok / n_total if n_total > 0 else 0.0

        if success_rate < cal_config.min_success_fraction:
            raise CalibrationQualityError(
                f"source {source!r}: cell detection succeeded in only "
                f"{n_ok}/{n_total} FOVs "
                f"(min_success_fraction={cal_config.min_success_fraction:.0%}). "
                f"Failed FOVs: {failed_fovs}"
            )

        nuc_arr = np.array(nuc_diams)
        cell_arr = np.array(cell_diams)
        nuc_median = float(np.median(nuc_arr))
        cell_median = float(np.median(cell_arr))
        nuc_iqr = float(np.percentile(nuc_arr, 75) - np.percentile(nuc_arr, 25))
        cell_iqr = float(np.percentile(cell_arr, 75) - np.percentile(cell_arr, 25))

        per_source_stats[source] = {
            "nuc_diams": nuc_diams,
            "cell_diams": cell_diams,
            "nuc_median": nuc_median,
            "cell_median": cell_median,
            "nuc_iqr": nuc_iqr,
            "cell_iqr": cell_iqr,
            "n_ok": n_ok,
            "n_total": n_total,
        }
        per_source_diameters[source] = {
            "nucleus": round(nuc_median, 1),
            "cytosol": round(cell_median, 1),
        }
        logger.info(
            "source %r: nucleus %.1f px, cytosol %.1f px (%d/%d FOVs)",
            source,
            nuc_median,
            cell_median,
            n_ok,
            n_total,
        )

    default = seg_config.default_diameters
    (output_dir / f"{config_hash}.yml").write_text(_build_yml(per_source_diameters))
    (output_dir / f"{config_hash}.md").write_text(
        _build_md(per_source_stats, default.nucleus, default.cytosol)
    )
    _build_png(per_source_stats, output_dir / f"{config_hash}.png")

    return config_hash, output_dir
