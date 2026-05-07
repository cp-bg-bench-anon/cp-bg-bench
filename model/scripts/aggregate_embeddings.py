"""Aggregate single-cell embeddings to well-level profiles and integrate.

Takes the AnnData produced by ``embed_to_anndata.py`` (one row per cell,
named ``<config_id>_singlecell.h5ad``) and applies the standard cell-painting
aggregation pipeline:

    1. Mean pool per well  (group by plate + well + perturbation)
    2. Spherize (ZCA whitening) fit on control wells
    3. PCA
    4. Harmony batch correction on ``obs["batch"]``

The ``batch`` column is set by the upstream Snakemake data pipeline
(``data/snakemake/scripts/extract_crops_plate.py``) per dataset:

    JUMP    : batch == source                (source_2..source_10)
    RxRx1   : batch == source                (HEPG2-01..HEPG2-11)
    Rxrx3C  : batch == f"plate_{{plate}}"    (plate_1..plate_9)

Aggregate is dataset-agnostic — it just trusts ``obs["batch"]``.

Output is ``<output_dir>/<config_id>_aggregated.h5ad`` with:
    X        = spherized well-mean embeddings (pre-Harmony)
    obsm     = {"X_pca": PCA coords, "X_pca_harmony": Harmony-corrected coords}
    obs      = well-level metadata + all config provenance columns from step 1
               (config_id, config_num, config_dataset, config_image_encoder,
                config_perturbation_encoder, config_view)

Example::

    pixi run python scripts/aggregate_embeddings.py \\
        --input  ${DATA_ROOT} \\
        --output-dir ${DATA_ROOT} \\
        --group-cols Metadata_Plate,Metadata_Well,Metadata_InChIKey \\
        --control-col Metadata_InChIKey --control-value EMPTY
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
import pycytominer
import scanpy as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Seed for any stochastic step in this script (currently: Harmony k-means
# initialisation). Pinned for reproducibility — re-running aggregation on
# the same singlecell h5ad must produce bit-identical X_pca_harmony.
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

BATCH_COL = "batch"

# Expected ``obs["batch"]`` cardinality per dataset, set upstream by
# ``data/snakemake/scripts/extract_crops_plate.py``:
#   JUMP    → source_2..source_10  (10 production sites)
#   Rxrx1   → HEPG2-01..HEPG2-11   (11 experiments)
#   Rxrx3C  → plate_1..plate_9     (9 plates)
# We only support these three datasets, so a count above the expected
# value almost certainly means the upstream pipeline drifted and is now
# emitting a high-cardinality column (e.g. gene_targets, ~176 values for
# Rxrx3C). Harmony would still run on that input but produce a
# meaningless correction, so we hard-fail here. A count *below* the
# expected value is allowed (partial inference is a real workflow) but
# warned so it does not slip past unnoticed during full eval runs.
_EXPECTED_BATCH_COUNT_BY_DATASET: dict[str, int] = {
    "JUMP": 10,
    "Rxrx1": 11,
    "Rxrx3C": 9,
}


def _detect_dataset(config_id: str) -> str | None:
    """Return the dataset key embedded in *config_id* (e.g. ``13_Rxrx1_DINO_…``)."""
    tokens = config_id.split("_")
    for ds in _EXPECTED_BATCH_COUNT_BY_DATASET:
        if ds in tokens:
            return ds
    return None


def mean_aggregate_per_well(
    adata: ad.AnnData,
    group_cols: list[str],
) -> pd.DataFrame:
    """Mean-aggregate single-cell features to well level with deterministic metadata.

    ``.first()`` on metadata columns is sensitive to input row order, while
    ``.mean()`` on features is not. Upstream dataloader order varies across
    encoder runs, so this function sorts on a per-cell key before grouping to
    keep ``row_key`` / ``tile`` / ``id_local`` stable across runs over the same
    cell set. At least one of those determinism keys must be present in
    ``obs`` — we fail loudly rather than silently sorting on every metadata
    column, which would be slow and produce a different (but still
    deterministic) order per dataset.
    """
    missing = [c for c in group_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"group_cols not found in obs: {missing}")

    feature_cols = list(adata.var_names)
    meta_cols = [c for c in adata.obs.columns if c not in feature_cols]
    meta_extra = [c for c in meta_cols if c not in group_cols]

    determinism_keys = ("row_key", "tile", "id_local")
    sort_keys = [c for c in determinism_keys if c in meta_extra]
    if not sort_keys:
        raise KeyError(
            f"None of the per-cell determinism keys {determinism_keys} found in "
            f"obs columns. Re-run upstream embedding so at least one is present."
        )

    X_dense = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    obs_reset = adata.obs[group_cols + meta_extra].reset_index(drop=True)
    feat_reset = pd.DataFrame(X_dense, columns=feature_cols)
    df = pd.concat([obs_reset, feat_reset], axis=1)
    df = df.sort_values(by=sort_keys, kind="stable")
    # Split into two groupby calls to avoid pandas 2.x mixed-agg regression.
    meta_well = df.groupby(group_cols, observed=True)[meta_extra].first().reset_index()
    feat_well = df.groupby(group_cols, observed=True)[feature_cols].mean().reset_index()
    return meta_well.merge(feat_well, on=group_cols)


def aggregate_embeddings(
    input_path: Path,
    output_dir: Path,
    *,
    group_cols: list[str],
    control_col: str,
    control_value: str,
    n_pcs: int = 50,
) -> Path:
    # ── Derive config_id and output path from input filename ──────────────────
    stem = input_path.stem  # e.g. "13_Rxrx1_DINO_ESM2_C_singlecell"
    if stem.endswith("_singlecell"):
        config_id = stem[: -len("_singlecell")]
    else:
        config_id = stem
        log.warning(
            f"Input filename {input_path.name!r} doesn't end with '_singlecell' — "
            f"using full stem {config_id!r} as config_id."
        )
    output_path = output_dir / f"{config_id}_aggregated.h5ad"

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info(f"Loading single-cell AnnData from {input_path}")
    adata = ad.read_h5ad(input_path)
    log.info(f"  {adata.n_obs:,} cells × {adata.n_vars} dims")
    log.info(f"  config_id: {config_id}")

    if BATCH_COL not in adata.obs.columns:
        raise KeyError(
            f"obs[{BATCH_COL!r}] missing from input. The upstream Snakemake "
            "pipeline (extract_crops_plate.py) is responsible for setting this "
            "column. Re-extract crops and re-run inference."
        )

    dataset = _detect_dataset(config_id)
    n_batches = adata.obs[BATCH_COL].nunique()
    if dataset is not None:
        expected = _EXPECTED_BATCH_COUNT_BY_DATASET[dataset]
        if n_batches > expected:
            raise ValueError(
                f"obs[{BATCH_COL!r}] has {n_batches} unique values for {dataset}, "
                f"expected at most {expected}. Upstream pipeline likely emitted "
                "a high-cardinality column instead of the canonical batch axis "
                "— refusing to run Harmony against a drifted batch definition."
            )
        if n_batches < expected:
            log.warning(
                f"obs[{BATCH_COL!r}] has {n_batches} unique values for {dataset} "
                f"(expected {expected}); proceeding but Harmony correction will "
                "be partial."
            )
    else:
        log.warning(
            f"config_id {config_id!r} did not match any known dataset "
            f"({list(_EXPECTED_BATCH_COUNT_BY_DATASET)}); skipping batch-count "
            "validation."
        )

    feature_cols = list(adata.var_names)

    # ── 1. Mean aggregation per well ──────────────────────────────────────────
    log.info(f"Aggregating to well level, grouping by {group_cols}")
    well_df = mean_aggregate_per_well(adata, group_cols)
    meta_extra = [c for c in adata.obs.columns if c not in feature_cols and c not in group_cols]
    log.info(f"  {len(well_df):,} wells after aggregation")

    # ── 2. Spherize on control wells ──────────────────────────────────────────
    if control_col not in well_df.columns:
        raise KeyError(f"--control-col {control_col!r} not found after aggregation")

    n_ctrl = (well_df[control_col] == control_value).sum()
    if n_ctrl == 0:
        raise ValueError(
            f"No control wells found where {control_col} == {control_value!r}. "
            f"Check --control-col / --control-value."
        )
    log.info(f"Spherizing on {n_ctrl} control wells ({control_col}=={control_value!r})")

    all_meta = group_cols + meta_extra
    spherized = pycytominer.normalize(
        profiles=well_df,
        features=feature_cols,
        meta_features=all_meta,
        samples=f'{control_col} == "{control_value}"',
        method="spherize",
    )

    # ── 3. Build well-level AnnData ────────────────────────────────────────────
    X = spherized[feature_cols].values.astype(np.float32)
    obs = spherized[all_meta].copy().reset_index(drop=True)
    obs.index = obs.index.astype(str)

    well_adata = ad.AnnData(X=X, obs=obs)
    well_adata.obs_names_make_unique()
    well_adata.obs["is_control"] = (well_adata.obs[control_col] == control_value).astype(str)

    # ── 4. PCA ─────────────────────────────────────────────────────────────────
    n_pcs_actual = min(n_pcs, well_adata.n_obs - 1, well_adata.n_vars)
    log.info(f"Running PCA with {n_pcs_actual} components")
    sc.pp.pca(well_adata, n_comps=n_pcs_actual)

    # ── 5. Harmony batch correction on obs["batch"] ────────────────────────────
    n_batches = well_adata.obs[BATCH_COL].nunique()
    log.info(f"Running Harmony on obs[{BATCH_COL!r}] ({n_batches} batches)")
    pcs = np.ascontiguousarray(well_adata.obsm["X_pca"])
    harmony_out = hm.run_harmony(
        pcs,
        well_adata.obs,
        [BATCH_COL],
        max_iter_harmony=100,
        random_state=RANDOM_SEED,
    )
    well_adata.obsm["X_pca_harmony"] = np.asarray(harmony_out.Z_corr).astype(np.float32)

    log.info(f"Well AnnData shape: {well_adata.n_obs:,} × {well_adata.n_vars}")
    log.info(f"obs columns: {list(well_adata.obs.columns)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    well_adata.write_h5ad(output_path)
    log.info(f"Wrote {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="<config_id>_singlecell.h5ad from embed_to_anndata.py")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write <config_id>_aggregated.h5ad into")
    p.add_argument(
        "--group-cols",
        default="Metadata_Plate,Metadata_Well,Metadata_InChIKey",
        help="Comma-separated columns to group by for well aggregation "
        "(default: Metadata_Plate,Metadata_Well,Metadata_InChIKey)",
    )
    p.add_argument(
        "--control-col",
        default="Metadata_InChIKey",
        help="Column identifying control wells (default: Metadata_InChIKey)",
    )
    p.add_argument(
        "--control-value",
        default="EMPTY",
        help="Value in --control-col that marks control wells (default: EMPTY)",
    )
    p.add_argument("--n-pcs", type=int, default=50, help="Number of PCA components (default: 50)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    aggregate_embeddings(
        input_path=args.input,
        output_dir=args.output_dir,
        group_cols=args.group_cols.split(","),
        control_col=args.control_col,
        control_value=args.control_value,
        n_pcs=args.n_pcs,
    )


if __name__ == "__main__":
    main()
