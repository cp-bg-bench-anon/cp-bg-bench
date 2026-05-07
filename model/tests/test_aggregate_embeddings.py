"""Regression test for ``scripts/aggregate_embeddings.py``.

The script aggregates single-cell embeddings to well level using
``.mean()`` for features and ``.first()`` for metadata. Without an explicit
sort, ``.first()`` is order-dependent: upstream dataloader order varies
across inference runs, so two runs over the same cells produce identical
mean embeddings but different ``row_key`` / ``tile`` / ``id_local`` values.

This test locks in the determinism guarantee on
``mean_aggregate_per_well``: shuffling the input rows must not change the
output. It also asserts the fail-loud behaviour when no per-cell
determinism key is present, and the pre-flight check that ``obs["batch"]``
must exist before aggregation runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from aggregate_embeddings import (  # noqa: E402
    aggregate_embeddings,
    mean_aggregate_per_well,
)

GROUP_COLS = ["Metadata_Plate", "Metadata_Well", "Metadata_InChIKey"]


def _build_singlecell(seed: int) -> ad.AnnData:
    rng = np.random.default_rng(0)  # features are seed-fixed across calls
    records = []
    for plate in ("p1", "p2"):
        for well in ("A01", "A02", "A03"):
            pert = "EMPTY" if well == "A03" else f"DRUG_{well}"
            for cell_idx in range(4):
                records.append(
                    {
                        "Metadata_Plate": plate,
                        "Metadata_Well": well,
                        "Metadata_InChIKey": pert,
                        "row_key": f"{plate}__{well}__cell{cell_idx:02d}",
                        "tile": cell_idx,
                        "id_local": cell_idx,
                        "batch": plate,
                    }
                )
    obs = pd.DataFrame.from_records(records)
    X = rng.standard_normal((len(obs), 8)).astype(np.float32)
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"f{i}" for i in range(X.shape[1])]
    a.obs_names = a.obs_names.astype(str)
    perm = np.random.default_rng(seed).permutation(a.n_obs)
    return a[perm].copy()


def test_well_aggregation_is_invariant_to_input_row_order():
    a1 = _build_singlecell(seed=1)
    a42 = _build_singlecell(seed=42)

    df_a = mean_aggregate_per_well(a1, GROUP_COLS)
    df_b = mean_aggregate_per_well(a42, GROUP_COLS)

    df_a = df_a.sort_values(GROUP_COLS).reset_index(drop=True)
    df_b = df_b.sort_values(GROUP_COLS).reset_index(drop=True)

    pd.testing.assert_frame_equal(df_a, df_b, check_like=False)


def test_well_aggregation_picks_lexicographically_smallest_row_key():
    a = _build_singlecell(seed=7)
    df = mean_aggregate_per_well(a, GROUP_COLS).sort_values(GROUP_COLS).reset_index(drop=True)
    for _, row in df.iterrows():
        expected_row_key = f"{row['Metadata_Plate']}__{row['Metadata_Well']}__cell00"
        assert row["row_key"] == expected_row_key
        assert row["tile"] == 0
        assert row["id_local"] == 0


def test_well_aggregation_fails_loud_without_determinism_key():
    a = _build_singlecell(seed=0)
    a.obs = a.obs.drop(columns=["row_key", "tile", "id_local"])
    with pytest.raises(KeyError, match="determinism keys"):
        mean_aggregate_per_well(a, GROUP_COLS)


def test_aggregate_embeddings_fails_loud_without_batch_column(tmp_path: Path):
    a = _build_singlecell(seed=0)
    a.obs = a.obs.drop(columns=["batch"])
    input_path = tmp_path / "00_Test_singlecell.h5ad"
    a.write_h5ad(input_path)
    with pytest.raises(KeyError, match=r"obs\[.batch.\] missing"):
        aggregate_embeddings(
            input_path=input_path,
            output_dir=tmp_path,
            group_cols=GROUP_COLS,
            control_col="Metadata_InChIKey",
            control_value="EMPTY",
        )


def test_aggregate_embeddings_fails_loud_on_drifted_batch_count(tmp_path: Path):
    a = _build_singlecell(seed=0)
    # Rxrx3C is allowed at most 9 batches; emit 12 distinct values to simulate
    # an upstream drift to a high-cardinality column.
    a.obs["batch"] = [f"plate_{i % 12}" for i in range(a.n_obs)]
    input_path = tmp_path / "25_Rxrx3C_DINO_ESM2_C_singlecell.h5ad"
    a.write_h5ad(input_path)
    with pytest.raises(ValueError, match="expected at most"):
        aggregate_embeddings(
            input_path=input_path,
            output_dir=tmp_path,
            group_cols=GROUP_COLS,
            control_col="Metadata_InChIKey",
            control_value="EMPTY",
        )
