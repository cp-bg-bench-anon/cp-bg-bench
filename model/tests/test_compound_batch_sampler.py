"""Tests for PerturbationBatchSampler with and without row_strata."""
import numpy as np
import pytest
import torch

from cp_bg_bench_model.datamodule import PerturbationBatchSampler


def _make_sampler(compound_ids, row_strata=None, batch_size=8, perturbations_per_batch=2, seed=0):
    return PerturbationBatchSampler(
        np.asarray(compound_ids),
        batch_size=batch_size,
        perturbations_per_batch=perturbations_per_batch,
        generator=torch.Generator().manual_seed(seed),
        row_strata=np.asarray(row_strata) if row_strata is not None else None,
    )


# ── baseline: no strata ──────────────────────────────────────────────────────

def test_no_strata_yields_correct_batch_size():
    # 4 compounds × 10 rows each, batch=8, 2 cpd/batch → 4 rows/cpd
    ids = np.repeat([0, 1, 2, 3], 10)
    sampler = _make_sampler(ids)
    batches = list(sampler)
    assert all(len(b) == 8 for b in batches)


def test_no_strata_all_indices_valid():
    ids = np.repeat([0, 1, 2, 3], 10)
    sampler = _make_sampler(ids)
    for batch in sampler:
        assert all(0 <= i < len(ids) for i in batch)


# ── with strata ──────────────────────────────────────────────────────────────

def _compound_ids_multi_stratum():
    """4 compounds, each with rows from 4 experiments (strata 0-3), 3 rows/stratum."""
    return np.repeat([0, 1, 2, 3], 12), np.tile(np.repeat([0, 1, 2, 3], 3), 4)


def test_strata_yields_correct_batch_size():
    ids, strata = _compound_ids_multi_stratum()
    sampler = _make_sampler(ids, row_strata=strata, batch_size=8, perturbations_per_batch=2)
    assert all(len(b) == 8 for b in sampler)


def test_strata_multiple_experiments_per_compound():
    """Each compound's 4 sampled rows should come from at least 2 distinct strata."""
    ids, strata = _compound_ids_multi_stratum()
    # batch=8, 2 cpd/batch → 4 rows per compound; 4 strata available → expect ≥2
    sampler = _make_sampler(ids, row_strata=strata, batch_size=8, perturbations_per_batch=2)
    for batch in sampler:
        batch = np.asarray(batch)
        for cpd in range(4):
            cpd_rows = batch[np.isin(ids[batch], [cpd])]
            if len(cpd_rows) > 0:
                n_strata = len(np.unique(strata[cpd_rows]))
                assert n_strata >= 2, f"compound {cpd}: only {n_strata} stratum represented"


def test_strata_imbalanced_compound_single_stratum():
    """Compound with only 1 stratum should still work without error."""
    # cpd 0: rows 0-9 all stratum 0
    # cpd 1: rows 10-19 distributed across strata 0,1,2,3
    ids = np.array([0] * 10 + [1] * 12)
    strata = np.array([0] * 10 + [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    sampler = _make_sampler(ids, row_strata=strata, batch_size=8, perturbations_per_batch=2)
    batches = list(sampler)
    assert all(len(b) == 8 for b in batches)


def test_strata_backward_compat_no_arg():
    """Omitting row_strata should give the same behavior as before (no strata pools)."""
    ids = np.repeat([0, 1, 2, 3], 10)
    sampler = _make_sampler(ids)
    assert sampler._strata_pools is None
    assert all(len(b) == 8 for b in sampler)
