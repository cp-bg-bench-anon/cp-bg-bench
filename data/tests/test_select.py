"""Tests for cp_bg_bench.selection (rule E)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cp_bg_bench.selection import select_cells
from cp_bg_bench.selection.uniform import (
    select_uniform_per_compound_source,
    select_uniform_per_well,
    select_uniform_total,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_seg_df(
    n_wells: int = 3,
    cells_per_well: int = 10,
    plates: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(plates):
        plate = f"plate{p}"
        for w in range(n_wells):
            well = f"A{w:02d}"
            for s in range(cells_per_well):
                fov_id = f"src__bat__{plate}__{well}__1"
                rows.append(
                    {
                        "fov_id": fov_id,
                        "id_local": s + 1,
                        "nuc_id_global": s + 1,
                        "cell_id_global": s + 1,
                        "cyto_cent_row": int(rng.integers(10, 90)),
                        "cyto_cent_col": int(rng.integers(10, 90)),
                        "nuc_cent_row": int(rng.integers(10, 90)),
                        "nuc_cent_col": int(rng.integers(10, 90)),
                        "nuc_area": int(rng.integers(50, 200)),
                        "cyto_area": int(rng.integers(200, 800)),
                        "nuc_cyto_ratio": float(rng.uniform(0.1, 0.5)),
                        "n_cells_in_fov": cells_per_well,
                    }
                )
    return pd.DataFrame(rows)


def _make_meta_parquet(tmp_path: Path, fov_ids: list[str]) -> Path:
    rows = []
    for fov_id in fov_ids:
        rows.append(
            {
                "id": fov_id,
                "Metadata_JCP2022": "JCP2022_xyz",
                "Metadata_InChIKey": "INCHI_KEY",
                "Metadata_InChI": "InChI=1S/test",
                "Metadata_PlateType": "COMPOUND",
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    path = tmp_path / "selected_metadata.parquet"
    df.to_parquet(path, index=False)
    return path


# ── TestUniformSelection ──────────────────────────────────────────────────────


class TestUniformSelection:
    def test_caps_at_cells_per_well(self) -> None:
        df = _make_seg_df(n_wells=2, cells_per_well=20)
        result = select_uniform_per_well(df, cells_per_well=5, seed=0)
        # Each (plate_key, well) group should have ≤ 5 cells
        parts = result["fov_id"].str.split("__", n=4, expand=True)
        result["_pk"] = parts[0] + "__" + parts[1] + "__" + parts[2]
        result["_well"] = parts[3]
        for _, grp in result.groupby(["_pk", "_well"]):
            assert len(grp) <= 5

    def test_keeps_all_when_fewer_than_limit(self) -> None:
        df = _make_seg_df(n_wells=1, cells_per_well=3)
        result = select_uniform_per_well(df, cells_per_well=10, seed=0)
        assert len(result) == 3

    def test_deterministic(self) -> None:
        df = _make_seg_df(n_wells=2, cells_per_well=20)
        r1 = select_uniform_per_well(df, cells_per_well=5, seed=42)
        r2 = select_uniform_per_well(df, cells_per_well=5, seed=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_different_seeds_give_different_results(self) -> None:
        df = _make_seg_df(n_wells=1, cells_per_well=50)
        r1 = select_uniform_per_well(df, cells_per_well=10, seed=1)
        r2 = select_uniform_per_well(df, cells_per_well=10, seed=2)
        assert not r1["id_local"].equals(r2["id_local"])

    def test_no_extra_columns(self) -> None:
        df = _make_seg_df()
        result = select_uniform_per_well(df, cells_per_well=5, seed=0)
        assert "_well" not in result.columns
        assert "_plate_key" not in result.columns


# ── TestUniformTotal ──────────────────────────────────────────────────────────


class TestUniformTotal:
    def test_caps_at_max_cells(self) -> None:
        df = _make_seg_df(n_wells=3, cells_per_well=20)
        result = select_uniform_total(df, max_cells=10, seed=0)
        assert len(result) == 10

    def test_returns_all_when_fewer_available(self) -> None:
        df = _make_seg_df(n_wells=1, cells_per_well=5)
        result = select_uniform_total(df, max_cells=100, seed=0)
        assert len(result) == 5

    def test_empty_input_returns_empty(self) -> None:
        df = _make_seg_df(n_wells=1, cells_per_well=1).iloc[:0].copy()
        result = select_uniform_total(df, max_cells=10, seed=0)
        assert len(result) == 0
        assert list(result.columns) == list(df.columns)

    def test_deterministic(self) -> None:
        df = _make_seg_df(n_wells=2, cells_per_well=20)
        r1 = select_uniform_total(df, max_cells=10, seed=42)
        r2 = select_uniform_total(df, max_cells=10, seed=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_different_seeds_differ(self) -> None:
        df = _make_seg_df(n_wells=1, cells_per_well=50)
        r1 = select_uniform_total(df, max_cells=10, seed=1)
        r2 = select_uniform_total(df, max_cells=10, seed=2)
        assert not r1["id_local"].equals(r2["id_local"])

    def test_draws_across_wells(self) -> None:
        df = _make_seg_df(n_wells=3, cells_per_well=50)
        result = select_uniform_total(df, max_cells=30, seed=0)
        wells = set(result["fov_id"].str.split("__", expand=True)[3])
        assert len(wells) > 1  # cells drawn from multiple wells


# ── TestUniformPerCompoundSource ──────────────────────────────────────────────


def _make_multisource_seg_df(
    sources: list[str],
    inchikeys: list[str],
    cells_per_pair: int,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build seg_df + meta_df where every (inchikey × source) pair has N cells."""
    rng = np.random.default_rng(seed)
    seg_rows: list[dict] = []
    meta_rows: list[dict] = []
    for src in sources:
        for i, ik in enumerate(inchikeys):
            fov_id = f"{src}__bat__plate0__A{i:02d}__1"
            meta_rows.append({"fov_id": fov_id, "Metadata_InChIKey": ik})
            for s in range(cells_per_pair):
                seg_rows.append(
                    {
                        "fov_id": fov_id,
                        "id_local": s + 1,
                        "nuc_id_global": s + 1,
                        "cell_id_global": s + 1,
                        "cyto_cent_row": int(rng.integers(10, 90)),
                        "cyto_cent_col": int(rng.integers(10, 90)),
                        "nuc_cent_row": int(rng.integers(10, 90)),
                        "nuc_cent_col": int(rng.integers(10, 90)),
                        "nuc_area": int(rng.integers(50, 200)),
                        "cyto_area": int(rng.integers(200, 800)),
                        "nuc_cyto_ratio": float(rng.uniform(0.1, 0.5)),
                        "n_cells_in_fov": cells_per_pair,
                    }
                )
    return pd.DataFrame(seg_rows), pd.DataFrame(meta_rows)


class TestUniformPerCompoundSource:
    def test_caps_per_pair(self) -> None:
        seg_df, meta_df = _make_multisource_seg_df(
            sources=["srcA", "srcB"],
            inchikeys=["IK1", "IK2", "IK3"],
            cells_per_pair=50,
        )
        result = select_uniform_per_compound_source(
            seg_df, meta_df=meta_df, target_cells=10, seed=0
        )
        parts = result["fov_id"].str.split("__", n=1, expand=True)
        result = result.copy()
        result["_src"] = parts[0]
        ik = result["fov_id"].map(meta_df.set_index("fov_id")["Metadata_InChIKey"])
        result["_ik"] = ik
        for _, grp in result.groupby(["_ik", "_src"]):
            assert len(grp) <= 10

    def test_nan_inchikey_bypasses_cap(self) -> None:
        seg_df, meta_df = _make_multisource_seg_df(
            sources=["srcA"], inchikeys=["IK1"], cells_per_pair=50
        )
        # Add an NaN-InChIKey FOV that exceeds target_cells
        dmso_fov = "srcA__bat__plate0__B00__1"
        meta_df = pd.concat(
            [
                meta_df,
                pd.DataFrame([{"fov_id": dmso_fov, "Metadata_InChIKey": None}]),
            ],
            ignore_index=True,
        )
        dmso_rows = pd.DataFrame(
            [
                {
                    "fov_id": dmso_fov,
                    "id_local": s + 1,
                    "nuc_id_global": s + 1,
                    "cell_id_global": s + 1,
                    "cyto_cent_row": 10,
                    "cyto_cent_col": 10,
                    "nuc_cent_row": 10,
                    "nuc_cent_col": 10,
                    "nuc_area": 50,
                    "cyto_area": 200,
                    "nuc_cyto_ratio": 0.3,
                    "n_cells_in_fov": 100,
                }
                for s in range(100)
            ]
        )
        seg_df = pd.concat([seg_df, dmso_rows], ignore_index=True)
        result = select_uniform_per_compound_source(seg_df, meta_df=meta_df, target_cells=5, seed=0)
        # All 100 DMSO cells retained; treatment capped at 5
        assert (result["fov_id"] == dmso_fov).sum() == 100
        assert (result["fov_id"] != dmso_fov).sum() == 5

    def test_explicit_control_label_bypasses_cap(self) -> None:
        seg_df, meta_df = _make_multisource_seg_df(
            sources=["srcA"], inchikeys=["IK1", "EMPTY"], cells_per_pair=50
        )
        result = select_uniform_per_compound_source(
            seg_df,
            meta_df=meta_df,
            target_cells=5,
            seed=0,
            control_labels=["EMPTY"],
        )
        empty_fov = meta_df.loc[meta_df["Metadata_InChIKey"] == "EMPTY", "fov_id"].iloc[0]
        # EMPTY rows bypass cap; IK1 treatment is capped
        assert (result["fov_id"] == empty_fov).sum() == 50
        assert (result["fov_id"] != empty_fov).sum() == 5

    def test_deterministic(self) -> None:
        seg_df, meta_df = _make_multisource_seg_df(
            sources=["srcA", "srcB"], inchikeys=["IK1", "IK2"], cells_per_pair=40
        )
        r1 = select_uniform_per_compound_source(seg_df, meta_df=meta_df, target_cells=5, seed=42)
        r2 = select_uniform_per_compound_source(seg_df, meta_df=meta_df, target_cells=5, seed=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_empty_input(self) -> None:
        seg_df, meta_df = _make_multisource_seg_df(
            sources=["srcA"], inchikeys=["IK1"], cells_per_pair=1
        )
        result = select_uniform_per_compound_source(
            seg_df.iloc[:0].copy(), meta_df=meta_df, target_cells=5, seed=0
        )
        assert len(result) == 0


# ── TestSelectCells ───────────────────────────────────────────────────────────


class TestSelectCells:
    def test_adds_row_key(self, tmp_path: Path) -> None:
        df = _make_seg_df(n_wells=2, cells_per_well=5)
        fov_ids = df["fov_id"].unique().tolist()
        (tmp_path / "segmentation").mkdir()
        df.to_parquet(tmp_path / "segmentation" / "plate0.parquet", index=False)
        meta_path = _make_meta_parquet(tmp_path, fov_ids)
        selected, stats = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="all",
            cells_per_well=100,
            seed=0,
        )
        assert "row_key" in selected.columns
        assert selected["row_key"].str.contains("__").all()

    def test_parses_fov_id(self, tmp_path: Path) -> None:
        df = _make_seg_df(n_wells=1, cells_per_well=3)
        fov_ids = df["fov_id"].unique().tolist()
        (tmp_path / "segmentation").mkdir()
        df.to_parquet(tmp_path / "segmentation" / "plate0.parquet", index=False)
        meta_path = _make_meta_parquet(tmp_path, fov_ids)
        selected, _ = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="all",
            cells_per_well=100,
            seed=0,
        )
        assert "source" in selected.columns
        assert (selected["source"] == "src").all()
        assert (selected["batch"] == "bat").all()

    def test_n_cells_scaled_bounds(self, tmp_path: Path) -> None:
        df = _make_seg_df(n_wells=2, cells_per_well=5)
        fov_ids = df["fov_id"].unique().tolist()
        (tmp_path / "segmentation").mkdir()
        df.to_parquet(tmp_path / "segmentation" / "plate0.parquet", index=False)
        meta_path = _make_meta_parquet(tmp_path, fov_ids)
        selected, stats = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="all",
            cells_per_well=100,
            seed=0,
        )
        assert (selected["n_cells_scaled"] >= 0).all()
        assert (selected["n_cells_scaled"] <= 255).all()

    def test_scaling_stats_returned(self, tmp_path: Path) -> None:
        df = _make_seg_df()
        (tmp_path / "segmentation").mkdir()
        df.to_parquet(tmp_path / "segmentation" / "p.parquet", index=False)
        meta_path = _make_meta_parquet(tmp_path, df["fov_id"].unique().tolist())
        _, stats = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="all",
            cells_per_well=100,
            seed=0,
        )
        assert "n_min" in stats and "n_max" in stats
        assert stats["n_min"] <= stats["n_max"]

    def test_uniform_strategy_caps_per_well(self, tmp_path: Path) -> None:
        df = _make_seg_df(n_wells=3, cells_per_well=20)
        fov_ids = df["fov_id"].unique().tolist()
        (tmp_path / "segmentation").mkdir()
        df.to_parquet(tmp_path / "segmentation" / "p.parquet", index=False)
        meta_path = _make_meta_parquet(tmp_path, fov_ids)
        selected, _ = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="uniform_per_well",
            cells_per_well=5,
            seed=0,
        )
        parts = selected["fov_id"].str.split("__", n=4, expand=True)
        selected = selected.copy()
        selected["_pk"] = parts[0] + "__" + parts[1] + "__" + parts[2]
        selected["_well"] = parts[3]
        for _, grp in selected.groupby(["_pk", "_well"]):
            assert len(grp) <= 5

    def test_uniform_total_strategy(self, tmp_path: Path) -> None:
        df = _make_seg_df(n_wells=3, cells_per_well=20)
        fov_ids = df["fov_id"].unique().tolist()
        (tmp_path / "segmentation").mkdir()
        df.to_parquet(tmp_path / "segmentation" / "p.parquet", index=False)
        meta_path = _make_meta_parquet(tmp_path, fov_ids)
        selected, _ = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="uniform_total",
            max_cells=10,
            seed=0,
        )
        assert len(selected) == 10

    def test_uniform_total_returns_all_when_fewer_available(self, tmp_path: Path) -> None:
        df = _make_seg_df(n_wells=1, cells_per_well=5)
        fov_ids = df["fov_id"].unique().tolist()
        (tmp_path / "segmentation").mkdir()
        df.to_parquet(tmp_path / "segmentation" / "p.parquet", index=False)
        meta_path = _make_meta_parquet(tmp_path, fov_ids)
        selected, _ = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="uniform_total",
            max_cells=1000,
            seed=0,
        )
        assert len(selected) == 5

    def test_uniform_per_compound_source_strategy(self, tmp_path: Path) -> None:
        """select_cells with uniform_per_compound_source caps per (InChIKey × source)
        and keeps EMPTY-labelled controls intact via control_labels."""
        sources = ["srcA"]
        inchikeys = ["IK1", "IK2", "EMPTY"]
        cells_per_pair = 20
        rng = np.random.default_rng(99)
        seg_rows: list[dict] = []
        meta_rows: list[dict] = []
        for src in sources:
            for i, ik in enumerate(inchikeys):
                fov_id = f"{src}__bat__plate0__A{i:02d}__1"
                meta_rows.append({"id": fov_id, "Metadata_InChIKey": ik})
                for s in range(cells_per_pair):
                    seg_rows.append(
                        {
                            "fov_id": fov_id,
                            "id_local": s + 1,
                            "nuc_id_global": s + 1,
                            "cell_id_global": s + 1,
                            "cyto_cent_row": int(rng.integers(10, 90)),
                            "cyto_cent_col": int(rng.integers(10, 90)),
                            "nuc_cent_row": int(rng.integers(10, 90)),
                            "nuc_cent_col": int(rng.integers(10, 90)),
                            "nuc_area": int(rng.integers(50, 200)),
                            "cyto_area": int(rng.integers(200, 800)),
                            "nuc_cyto_ratio": float(rng.uniform(0.1, 0.5)),
                            "n_cells_in_fov": cells_per_pair,
                        }
                    )
        seg_df = pd.DataFrame(seg_rows)
        meta_df = pd.DataFrame(meta_rows)

        (tmp_path / "segmentation").mkdir()
        seg_df.to_parquet(tmp_path / "segmentation" / "plate0.parquet", index=False)
        meta_path = tmp_path / "selected_metadata.parquet"
        meta_df.to_parquet(meta_path, index=False)

        selected, stats = select_cells(
            seg_parquet_dir=tmp_path / "segmentation",
            meta_parquet=meta_path,
            strategy="uniform_per_compound_source",
            target_cells=5,
            seed=0,
            control_labels=["EMPTY"],
        )

        # IK1 and IK2 must each be capped at 5; EMPTY (control_labels) bypasses cap
        empty_fov = f"srcA__bat__plate0__A{inchikeys.index('EMPTY'):02d}__1"
        assert (selected["fov_id"] == empty_fov).sum() == cells_per_pair
        for ik in ["IK1", "IK2"]:
            fov = f"srcA__bat__plate0__A{inchikeys.index(ik):02d}__1"
            assert (selected["fov_id"] == fov).sum() <= 5

        assert "row_key" in selected.columns
        assert "n_cells_scaled" in selected.columns
        assert stats["n_min"] <= stats["n_max"]

    def test_no_parquets_raises(self, tmp_path: Path) -> None:
        (tmp_path / "segmentation").mkdir()
        meta_path = tmp_path / "meta.parquet"
        pd.DataFrame({"id": []}).to_parquet(meta_path)
        with pytest.raises(FileNotFoundError):
            select_cells(
                seg_parquet_dir=tmp_path / "segmentation",
                meta_parquet=meta_path,
                strategy="all",
                cells_per_well=10,
                seed=0,
            )
