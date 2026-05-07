"""Generate the training data-source config for CP-BG-Bench.

Selects 1,467 unique compounds (TARGET2 + RepHub-annotated at ≥5 sources)
from the JUMP-CP cpg0016 dataset, applying batch-diverse well sampling and
DMSO controls from every touched plate.

Selection logic
---------------
- Exclude source_1 and source_9 (1536-well plate format, incompatible geometry).
- Compound universe: union of TARGET2 compounds and RepHub-annotated compounds
  present in ≥5 of the retained 384-well sources.
- Per (compound × source): select up to MAX_WELLS_PER_SRC wells, maximising
  Metadata_Batch diversity (round-robin across batches).
- Per well: up to CELLS_PER_WELL cells from MAX_FOVS_PER_WELL random FOVs
  (enforced by global config, not this script).
- DMSO (JCP2022_033924): 1 well per compound plate touched, DMSO_CELLS_PER_PLATE
  cells each.
- Oversample map: oversample_factor = min(MAX_OVERSAMPLE_FACTOR,
  ceil(TARGET_CELLS_PER_CS / estimated_available)) per (compound × source),
  written to a companion parquet for use by the training framework.

Expected output
---------------
  ~1,467 compounds × ~6 sources × 1-5 wells = ~17k YAML sample entries
  + ~856 DMSO entries (one per touched plate)
  Estimated cells after pipeline (uniform_per_compound_source, target=100):
  ~778k treatment + ~43k DMSO = ~820k pre-QC → ~700k post-QC

Usage
-----
    pixi run -e default python scripts/make_jump_training_config.py \\
        --out config/jump_training.yml \\
        --oversample-out config/jump_training_oversample.parquet \\
        --seed 42 \\
        --cache-dir .metadata_cache
"""

from __future__ import annotations

import argparse
import io
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cp_bg_bench.io.jump import download_metadata_tables

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_SOURCES: frozenset[str] = frozenset({"source_1", "source_9"})
N_SOURCES_MIN = 5
MAX_WELLS_PER_SRC = 5
MAX_FOVS_PER_WELL = 9  # full-well coverage (JUMP uses 9 sites/well)
CELLS_PER_WELL = 100   # 9 FOVs × ~11 cells/FOV; used for oversample_factor estimate
TARGET_CELLS_PER_CS = 100
MAX_OVERSAMPLE_FACTOR = 5
DMSO_CELLS_PER_PLATE = 50
DMSO_JCP2022 = "JCP2022_033924"
DMSO_INCHIKEY = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"

_METADATA_URLS: dict[str, str] = {
    "plate":    "https://github.com/jump-cellpainting/datasets/raw/main/metadata/plate.csv.gz",
    "well":     "https://github.com/jump-cellpainting/datasets/raw/main/metadata/well.csv.gz",
    "compound": "https://github.com/jump-cellpainting/datasets/raw/main/metadata/compound.csv.gz",
    "orf":      "https://github.com/jump-cellpainting/datasets/raw/main/metadata/orf.csv.gz",
    "crispr":   "https://github.com/jump-cellpainting/datasets/raw/main/metadata/crispr.csv.gz",
}

_REPHUB_DRUGS_URL = (
    "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/"
    "repurposing_drugs_20200324.txt"
)
_REPHUB_SAMPLES_URL = (
    "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/"
    "repurposing_samples_20170327.txt"
)

_CHANNEL_S3_KEYS = ["s3_OrigDNA", "s3_OrigAGP", "s3_OrigER", "s3_OrigMito", "s3_OrigRNA"]

_SEGMENTATION_BLOCK = {
    "model": "cpsam",
    "channels_for_nucleus": [0],
    "channels_for_cell": [3, 0],
    "default_diameters": {"nucleus": 21, "cytosol": 53},
}


# ---------------------------------------------------------------------------
# Raw JUMP metadata loading (bypasses load_metadata to avoid annotation-join
# side-effects that silently drop COMPOUND plate wells for some sources)
# ---------------------------------------------------------------------------

def _build_jump_well_table(table_paths: dict[str, Path]) -> pd.DataFrame:
    """Build plate × well × compound table directly from raw CSV files.

    Uses explicit gzip decompression and a left-join for compound annotations
    so that wells without a compound entry (e.g. DMSO, controls) are retained
    with NaN InChIKey rather than dropped.
    """
    plate    = pd.read_csv(table_paths["plate"],    compression="gzip")
    well     = pd.read_csv(table_paths["well"],     compression="gzip")
    compound = pd.read_csv(table_paths["compound"], compression="gzip")

    merged = plate.merge(well, on=["Metadata_Source", "Metadata_Plate"], how="inner")
    merged = merged.merge(
        compound[["Metadata_JCP2022", "Metadata_InChIKey"]],
        on="Metadata_JCP2022",
        how="left",
    )
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# RepHub helpers
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str, timeout: int = 120) -> io.BytesIO:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return io.BytesIO(r.read())


def _load_rephub(cache_dir: Path) -> pd.DataFrame:
    """Download (and cache) RepHub drugs + samples; return joined InChIKey table."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    drugs_path   = cache_dir / "repurposing_drugs_20200324.txt"
    samples_path = cache_dir / "repurposing_samples_20170327.txt"

    if not drugs_path.exists():
        print("  downloading RepHub drugs …")
        drugs_path.write_bytes(_fetch_bytes(_REPHUB_DRUGS_URL).read())
    if not samples_path.exists():
        print("  downloading RepHub samples …")
        samples_path.write_bytes(_fetch_bytes(_REPHUB_SAMPLES_URL).read())

    drugs   = pd.read_csv(drugs_path,   sep="\t", comment="!")
    samples = pd.read_csv(samples_path, sep="\t", comment="!", encoding="latin-1")

    rephub = drugs.merge(
        samples[["pert_iname", "InChIKey"]].drop_duplicates(subset=["InChIKey"]),
        on="pert_iname",
        how="left",
    ).rename(columns={"InChIKey": "Metadata_InChIKey"})

    return rephub


# ---------------------------------------------------------------------------
# Compound universe selection
# ---------------------------------------------------------------------------

def _build_training_inchikeys(
    meta: pd.DataFrame,
    rephub: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """Return (annotated_5plus_iks, target2_iks) after excluding 1536-well sources."""
    # Drop 1536-well sources
    meta_384 = meta[~meta["Metadata_Source"].isin(EXCLUDED_SOURCES)].copy()

    # --- Annotated ≥5-source compounds ---
    compound_wells = meta_384[meta_384["Metadata_PlateType"] == "COMPOUND"].dropna(
        subset=["Metadata_InChIKey"]
    )
    source_cov = (
        compound_wells.groupby("Metadata_InChIKey")["Metadata_Source"]
        .nunique()
        .rename("n_sources")
        .reset_index()
    )
    enriched = source_cov.merge(rephub, on="Metadata_InChIKey", how="left")
    annotated_5plus = enriched[
        enriched["clinical_phase"].notna() & (enriched["n_sources"] >= N_SOURCES_MIN)
    ]
    ann_iks = set(annotated_5plus["Metadata_InChIKey"].dropna())

    # --- TARGET2 compounds ---
    t2 = meta_384[meta_384["Metadata_PlateType"] == "TARGET2"].dropna(
        subset=["Metadata_InChIKey"]
    )
    t2_iks = set(t2["Metadata_InChIKey"].unique())

    # DMSO appears on TARGET2 plates with an InChIKey mapping; exclude it so
    # it isn't double-counted (once as treatment, once as the DMSO control).
    ann_iks.discard(DMSO_INCHIKEY)
    t2_iks.discard(DMSO_INCHIKEY)

    overlap = ann_iks & t2_iks
    union   = ann_iks | t2_iks
    print(
        f"  Annotated ≥{N_SOURCES_MIN} src: {len(ann_iks):,}  "
        f"TARGET2: {len(t2_iks):,}  overlap: {len(overlap):,}  union: {len(union):,}"
    )
    return ann_iks, t2_iks


# ---------------------------------------------------------------------------
# Batch-diverse well selection
# ---------------------------------------------------------------------------

def _select_batch_diverse(
    wells: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pick up to n wells from `wells`, maximising Metadata_Batch diversity.

    Uses round-robin across batches so distinct batches are preferred before
    falling back to the same batch.
    """
    if len(wells) <= n:
        return wells.copy()

    batches = list(wells["Metadata_Batch"].unique())
    rng.shuffle(batches)
    chosen_indices: list[int] = []
    remaining = wells.copy()

    while len(chosen_indices) < n and len(remaining) > 0:
        made_progress = False
        for b in batches:
            if len(chosen_indices) >= n:
                break
            batch_rows = remaining[remaining["Metadata_Batch"] == b]
            if len(batch_rows) == 0:
                continue
            pick = batch_rows.sample(n=1, random_state=int(rng.integers(1 << 31)))
            chosen_indices.extend(pick.index.tolist())
            remaining = remaining.drop(pick.index)
            made_progress = True
        if not made_progress:
            break

    return wells.loc[chosen_indices]


# ---------------------------------------------------------------------------
# Main sampling
# ---------------------------------------------------------------------------

def _plates_with_dmso(meta: pd.DataFrame) -> set[tuple[str, str]]:
    """Return the set of (source, plate) tuples that contain ≥1 DMSO well.

    DMSO anchors are a hard requirement: any compound plate without a DMSO
    control is excluded from treatment selection so every touched plate
    contributes its own batch-matched negative.
    """
    meta_384 = meta[~meta["Metadata_Source"].isin(EXCLUDED_SOURCES)]
    dmso = meta_384[meta_384["Metadata_JCP2022"] == DMSO_JCP2022]
    return set(zip(dmso["Metadata_Source"], dmso["Metadata_Plate"], strict=False))


_ELIGIBLE_PLATE_TYPES: frozenset[str] = frozenset({"COMPOUND", "TARGET2"})


def _sample_treatment_wells(
    meta: pd.DataFrame,
    training_iks: set[str],
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    """Return (selected_well_rows, oversample_stats).

    selected_well_rows: list of row-dicts with Source/Plate/Well/JCP2022/InChIKey.
    oversample_stats: list of dicts for the oversample parquet.

    Treatment wells are drawn from COMPOUND plates first and fall back to
    TARGET2 plates only for compounds with no COMPOUND-plate presence (the
    ~66 TARGET2-exclusive compounds in our training universe). This preserves
    COMPOUND-plate data distribution for overlap compounds while recovering
    the TARGET2-only tail.

    Only plates that also carry DMSO wells are eligible, so every touched
    plate has its matched per-plate control — including TARGET2 plates drawn
    from in the fallback pass.
    """
    meta_384 = meta[~meta["Metadata_Source"].isin(EXCLUDED_SOURCES)]
    dmso_plates = _plates_with_dmso(meta)
    plate_key = list(zip(meta_384["Metadata_Source"], meta_384["Metadata_Plate"], strict=False))
    has_dmso_mask = pd.Series([p in dmso_plates for p in plate_key], index=meta_384.index)

    all_eligible = meta_384[
        meta_384["Metadata_PlateType"].isin(_ELIGIBLE_PLATE_TYPES)
        & meta_384["Metadata_InChIKey"].isin(training_iks)
    ]
    compound_wells = all_eligible[has_dmso_mask.loc[all_eligible.index]].copy()

    dropped = len(all_eligible) - len(compound_wells)
    if dropped:
        dropped_plates = set(
            zip(
                all_eligible.loc[~has_dmso_mask.loc[all_eligible.index], "Metadata_Source"],
                all_eligible.loc[~has_dmso_mask.loc[all_eligible.index], "Metadata_Plate"],
                strict=False,
            )
        )
        print(
            f"  Excluded {dropped:,} treatment wells on {len(dropped_plates)} "
            f"plates lacking DMSO controls"
        )

    selected_rows: list[dict] = []
    oversample_stats: list[dict] = []

    iks = sorted(training_iks)
    for ik in iks:
        ik_wells = compound_wells[compound_wells["Metadata_InChIKey"] == ik]
        for src, src_wells in ik_wells.groupby("Metadata_Source"):
            chosen = _select_batch_diverse(src_wells, MAX_WELLS_PER_SRC, rng)
            for _, row in chosen.iterrows():
                selected_rows.append(
                    {
                        "Metadata_Source":   str(row["Metadata_Source"]),
                        "Metadata_Batch":    str(row["Metadata_Batch"]),
                        "Metadata_Plate":    str(row["Metadata_Plate"]),
                        "Metadata_Well":     str(row["Metadata_Well"]),
                        "Metadata_JCP2022":  str(row["Metadata_JCP2022"])
                        if pd.notna(row.get("Metadata_JCP2022"))
                        else None,
                        "Metadata_InChIKey": ik,
                    }
                )
            # Estimated cells available: n_wells × cells_per_well
            est_available = len(chosen) * CELLS_PER_WELL
            oversample_factor = min(
                MAX_OVERSAMPLE_FACTOR,
                math.ceil(TARGET_CELLS_PER_CS / max(est_available, 1)),
            )
            oversample_stats.append(
                {
                    "Metadata_InChIKey":   ik,
                    "Metadata_Source":     str(src),
                    "n_wells_selected":    len(chosen),
                    "n_batches_covered":   chosen["Metadata_Batch"].nunique(),
                    "est_cells_available": est_available,
                    "oversample_factor":   oversample_factor,
                }
            )

    print(
        f"  Treatment wells selected: {len(selected_rows):,} "
        f"across {len(training_iks):,} compounds"
    )
    return selected_rows, oversample_stats


def _sample_dmso_wells(
    meta: pd.DataFrame,
    touched_plates: set[tuple[str, str]],
    rng: np.random.Generator,
) -> list[dict]:
    """Return one DMSO well per touched (source, plate)."""
    meta_384 = meta[~meta["Metadata_Source"].isin(EXCLUDED_SOURCES)]
    dmso_wells = meta_384[meta_384["Metadata_JCP2022"] == DMSO_JCP2022].copy()

    selected: list[dict] = []
    for src, plate in sorted(touched_plates):
        plate_dmso = dmso_wells[
            (dmso_wells["Metadata_Source"] == src)
            & (dmso_wells["Metadata_Plate"] == plate)
        ]
        if plate_dmso.empty:
            continue
        pick = plate_dmso.sample(n=1, random_state=int(rng.integers(1 << 31)))
        row = pick.iloc[0]
        selected.append(
            {
                "Metadata_Source":   str(row["Metadata_Source"]),
                "Metadata_Batch":    str(row["Metadata_Batch"]),
                "Metadata_Plate":    str(row["Metadata_Plate"]),
                "Metadata_Well":     str(row["Metadata_Well"]),
                "Metadata_JCP2022":  DMSO_JCP2022,
                "Metadata_InChIKey": None,
            }
        )

    print(
        f"  DMSO wells selected: {len(selected):,} "
        f"({len(touched_plates):,} plates touched)"
    )
    return selected


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _to_sample_entry(row: dict) -> dict:
    entry: dict = {
        "metadata_source": row["Metadata_Source"],
        "metadata_plate":  row["Metadata_Plate"],
        "metadata_well":   row["Metadata_Well"],
    }
    if row.get("Metadata_JCP2022"):
        entry["metadata_jcp2022"] = row["Metadata_JCP2022"]
    return entry


def _build_config(sample_rows: list[dict]) -> dict:
    return {
        "samples":         [_to_sample_entry(r) for r in sample_rows],
        "metadata_tables": _METADATA_URLS,
        "channel_s3_keys": _CHANNEL_S3_KEYS,
        "segmentation":    _SEGMENTATION_BLOCK,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="config/jump_training.yml",
        help="Output data-source YAML (default: config/jump_training.yml)",
    )
    parser.add_argument(
        "--oversample-out", default="config/jump_training_oversample.parquet",
        help="Companion oversample-factor parquet",
    )
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--cache-dir", default=".metadata_cache")
    args = parser.parse_args()

    cache_dir  = Path(args.cache_dir)
    out_path   = Path(args.out)
    over_path  = Path(args.oversample_out)
    rng        = np.random.default_rng(args.seed)

    # 1. Download / cache metadata
    print("Downloading/caching JUMP metadata …")
    table_paths = download_metadata_tables(_METADATA_URLS, cache_dir)
    print("Building well table from raw files …")
    meta = _build_jump_well_table(table_paths)
    print(f"  {len(meta):,} well rows, sources: {sorted(meta['Metadata_Source'].unique())}")

    print("Downloading/caching RepHub annotations …")
    rephub = _load_rephub(cache_dir)

    # 2. Build compound universe
    print("Building compound universe …")
    ann_iks, t2_iks = _build_training_inchikeys(meta, rephub)
    training_iks = ann_iks | t2_iks

    # 3. Sample treatment wells
    print("Sampling treatment wells …")
    treatment_rows, oversample_stats = _sample_treatment_wells(meta, training_iks, rng)

    # 4. DMSO pass — one well per touched plate
    touched_plates = {
        (r["Metadata_Source"], r["Metadata_Plate"]) for r in treatment_rows
    }
    print("Sampling DMSO wells …")
    dmso_rows = _sample_dmso_wells(meta, touched_plates, rng)

    all_rows = treatment_rows + dmso_rows

    # 5. Summary
    compounds_in_yaml = len({r["Metadata_JCP2022"] for r in all_rows if r.get("Metadata_JCP2022")})
    sources_in_yaml   = len({r["Metadata_Source"] for r in all_rows})
    print(
        f"\nSummary:"
        f"\n  Total sample entries : {len(all_rows):,}"
        f"\n  Unique JCP2022 IDs   : {compounds_in_yaml:,}"
        f"\n  Unique sources       : {sources_in_yaml}"
        f"\n  Est. max cells       : {len(treatment_rows) * CELLS_PER_WELL:,} treatment"
        f"  +  {len(dmso_rows) * DMSO_CELLS_PER_PLATE:,} DMSO"
    )

    # 6. Write YAML
    cfg = _build_config(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )
    print(f"Wrote {out_path}")

    # 7. Write oversample map
    over_df = pd.DataFrame(oversample_stats)
    over_path.parent.mkdir(parents=True, exist_ok=True)
    over_df.to_parquet(over_path, index=False)
    print(f"Wrote {over_path}  ({len(over_df):,} rows)")


if __name__ == "__main__":
    main()
