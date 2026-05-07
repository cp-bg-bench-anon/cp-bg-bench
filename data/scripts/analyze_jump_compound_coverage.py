"""Analyze source coverage and RepHub annotations for JUMP CPG0016 compounds.

Downloads:
  - JUMP plate.csv, well.csv, compound.csv metadata
  - RepHub annotations from jump-cellpainting/compound-annotator

Outputs a text report to stdout.
"""

import io
import urllib.request
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Download JUMP metadata
# ---------------------------------------------------------------------------

PLATE_URL = "https://github.com/jump-cellpainting/datasets/raw/main/metadata/plate.csv.gz"
WELL_URL = "https://github.com/jump-cellpainting/datasets/raw/main/metadata/well.csv.gz"
COMPOUND_URL = "https://github.com/jump-cellpainting/datasets/raw/main/metadata/compound.csv.gz"

# Broad Institute Drug Repurposing Hub (RepHub)
REPHUB_DRUGS_URL = "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt"
REPHUB_SAMPLES_URL = "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20170327.txt"

print("Downloading JUMP metadata tables…", flush=True)

def _fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return io.BytesIO(r.read())

plate_df = pd.read_csv(_fetch(PLATE_URL), compression="gzip")
well_df = pd.read_csv(_fetch(WELL_URL), compression="gzip")
compound_df = pd.read_csv(_fetch(COMPOUND_URL), compression="gzip")

print(f"  plate:    {len(plate_df):,} rows, types: {sorted(plate_df['Metadata_PlateType'].unique())}")
print(f"  well:     {len(well_df):,} rows")
print(f"  compound: {len(compound_df):,} rows", flush=True)

# ---------------------------------------------------------------------------
# 2. Build COMPOUND-plate well × source table
# ---------------------------------------------------------------------------

compound_plates = plate_df[plate_df["Metadata_PlateType"] == "COMPOUND"].copy()
print(f"\nCOMPOUND plates: {len(compound_plates):,}", flush=True)

# Merge well onto compound plates → one row per (source, plate, well)
well_plate = compound_plates.merge(
    well_df,
    on=["Metadata_Source", "Metadata_Plate"],
    how="inner",
)
print(f"Well rows on COMPOUND plates: {len(well_plate):,}", flush=True)

# Merge compound annotations to get InChIKey
well_plate = well_plate.merge(
    compound_df[["Metadata_JCP2022", "Metadata_InChIKey"]],
    on="Metadata_JCP2022",
    how="left",
)

# Count how many distinct sources each compound appears in
source_cov = (
    well_plate.dropna(subset=["Metadata_InChIKey"])
    .groupby("Metadata_InChIKey")["Metadata_Source"]
    .nunique()
    .rename("n_sources")
    .reset_index()
)
print(f"\nUnique compounds with InChIKey on COMPOUND plates: {len(source_cov):,}")

# ---------------------------------------------------------------------------
# 3. Download RepHub (drugs + samples for InChIKey mapping)
# ---------------------------------------------------------------------------

print("\nDownloading RepHub drug annotations…", flush=True)
rephub_df = None
try:
    drugs_df = pd.read_csv(_fetch(REPHUB_DRUGS_URL), sep="\t", comment="!")
    print(f"  drugs rows: {len(drugs_df):,}, cols: {list(drugs_df.columns)}")
    samples_df = pd.read_csv(_fetch(REPHUB_SAMPLES_URL), sep="\t", comment="!", encoding="latin-1")
    print(f"  samples rows: {len(samples_df):,}, cols: {list(samples_df.columns)}", flush=True)

    # Join on pert_iname to get InChIKey per drug
    rephub_df = drugs_df.merge(
        samples_df[["pert_iname", "InChIKey"]].drop_duplicates(subset=["InChIKey"]),
        on="pert_iname",
        how="left",
    )
    print(f"  joined rows: {len(rephub_df):,}")
    print(f"  phase values: {sorted(rephub_df['clinical_phase'].dropna().unique())}", flush=True)
except Exception as e:
    print(f"  Failed to download RepHub: {e}")

# ---------------------------------------------------------------------------
# 4. Join and analyse
# ---------------------------------------------------------------------------

if rephub_df is not None and "InChIKey" in rephub_df.columns:
    rephub_df = rephub_df.rename(columns={"InChIKey": "Metadata_InChIKey"})
    enriched = source_cov.merge(rephub_df, on="Metadata_InChIKey", how="left")

    print(f"\n=== COVERAGE ANALYSIS ===")
    print(f"Total COMPOUND-plate compounds (with InChIKey): {len(enriched):,}")

    annotated = enriched[enriched["clinical_phase"].notna()]
    print(f"Compounds with RepHub clinical_phase annotation: {len(annotated):,}")

    phase_dist = annotated["clinical_phase"].value_counts()
    print(f"\nClinical phase breakdown:")
    for phase, count in phase_dist.items():
        print(f"  {phase!r:30s}: {count:5,}")

    launched = enriched[enriched["clinical_phase"] == "Launched"]
    print(f"\n=== LAUNCHED DRUGS ===")
    print(f"Total in CPG0016 COMPOUND plates: {len(launched):,}")

    lcov = launched["n_sources"].value_counts().sort_index()
    print(f"\nSource coverage distribution for launched drugs:")
    for n, count in lcov.items():
        pct = count / len(launched) * 100
        print(f"  {n:2d} sources: {count:6,} compounds ({pct:.1f}%)")

    print(f"\nCumulative (>= threshold):")
    for threshold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        n_above = (launched["n_sources"] >= threshold).sum()
        if n_above > 0:
            print(f"  >= {threshold} sources: {n_above:5,} ({n_above/len(launched)*100:.1f}%)")

    # MoA distribution for launched drugs in >= 5 sources
    launched_5 = launched[launched["n_sources"] >= 5]
    print(f"\nLaunched drugs in >= 5 sources: {len(launched_5):,}")
    if "moa" in launched_5.columns:
        moa_counts = launched_5["moa"].dropna().str.split("|").explode().str.strip().value_counts()
        print(f"Top 20 MoAs:")
        for moa, count in moa_counts.head(20).items():
            print(f"  {moa!r:45s}: {count:4,}")

    # High-coverage baseline
    high_cov = enriched[enriched["n_sources"] >= 10]
    print(f"\n=== HIGH COVERAGE (>=10 sources, TARGET2-like) ===")
    print(f"Total: {len(high_cov):,}")
    launched_high = high_cov[high_cov["clinical_phase"] == "Launched"]
    print(f"Launched subset: {len(launched_high):,}")

else:
    print("\nSkipping RepHub join — download failed.")

print("\n=== SOURCE COVERAGE DISTRIBUTION (all compounds) ===")
dist = source_cov["n_sources"].value_counts().sort_index()
for n, count in dist.items():
    print(f"  {n:2d} sources: {count:6,} compounds")
