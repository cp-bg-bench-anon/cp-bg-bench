"""Generate the rxrx1 training data-source config for CP-BG-Bench.

Selects HepG2 siRNA perturbations across all 11 HEPG2 experiments (both
official train + test splits), plus sub-sampled positive controls and
full per-plate EMPTY negative-control anchors.

Selection logic
---------------
- Restrict to ``cell_type == "HEPG2"``.
- Treatment pool: every well with ``well_type == "treatment"`` retained as-is.
  Every (sirna × experiment) pair has exactly 2 wells in rxrx1 so no subsample
  is needed — downstream rule E caps at ``target_cells=100`` per pair.
- Positive-control pool: ``well_type == "positive_control"`` wells subsampled
  to ``MAX_WELLS_PER_POSCTRL_PAIR`` per (sirna × experiment) — matches
  treatment density so the 30 repeated QC siRNAs don't dominate.
- Negative-control pool: all ``well_type == "negative_control"`` wells
  (sirna = ``"EMPTY"``) are kept intact as DMSO-analogue per-plate anchors.
  Rule E uses ``control_inchikeys=["EMPTY"]`` to bypass the cap for these.

Expected output
---------------
  ~1,108 treatment sirnas × 11 experiments × 2 wells  = ~24.4k treatment wells
  +   30 pos-ctrl sirnas × 11 experiments × 2 wells   = ~660 pos-ctrl wells
  +   1 EMPTY × 44 plates                            = ~44 neg-ctrl wells
  Estimated cells after pipeline (uniform_per_compound_source, target=100):
  ~1.22M treatment + ~33k pos-ctrl + ~2.6k EMPTY ≈ 1.26M pre-QC → ~1.07M post-QC

Usage
-----
    pixi run -e default python scripts/make_rxrx1_training_config.py \\
        --out config/rxrx1_training.yml \\
        --oversample-out config/rxrx1_training_oversample.parquet \\
        --seed 42 \\
        --cache-dir .metadata_cache/rxrx1_metadata
"""

from __future__ import annotations

import argparse
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_TYPE = "HEPG2"
MAX_WELLS_PER_POSCTRL_PAIR = 2
SITES_PER_WELL = 2  # rxrx1 has exactly 2 sites per well
CELLS_PER_WELL = 60  # 2 sites × ~30 cells/FOV; used for oversample estimate
TARGET_CELLS_PER_CS = 100
MAX_OVERSAMPLE_FACTOR = 5

EMPTY_SIRNA = "EMPTY"

# siRNA values with no valid gene target — GeneLookupEncoder raises KeyError on these.
# "NA" is included because the ThermoFisher scraper occasionally returns "NA" for
# discontinued products; pandas also converts it to NaN silently, so we make the
# exclusion explicit to avoid fragile reliance on that side-effect.
_NO_GENE_SIRNA_IDS = {EMPTY_SIRNA, "NEGATIVE_CONTROL", "NA"}

# Path to the siRNA→gene mapping produced by resolve_sirna_genes.py.
# When present, the script adds a `siRNA_ID` column (= gene symbol) to each
# well row so the model's perturbation encoder can consume a gene symbol
# instead of the raw ThermoFisher catalog ID.
_SIRNA_TO_GENE_CSV = Path("data/meta/rxrx1_sirna_to_gene.csv")

_METADATA_URL = "https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip"
_IMAGES_ZIP_URL = "https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip"
_METADATA_CSV_MEMBER = "rxrx1/metadata.csv"

_CHANNEL_NAMES = [
    "Hoechst",      # w1 — DNA / nucleus
    "ConA",         # w2 — ER
    "Phalloidin",   # w3 — AGP / actin
    "Syto14",       # w4 — RNA
    "MitoTracker",  # w5 — Mitochondria
    "WGA",          # w6 — AGP / plasma membrane
]

_SEGMENTATION_BLOCK = {
    "model": "cpsam",
    "channels_for_nucleus": [0],
    "channels_for_cell": [2, 0],
    "default_diameters": {"nucleus": 21, "cytosol": 53},
    "per_source_diameters": {},
}


# ---------------------------------------------------------------------------
# Metadata download
# ---------------------------------------------------------------------------

def _download_metadata(cache_dir: Path) -> pd.DataFrame:
    """Download + cache the rxrx1 metadata zip; return the parsed CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_zip = cache_dir / "rxrx1-metadata.zip"
    local_csv = cache_dir / "metadata.csv"

    if not local_csv.is_file():
        if not local_zip.is_file():
            print(f"  downloading rxrx1 metadata from {_METADATA_URL}")
            tmp = local_zip.with_suffix(".zip.tmp")
            with urllib.request.urlopen(_METADATA_URL, timeout=60) as resp:
                tmp.write_bytes(resp.read())
            tmp.rename(local_zip)
        with zipfile.ZipFile(local_zip) as zf:
            with zf.open(_METADATA_CSV_MEMBER) as src:
                local_csv.write_bytes(src.read())

    return pd.read_csv(local_csv, dtype={"plate": str, "well": str, "sirna": str})


# ---------------------------------------------------------------------------
# Pool assembly
# ---------------------------------------------------------------------------

def _unique_wells(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse site-level metadata to well-level (experiment, plate, well, sirna)."""
    cols = ["experiment", "plate", "well", "sirna", "well_type"]
    return df[cols].drop_duplicates().reset_index(drop=True)


def _subsample_posctrl(
    wells: pd.DataFrame,
    max_per_pair: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Cap positive-control wells to ``max_per_pair`` per (sirna × experiment)."""
    chosen: list[pd.DataFrame] = []
    for (_sirna, _exp), grp in wells.groupby(["sirna", "experiment"], sort=True):
        if len(grp) <= max_per_pair:
            chosen.append(grp)
        else:
            idx = sorted(rng.choice(len(grp), size=max_per_pair, replace=False).tolist())
            chosen.append(grp.iloc[idx])
    if not chosen:
        return wells.iloc[:0].copy()
    return pd.concat(chosen, ignore_index=True)


# ---------------------------------------------------------------------------
# Oversample stats
# ---------------------------------------------------------------------------

def _oversample_stats(wells: pd.DataFrame) -> list[dict]:
    """Per (sirna × experiment) oversample factor, mirroring JUMP generator output."""
    stats: list[dict] = []
    for (sirna, experiment), grp in wells.groupby(["sirna", "experiment"], sort=True):
        n_wells = len(grp)
        est_available = n_wells * CELLS_PER_WELL
        oversample_factor = min(
            MAX_OVERSAMPLE_FACTOR,
            math.ceil(TARGET_CELLS_PER_CS / max(est_available, 1)),
        )
        stats.append(
            {
                "Metadata_InChIKey": sirna,
                "Metadata_Source": experiment,
                "n_wells_selected": n_wells,
                "est_cells_available": est_available,
                "oversample_factor": oversample_factor,
            }
        )
    return stats


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _to_sample_entry(row: dict) -> dict:
    entry: dict = {
        "experiment": row["experiment"],
        "plate": str(row["plate"]),
        "well": row["well"],
    }
    if "siRNA_ID" in row:
        entry["siRNA_ID"] = row["siRNA_ID"]
    return entry


def _build_config(sample_rows: list[dict]) -> dict:
    return {
        "samples": [_to_sample_entry(r) for r in sample_rows],
        "metadata_url": _METADATA_URL,
        "images_zip_url": _IMAGES_ZIP_URL,
        "channel_names": _CHANNEL_NAMES,
        "segmentation": _SEGMENTATION_BLOCK,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="config/rxrx1_training.yml",
        help="Output data-source YAML (default: config/rxrx1_training.yml)",
    )
    parser.add_argument(
        "--oversample-out", default="config/rxrx1_training_oversample.parquet",
        help="Companion oversample-factor parquet",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=".metadata_cache/rxrx1_metadata")
    parser.add_argument(
        "--embedding-parquet", type=Path, default=None,
        dest="embedding_parquet",
        help=(
            "Optional: path to a gene embedding parquet (e.g. model/gene_embeddings/esm2_1280.parquet). "
            "When provided, wells whose gene symbol is absent from the parquet are dropped with a warning, "
            "preventing a KeyError crash at training startup."
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)
    over_path = Path(args.oversample_out)
    rng = np.random.default_rng(args.seed)

    # 1. Download / cache metadata
    print("Downloading/caching rxrx1 metadata …")
    meta = _download_metadata(cache_dir)
    print(f"  {len(meta):,} site rows, cell_types: {sorted(meta['cell_type'].unique())}")

    # 2. Filter to HEPG2
    hepg2 = meta[meta["cell_type"] == CELL_TYPE].copy()
    experiments = sorted(hepg2["experiment"].unique())
    print(
        f"  {CELL_TYPE}: {len(hepg2):,} site rows across {len(experiments)} experiments "
        f"({experiments[0]}..{experiments[-1]})"
    )

    hepg2_wells = _unique_wells(hepg2)
    print(f"  {len(hepg2_wells):,} unique wells")

    # 3. Split pools by well_type
    treatment = hepg2_wells[hepg2_wells["well_type"] == "treatment"].copy()
    posctrl = hepg2_wells[hepg2_wells["well_type"] == "positive_control"].copy()
    negctrl = hepg2_wells[hepg2_wells["well_type"] == "negative_control"].copy()
    print(
        f"  Pool sizes (wells): treatment={len(treatment):,}  "
        f"pos_ctrl={len(posctrl):,}  neg_ctrl={len(negctrl):,}"
    )

    # 4. Subsample positive controls
    posctrl_sub = _subsample_posctrl(posctrl, MAX_WELLS_PER_POSCTRL_PAIR, rng)
    print(f"  Pos-ctrl subsampled: {len(posctrl):,} → {len(posctrl_sub):,} wells")

    # 5. Negative controls retained as-is
    assert (negctrl["sirna"] == EMPTY_SIRNA).all(), (
        f"Expected all negative-control wells to have sirna={EMPTY_SIRNA!r}, "
        f"got {negctrl['sirna'].unique()}"
    )

    # 6. Assemble all rows
    all_wells = pd.concat([treatment, posctrl_sub, negctrl], ignore_index=True)

    # 6b. Join siRNA→gene mapping to add siRNA_ID (gene symbol) per well.
    #     Wells with no gene target (EMPTY, NEGATIVE_CONTROL) and wells whose siRNA
    #     could not be resolved are excluded — GeneLookupEncoder requires a real
    #     gene symbol in the embedding parquet for every entry it receives.
    if _SIRNA_TO_GENE_CSV.exists():
        # keep_default_na=False so "NA" gene symbols are read as the string "NA",
        # not converted to float NaN — we exclude them explicitly via _NO_GENE_SIRNA_IDS.
        sirna_map = pd.read_csv(
            _SIRNA_TO_GENE_CSV, dtype=str, keep_default_na=False
        )[["sirna_id", "gene_symbol"]]
        sirna_map = sirna_map.rename(columns={"sirna_id": "sirna", "gene_symbol": "siRNA_ID"})
        all_wells = all_wells.merge(sirna_map, on="sirna", how="left")
        n_before = len(all_wells)
        # Drop: unresolved (NaN), EMPTY negative controls, n337250 scrambled controls
        keep = all_wells["siRNA_ID"].notna() & ~all_wells["siRNA_ID"].isin(_NO_GENE_SIRNA_IDS)
        all_wells = all_wells[keep].reset_index(drop=True)
        n_dropped = n_before - len(all_wells)
        print(f"  Joined siRNA→gene: {len(all_wells):,} wells retained, {n_dropped} dropped (unresolved/control)")

        # Optional cross-check: drop wells whose gene has no entry in the target
        # embedding parquet. Without this, GeneLookupEncoder raises KeyError at
        # training startup for genes excluded during embedding generation (e.g.
        # pseudogenes with no UniProt sequence).
        if args.embedding_parquet is None:
            print(
                "  WARNING: --embedding-parquet not provided. The output config may contain "
                "genes that have no embedding (e.g. pseudogenes). Pass "
                "--embedding-parquet <path/to/esm2_1280.parquet> to drop those wells now "
                "and avoid a KeyError at training startup."
            )
        if args.embedding_parquet is not None:
            if not args.embedding_parquet.exists():
                raise FileNotFoundError(
                    f"--embedding-parquet not found: {args.embedding_parquet}"
                )
            emb_genes = set(
                pd.read_parquet(args.embedding_parquet, columns=["gene_symbol"])["gene_symbol"]
            )
            missing_genes = sorted(set(all_wells["siRNA_ID"].unique()) - emb_genes)
            if missing_genes:
                n_before_emb = len(all_wells)
                all_wells = all_wells[all_wells["siRNA_ID"].isin(emb_genes)].reset_index(drop=True)
                n_no_emb = n_before_emb - len(all_wells)
                print(
                    f"  WARNING: {n_no_emb} well(s) dropped — {len(missing_genes)} gene(s) absent from "
                    f"{args.embedding_parquet.name}: {missing_genes}. "
                    f"Re-run the embedding script with the updated gene list to include them."
                )
            else:
                print(f"  Embedding coverage: all {len(all_wells):,} wells have entries in {args.embedding_parquet.name}")
    else:
        print(
            f"  NOTE: {_SIRNA_TO_GENE_CSV} not found — run resolve_sirna_genes.py first "
            f"to add gene symbols. Output will omit siRNA_ID."
        )

    all_rows = all_wells.to_dict(orient="records")

    # 7. Summary
    n_sirnas = all_wells["sirna"].nunique()
    n_exps = all_wells["experiment"].nunique()
    est_cells = len(all_wells) * CELLS_PER_WELL
    print(
        f"\nSummary:"
        f"\n  Total sample entries : {len(all_rows):,}"
        f"\n  Unique sirnas        : {n_sirnas:,}"
        f"\n  Experiments          : {n_exps}"
        f"\n  Est. raw cells       : {est_cells:,}"
    )

    # 8. Write YAML
    cfg = _build_config(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )
    print(f"Wrote {out_path}")

    # 9. Write oversample map (treatment + pos-ctrl only; EMPTY bypasses cap)
    capped_wells = pd.concat([treatment, posctrl_sub], ignore_index=True)
    stats = _oversample_stats(capped_wells)
    over_df = pd.DataFrame(stats)
    over_path.parent.mkdir(parents=True, exist_ok=True)
    over_df.to_parquet(over_path, index=False)
    print(f"Wrote {over_path}  ({len(over_df):,} rows)")


if __name__ == "__main__":
    main()
