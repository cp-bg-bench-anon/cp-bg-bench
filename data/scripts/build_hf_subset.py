"""Build a small HF-publishable subset of one JUMP view.

Selects a fixed list of (source, plate, well) triples and writes a filtered
HF Datasets directory. Run on every view with the same plate/well list to
produce row-aligned subsets across views.

Usage:
    python build_hf_subset.py \\
        --in_dataset ${DATA_ROOT} \\
        --out_dataset ${DATA_ROOT}
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

# JUMP-CP control InChIKeys.
DMSO = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
POSCONS_TOP8 = (
    "ZWVZORIKUNOTCS-UHFFFAOYSA-N", "ZYGHJZDHTFUPRJ-UHFFFAOYSA-N",
    "KXBDTLQSDKGAEB-UHFFFAOYSA-N", "PIMZUZSSNYHVCU-UHFFFAOYSA-N",
    "WGZOTBUYUFBEPZ-UHFFFAOYSA-N", "PFHDWRIVDDIFRP-UHFFFAOYSA-N",
    "PHOGQKDIVUJGMJ-UHFFFAOYSA-N", "PDMUGYOXRHVNMO-UHFFFAOYSA-N",
)

# Fixed subset spec: (source, plate). One plate per source; chosen for
# diverse plate sizes and presence of DMSO + standard poscons.
PLATES: tuple[tuple[str, str], ...] = (
    ("source_4", "BR00121430"),
    ("source_8", "A1166179"),
    ("source_13", "CP-CC9-R3-29"),
)
MAX_WELLS_PER_PLATE = 50
SEED = 42


def pick_wells(meta: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[source, plate, well, role] for the chosen subset.

    Per plate: 1 DMSO well + all poscon wells + random treatment wells, up to
    `MAX_WELLS_PER_PLATE` total. Treatment selection is seeded for
    reproducibility across views.
    """
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, str]] = []
    for src, plate in PLATES:
        sub = meta[(meta["source"] == src) & (meta["plate"] == plate)]
        wells = sub.drop_duplicates("well")[["well", "Metadata_InChIKey"]]
        dmso = sorted(wells[wells["Metadata_InChIKey"] == DMSO]["well"].tolist())
        poscon = sorted(wells[wells["Metadata_InChIKey"].isin(POSCONS_TOP8)]["well"].tolist())
        chosen: dict[str, str] = {w: "dmso" for w in dmso}
        for w in poscon:
            chosen.setdefault(w, "poscon")
        remaining = sorted(set(wells["well"]) - set(chosen))
        n_more = MAX_WELLS_PER_PLATE - len(chosen)
        if n_more > 0 and remaining:
            picks = rng.choice(remaining, size=min(n_more, len(remaining)), replace=False)
            for w in picks:
                chosen[str(w)] = "treatment"
        for w, role in chosen.items():
            rows.append({"source": src, "plate": plate, "well": w, "role": role})
        n_dmso = sum(r == "dmso" for r in chosen.values())
        n_pos = sum(r == "poscon" for r in chosen.values())
        n_trt = sum(r == "treatment" for r in chosen.values())
        print(f"  {src}/{plate}: {len(chosen)} wells (DMSO={n_dmso}, poscon={n_pos}, trt={n_trt})")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dataset", type=Path, required=True)
    ap.add_argument("--out_dataset", type=Path, required=True)
    ap.add_argument(
        "--well_list_csv",
        type=Path,
        default=None,
        help="If set, write the picked (source, plate, well, role) list here; "
             "lets other views reuse the exact same selection.",
    )
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.in_dataset}...")
    ds = load_from_disk(str(args.in_dataset))
    print(f"  {len(ds)} rows in {time.time() - t0:.1f}s")

    print("picking wells...")
    meta = ds.remove_columns([c for c in ("mask", "cell") if c in ds.column_names]).to_pandas()
    pick_df = pick_wells(meta)
    if args.well_list_csv is not None:
        args.well_list_csv.parent.mkdir(parents=True, exist_ok=True)
        pick_df.to_csv(args.well_list_csv, index=False)
        print(f"  well list -> {args.well_list_csv}")

    pick_keys = (pick_df["source"] + "|" + pick_df["plate"] + "|" + pick_df["well"]).values
    keys = (meta["source"] + "|" + meta["plate"] + "|" + meta["well"]).values
    keep_idx = np.flatnonzero(np.isin(keys, pick_keys)).tolist()
    print(f"  keep rows: {len(keep_idx):,} / {len(ds):,}")

    print(f"writing {args.out_dataset}...")
    if args.out_dataset.exists():
        shutil.rmtree(args.out_dataset)
    args.out_dataset.parent.mkdir(parents=True, exist_ok=True)
    sub = ds.select(keep_idx)
    sub.save_to_disk(str(args.out_dataset))

    size_b = sum(p.stat().st_size for p in args.out_dataset.rglob("*") if p.is_file())
    print(f"\ndone: {len(sub):,} rows  {size_b / 1e6:.1f} MB at {args.out_dataset}")


if __name__ == "__main__":
    main()
