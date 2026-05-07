"""Generate a jump data-source config sampling N wells per target2 compound.

Downloads the JUMP-CP global metadata tables (cached locally), filters to
target2 plate type, samples up to ``--wells`` wells per unique JCP2022
compound, and writes a jump YAML config with one ``JumpSample`` entry per
selected well.

Usage
-----
    pixi run -e default python scripts/make_jump_target2_config.py \
        --out config/jump_target2.yml

Options
-------
    --out PATH          Output YAML path (default: config/jump_target2.yml)
    --wells N           Wells to sample per compound (default: 3)
    --seed INT          Random seed (default: 42)
    --cache-dir PATH    Where to cache downloaded CSVs (default: .metadata_cache)
    --plate-type STR    Plate type to filter on (default: target2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cp_bg_bench.io.jump import download_metadata_tables, load_metadata

# URLs copied from config/jump.yml — the canonical metadata source.
_METADATA_URLS: dict[str, str] = {
    "plate":    "https://github.com/jump-cellpainting/datasets/raw/main/metadata/plate.csv.gz",
    "well":     "https://github.com/jump-cellpainting/datasets/raw/main/metadata/well.csv.gz",
    "compound": "https://github.com/jump-cellpainting/datasets/raw/main/metadata/compound.csv.gz",
    "orf":      "https://github.com/jump-cellpainting/datasets/raw/main/metadata/orf.csv.gz",
    "crispr":   "https://github.com/jump-cellpainting/datasets/raw/main/metadata/crispr.csv.gz",
}

_CHANNEL_S3_KEYS = ["s3_OrigDNA", "s3_OrigAGP", "s3_OrigER", "s3_OrigMito", "s3_OrigRNA"]

_SEGMENTATION_BLOCK = {
    "model": "cpsam",
    "channels_for_nucleus": [0],
    "channels_for_cell": [3, 0],
    "default_diameters": {"nucleus": 21, "cytosol": 53},
}


def _sample_wells(
    meta: pd.DataFrame,
    plate_type: str,
    wells_per_compound: int,
    seed: int,
) -> pd.DataFrame:
    target = meta[meta["Metadata_PlateType"] == plate_type].copy()
    if target.empty:
        raise ValueError(
            f"No rows with Metadata_PlateType={plate_type!r}. "
            f"Available types: {sorted(meta['Metadata_PlateType'].dropna().unique())}"
        )

    rng = np.random.default_rng(seed)
    groups: list[pd.DataFrame] = []
    for jcp2022, grp in target.groupby("Metadata_JCP2022", sort=True):
        if pd.isna(jcp2022):
            continue
        n = min(wells_per_compound, len(grp))
        idx = sorted(rng.choice(len(grp), size=n, replace=False).tolist())
        groups.append(grp.iloc[idx])

    if not groups:
        raise ValueError(f"No compound wells found for plate type {plate_type!r}")

    result = pd.concat(groups, ignore_index=True)
    print(
        f"Selected {len(result)} wells across "
        f"{result['Metadata_JCP2022'].nunique()} compounds "
        f"(plate_type={plate_type!r}, wells_per_compound={wells_per_compound})"
    )
    return result


def _build_config(selected: pd.DataFrame) -> dict:
    samples = []
    for _, row in selected.iterrows():
        entry: dict = {
            "metadata_source": str(row["Metadata_Source"]),
            "metadata_plate":  str(row["Metadata_Plate"]),
            "metadata_well":   str(row["Metadata_Well"]),
        }
        # Attach compound identity so the intent is self-documenting.
        if pd.notna(row.get("Metadata_JCP2022")):
            entry["metadata_jcp2022"] = str(row["Metadata_JCP2022"])
        samples.append(entry)

    return {
        "samples": samples,
        "metadata_tables": _METADATA_URLS,
        "channel_s3_keys": _CHANNEL_S3_KEYS,
        "segmentation": _SEGMENTATION_BLOCK,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="config/jump_target2.yml")
    parser.add_argument("--wells", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=".metadata_cache")
    parser.add_argument("--plate-type", default="TARGET2")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)

    print(f"Downloading/caching metadata to {cache_dir} ...")
    table_paths = download_metadata_tables(_METADATA_URLS, cache_dir)

    print("Merging metadata tables ...")
    meta = load_metadata(table_paths)

    selected = _sample_wells(meta, args.plate_type, args.wells, args.seed)

    cfg = _build_config(selected)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
