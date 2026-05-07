"""Extract unique CRISPR target gene symbols from the rxrx3-core dataset.

Downloads ``metadata_rxrx3_core.csv`` from the ``recursionpharma/rxrx3-core``
HuggingFace dataset (small file, ~a few MB), filters to CRISPR perturbation
wells, and writes a sorted unique gene symbol list.

Outputs
-------
  data/meta/rxrx3_core_gene_symbols.txt
    Sorted unique HGNC symbols for all CRISPR KO targets (controls excluded).

Usage
-----
  pixi run python scripts/make_rxrx3_core_gene_list.py
  pixi run python scripts/make_rxrx3_core_gene_list.py -- --cache-dir /tmp/rxrx3_cache
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

HF_REPO = "recursionpharma/rxrx3-core"
METADATA_FILENAME = "metadata_rxrx3_core.csv"
CACHE_DIR_DEFAULT = Path(".metadata_cache/rxrx3_core_metadata")
OUT_TXT = Path("data/meta/rxrx3_core_gene_symbols.txt")

_CONTROL_VALUES = {"EMPTY_control", "EMPTY", ""}


def _download_metadata(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_csv = cache_dir / METADATA_FILENAME

    if not local_csv.is_file():
        import shutil
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        print(f"Downloading {METADATA_FILENAME} from {HF_REPO} …")
        hf_path = hf_hub_download(HF_REPO, METADATA_FILENAME, repo_type="dataset", token=token)
        shutil.copy2(hf_path, local_csv)
        print(f"  cached to {local_csv}")
    else:
        print(f"  metadata cache hit: {local_csv}")

    return pd.read_csv(local_csv, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_TXT)
    args = parser.parse_args()

    meta = _download_metadata(args.cache_dir)
    print(f"  {len(meta):,} total rows, columns: {list(meta.columns)}")

    if "perturbation_type" not in meta.columns:
        raise ValueError("Expected 'perturbation_type' column in metadata")
    if "gene" not in meta.columns:
        raise ValueError("Expected 'gene' column in metadata")

    crispr = meta[meta["perturbation_type"] == "CRISPR"].copy()
    print(f"  {len(crispr):,} CRISPR rows")

    genes = sorted(
        g for g in crispr["gene"].dropna().unique()
        if str(g).strip() not in _CONTROL_VALUES
    )
    print(f"  {len(genes)} unique CRISPR KO gene symbols (controls excluded)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(genes) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
