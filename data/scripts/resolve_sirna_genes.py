"""Map rxrx1 ThermoFisher Silencer Select siRNA catalog IDs to HGNC gene symbols.

ThermoFisher Silencer Select siRNA product pages are JavaScript SPAs; a simple
HTTP request returns an empty shell.  This script uses Playwright (headless
Chromium) to render the page and extract the target gene symbol from the
rendered HTML.

Run once before the first call:
  pixi run -e gene-map install-browsers

Sources tried in order for each siRNA ID:
  1. ThermoFisher product page — rendered via Playwright headless Chromium
  2. (future) additional databases as fallback

Special cases handled automatically:
  EMPTY     → skipped (negative-control wells have no perturbation target)
  n337250   → NEGATIVE_CONTROL (scrambled non-targeting siRNA)

Outputs
-------
  data/meta/rxrx1_sirna_to_gene.csv
    columns: sirna_id, gene_symbol, ncbi_gene_id, source
    source: "thermofisher" | "hardcoded" | "UNRESOLVED"

  data/meta/rxrx1_gene_symbols.txt
    Sorted unique HGNC symbols for resolved genes.

  data/meta/rxrx1_genes_unresolved.txt
    siRNA IDs that could not be resolved (for manual curation).

Usage
-----
  pixi run -e gene-map install-browsers    # one-time setup
  pixi run -e gene-map resolve-sirna
  pixi run -e gene-map resolve-sirna -- --metadata-csv .metadata_cache/rxrx1_metadata/metadata.csv
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_CSV_DEFAULT = Path(".metadata_cache/rxrx1_metadata/metadata.csv")
OUT_CSV = Path("data/meta/rxrx1_sirna_to_gene.csv")
GENE_SYMBOLS_TXT = Path("data/meta/rxrx1_gene_symbols.txt")
UNRESOLVED_TXT = Path("data/meta/rxrx1_genes_unresolved.txt")
CACHE_DIR_DEFAULT = Path(".metadata_cache/sirna_lookup_cache")

NEGATIVE_CONTROL_ID = "n337250"
EMPTY_ID = "EMPTY"

_SOURCE_THERMOFISHER = "thermofisher"
_SOURCE_HARDCODED = "hardcoded"
_SOURCE_UNRESOLVED = "UNRESOLVED"

# Rate limit: wait this long between page loads to be polite to ThermoFisher
_TF_PAGE_DELAY = 1.5  # seconds between page loads (be polite)

# URL pattern for ThermoFisher's RNAi siRNA search tool.
# The results table renders: siRNA_ID | GENE_SYMBOL | Human | Coding | ...
_TF_RNAI_SEARCH_URL = (
    "https://www.thermofisher.com/order/rnai/siRNA/result?keyword={sirna_id}%20sirna"
)

# Gene symbols: uppercase letter + uppercase letters/digits/hyphens, 2-15 chars.
_GENE_SYMBOL_RE = re.compile(r'^[A-Z][A-Z0-9\-\.]{1,14}$')

# Words that look like gene symbols but aren't
_GENE_SYMBOL_STOPWORDS = {
    "SELECT", "VALIDATED", "PRE", "DESIGNED", "AMBION", "THERMO", "FISHER",
    "SILENCER", "SIRNA", "RNA", "CELL", "FOR", "THE", "AND", "OR", "NOT",
    "IN", "AT", "BY", "US", "ADD", "ORDER", "PRODUCT", "GENE", "HUMAN",
    "CODING", "STEALTH", "SEARCH", "COMPARE", "EMAIL", "EXPORT", "PRICE",
    "EACH", "VIEW", "SAVE", "CART", "RESULTS", "SPECIES", "ASSAY",
    "INVITROGEN", "APPLIED", "BIOSYSTEMS", "GIBCO", "SCIENTIFIC",
    "SWITCH", "MIRNA", "VIVO",
}


# ---------------------------------------------------------------------------
# ThermoFisher RNAi search tool scraping via Playwright
# ---------------------------------------------------------------------------

def _parse_gene_from_search_results(sirna_id: str, text: str) -> str | None:
    """Extract gene symbol from ThermoFisher RNAi search results text.

    The rendered results table has rows like:
        <sirna_id>   <GENE_SYMBOL>   Human   Coding   Silencer Select   ...

    We find the siRNA ID in the text and take the next token that looks like
    a gene symbol (all-caps, 2-15 chars, not a stopword).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sid_upper = sirna_id.upper()
    for i, line in enumerate(lines):
        if line.upper() == sid_upper:
            for j in range(i + 1, min(i + 8, len(lines))):
                candidate = lines[j].strip()
                if (
                    _GENE_SYMBOL_RE.match(candidate)
                    and candidate not in _GENE_SYMBOL_STOPWORDS
                ):
                    return candidate
    return None


def _lookup_thermofisher_playwright(
    sirna_id: str,
    page,  # playwright Page object
) -> dict | None:
    """Search ThermoFisher's RNAi tool and parse the gene symbol from results."""
    url = _TF_RNAI_SEARCH_URL.format(sirna_id=sirna_id)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        time.sleep(_TF_PAGE_DELAY)
        # Wait for results table to appear
        try:
            page.wait_for_selector("text=Your search found", timeout=10_000)
        except Exception:
            pass
        text = page.inner_text("body")
        gene = _parse_gene_from_search_results(sirna_id, text)
        if gene:
            return {"gene_symbol": gene, "ncbi_gene_id": None}
    except Exception as exc:
        tqdm.write(f"  [WARN] {sirna_id}: page error — {exc}")
    return None


# ---------------------------------------------------------------------------
# Per-ID resolution (with disk cache)
# ---------------------------------------------------------------------------

def _resolve_one_cached(
    sirna_id: str,
    cache_dir: Path,
) -> dict | None:
    """Return cached result for sirna_id, or None if not cached."""
    cache_file = cache_dir / f"{sirna_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def _cache_result(sirna_id: str, result: dict, cache_dir: Path) -> None:
    (cache_dir / f"{sirna_id}.json").write_text(json.dumps(result))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=METADATA_CSV_DEFAULT,
        help="rxrx1 metadata.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR_DEFAULT,
        help="Per-ID JSON cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_CSV,
        help="Output CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: %(default)s)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run browser with visible window (for debugging)",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading siRNA IDs from {args.metadata_csv}")
    df_meta = pd.read_csv(args.metadata_csv, usecols=["sirna"], dtype={"sirna": str})
    sirna_ids = sorted(x for x in df_meta["sirna"].unique() if x != EMPTY_ID)
    print(f"  {len(sirna_ids)} unique non-EMPTY siRNA IDs")

    # n337250 is always hard-coded; write cache entry before counting
    for sid in sirna_ids:
        if sid == NEGATIVE_CONTROL_ID and not (args.cache_dir / f"{sid}.json").exists():
            _cache_result(sid, {
                "sirna_id": sid, "gene_symbol": "NEGATIVE_CONTROL",
                "ncbi_gene_id": None, "source": _SOURCE_HARDCODED,
            }, args.cache_dir)

    to_fetch = [
        sid for sid in sirna_ids
        if sid != NEGATIVE_CONTROL_ID and not (args.cache_dir / f"{sid}.json").exists()
    ]
    n_cached = len(sirna_ids) - len(to_fetch)
    print(f"  {n_cached} already cached, {len(to_fetch)} to fetch via Playwright")

    # Import playwright lazily (only needed when there's work to do)
    if to_fetch:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise SystemExit(
                "playwright not installed. Run: pixi run -e gene-map install-browsers"
            )

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=args.headless)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()

            for sid in tqdm(to_fetch, desc="ThermoFisher lookup", unit="siRNA"):
                hit = _lookup_thermofisher_playwright(sid, page)
                if hit:
                    result = {"sirna_id": sid, "source": _SOURCE_THERMOFISHER, **hit}
                else:
                    result = {
                        "sirna_id": sid,
                        "gene_symbol": None,
                        "ncbi_gene_id": None,
                        "source": _SOURCE_UNRESOLVED,
                    }
                _cache_result(sid, result, args.cache_dir)

            browser.close()

    # Collect all results from cache
    results = []
    for sid in sirna_ids:
        cached = _resolve_one_cached(sid, args.cache_dir)
        if cached:
            results.append(cached)
        else:
            results.append({
                "sirna_id": sid, "gene_symbol": None,
                "ncbi_gene_id": None, "source": _SOURCE_UNRESOLVED,
            })

    df = pd.DataFrame(results, columns=["sirna_id", "gene_symbol", "ncbi_gene_id", "source"])

    is_gene = df["gene_symbol"].notna() & ~df["gene_symbol"].isin(["NEGATIVE_CONTROL"])
    n_resolved = int(is_gene.sum())
    n_ctrl = int((df["gene_symbol"] == "NEGATIVE_CONTROL").sum())
    n_unresolved = int((df["source"] == _SOURCE_UNRESOLVED).sum())
    print(
        f"\nResolution summary:"
        f"\n  Resolved       : {n_resolved}"
        f"\n  Neg control    : {n_ctrl}"
        f"\n  UNRESOLVED     : {n_unresolved}"
        f"\n  By source      : {df['source'].value_counts().to_dict()}"
    )

    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(df)} rows)")

    gene_symbols = sorted(df.loc[is_gene, "gene_symbol"].unique())
    GENE_SYMBOLS_TXT.parent.mkdir(parents=True, exist_ok=True)
    GENE_SYMBOLS_TXT.write_text("\n".join(gene_symbols) + "\n")
    print(f"Wrote {GENE_SYMBOLS_TXT}  ({len(gene_symbols)} unique symbols)")

    unresolved_ids = df.loc[df["source"] == _SOURCE_UNRESOLVED, "sirna_id"].tolist()
    UNRESOLVED_TXT.parent.mkdir(parents=True, exist_ok=True)
    UNRESOLVED_TXT.write_text("\n".join(unresolved_ids) + ("\n" if unresolved_ids else ""))
    print(f"Wrote {UNRESOLVED_TXT}  ({len(unresolved_ids)} unresolved)")


if __name__ == "__main__":
    main()
