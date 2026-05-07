"""Fetch canonical human protein sequences for rxrx1 gene symbols via UniProt.

For each gene symbol in `data/meta/rxrx1_gene_symbols.txt`, queries the
UniProt REST API for the canonical Swiss-Prot (reviewed) human protein entry
and writes its FASTA sequence.

Genes with no UniProt hit (lncRNAs, pseudogenes) are excluded from the
embedding output and written to a sidecar ``.no_sequence.txt`` file.

Outputs
-------
  data/meta/rxrx1_gene_sequences.fasta    — canonical protein FASTA
  data/meta/rxrx1_genes_no_sequence.txt   — genes with no UniProt hit

Usage
-----
  pixi run -e gene-map fetch-sequences
  pixi run -e gene-map fetch-sequences -- --genes data/meta/rxrx1_gene_symbols.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENE_SYMBOLS_TXT_DEFAULT = Path("data/meta/rxrx1_gene_symbols.txt")
OUT_FASTA = Path("data/meta/rxrx1_gene_sequences.fasta")
NO_SEQ_TXT = Path("data/meta/rxrx1_genes_no_sequence.txt")
CACHE_DIR_DEFAULT = Path(".metadata_cache/uniprot_sequence_cache")

_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"

# UniProt REST: ~3 req/s without registration, 10/s with API key
_DELAY = 0.35

_HEADERS = {
    "User-Agent": "cp-bg-bench-gene-fetch/1.0 (anon@example.invalid)",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# UniProt helpers
# ---------------------------------------------------------------------------

def _search_uniprot(symbol: str, session: requests.Session) -> str | None:
    """Return the best UniProt accession for a human gene symbol, or None."""
    params = {
        "query": f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,gene_names,sequence",
        "format": "json",
        "size": 5,
    }
    resp = session.get(_UNIPROT_SEARCH, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None
    # Prefer the entry whose primaryGene matches the query symbol exactly
    for entry in results:
        genes = entry.get("genes", [])
        for g in genes:
            primary = g.get("geneName", {}).get("value", "")
            if primary.upper() == symbol.upper():
                return entry["primaryAccession"]
    # Fall back to first hit
    return results[0]["primaryAccession"]


def _fetch_fasta(accession: str, session: requests.Session) -> str | None:
    """Return the FASTA string for a UniProt accession, or None on error."""
    url = _UNIPROT_FASTA.format(accession=accession)
    resp = session.get(url, timeout=20, headers={"Accept": "text/plain"})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text.strip()


def _lookup_one(
    symbol: str,
    cache_dir: Path,
    session: requests.Session,
) -> tuple[str | None, str | None]:
    """Return (accession, fasta_text) for a gene symbol, using disk cache.

    Returns (None, None) if no protein sequence found.
    """
    cache_acc = cache_dir / f"{symbol}.accession"
    cache_fasta = cache_dir / f"{symbol}.fasta"
    cache_miss = cache_dir / f"{symbol}.nohit"

    if cache_miss.exists():
        return None, None
    try:
        return cache_acc.read_text().strip(), cache_fasta.read_text()
    except FileNotFoundError:
        pass

    # Search for accession
    try:
        accession = _search_uniprot(symbol, session)
        time.sleep(_DELAY)
        if not accession:
            cache_miss.touch()
            return None, None

        fasta = _fetch_fasta(accession, session)
        time.sleep(_DELAY)
        if not fasta:
            cache_miss.touch()
            return None, None

        cache_acc.write_text(accession)
        cache_fasta.write_text(fasta)
        return accession, fasta

    except requests.RequestException as exc:
        # Transient error — do NOT cache as miss; will retry next run
        tqdm.write(f"  [WARN] {symbol}: network error — {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--genes",
        type=Path,
        default=GENE_SYMBOLS_TXT_DEFAULT,
        help="Gene symbols file, one per line (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR_DEFAULT,
        help="Per-gene FASTA cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--out-fasta",
        type=Path,
        default=OUT_FASTA,
        help="Output FASTA path (default: %(default)s)",
    )
    parser.add_argument(
        "--out-noseq",
        type=Path,
        default=NO_SEQ_TXT,
        help="Path for genes with no sequence (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.genes.exists():
        raise FileNotFoundError(
            f"{args.genes} not found — run resolve_sirna_genes.py first"
        )

    symbols = [s.strip() for s in args.genes.read_text().splitlines() if s.strip()]
    print(f"Processing {len(symbols)} gene symbols")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.out_noseq.parent.mkdir(parents=True, exist_ok=True)

    n_cached = sum(
        1 for s in symbols
        if (args.cache_dir / f"{s}.fasta").exists() or (args.cache_dir / f"{s}.nohit").exists()
    )
    print(f"  {n_cached} already cached, {len(symbols) - n_cached} to fetch")

    session = requests.Session()
    session.headers.update(_HEADERS)

    found: list[tuple[str, str]] = []   # (gene_symbol, fasta_text)
    missing: list[str] = []

    for symbol in tqdm(symbols, desc="Fetching UniProt sequences", unit="gene"):
        accession, fasta = _lookup_one(symbol, args.cache_dir, session)
        if fasta:
            found.append((symbol, fasta))
        else:
            missing.append(symbol)

    # Write combined FASTA
    with args.out_fasta.open("w") as fh:
        for _sym, fasta in found:
            fh.write(fasta)
            if not fasta.endswith("\n"):
                fh.write("\n")

    # Write no-sequence list
    args.out_noseq.write_text("\n".join(missing) + ("\n" if missing else ""))

    pct = 100 * len(found) / len(symbols) if symbols else 0
    print(
        f"\nDone:"
        f"\n  Sequences found  : {len(found)} ({pct:.1f}%)"
        f"\n  No sequence      : {len(missing)}"
        f"\nWrote {args.out_fasta}"
        f"\nWrote {args.out_noseq}"
    )


if __name__ == "__main__":
    main()
