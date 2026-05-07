"""Compute ESM2 protein-sequence embeddings for a gene symbol list.

Outputs a parquet with columns ``[gene_symbol, esm2_1280]`` at the target path.
Genes with no UniProt canonical protein sequence (lncRNAs, pseudogenes) are
excluded from the parquet and written to ``<output>.no_sequence.txt``.

Sequences are fetched live from UniProt by default.  Pass ``--fasta`` to read
from a pre-fetched FASTA file (e.g. ``data/gene_embeddings/`` area) and avoid
the ~10-minute UniProt API round-trip on reruns.

Example::

    pixi run python scripts/compute_esm2_embeddings.py \\
        --genes data/meta/all_gene_symbols.txt \\
        --fasta data/meta/rxrx1_gene_sequences.fasta \\
        --output model/gene_embeddings/esm2_1280.parquet \\
        --device cuda
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


_HF_MODEL = "facebook/esm2_t33_650M_UR50D"
_COLUMN = "esm2_1280"
_BATCH = 32


def _load_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    if not genes:
        raise ValueError(f"No gene symbols found in {path}")
    return genes


def _read_fasta(fasta_path: Path, genes: list[str]) -> dict[str, str]:
    """Parse a FASTA file and return {gene_symbol: sequence} for requested genes.

    Header format assumed: ``>sp|ACC|NAME_HUMAN ... GN=GENE_SYMBOL ...``
    Entries without a GN= tag are skipped with a warning.
    """
    import re

    gene_set = set(genes)
    seqs: dict[str, str] = {}
    current_gene: str | None = None
    current_seq: list[str] = []
    headers_without_gn: list[str] = []

    def _flush() -> None:
        if current_gene and current_gene in gene_set:
            seqs[current_gene] = "".join(current_seq)

    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_seq = []
                m = re.search(r'\bGN=(\S+)', line)
                if m:
                    current_gene = m.group(1)
                else:
                    current_gene = None
                    headers_without_gn.append(line[:80])
            else:
                current_seq.append(line)
    _flush()

    if headers_without_gn:
        warnings.warn(
            f"{len(headers_without_gn)} FASTA header(s) have no GN= tag and were skipped "
            f"(first: {headers_without_gn[0]!r}). Check FASTA source.",
            stacklevel=2,
        )

    missing = [g for g in genes if g not in seqs]
    if missing:
        warnings.warn(
            f"{len(missing)} gene(s) not found in FASTA and will be excluded: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            stacklevel=2,
        )
    return seqs


def _fetch_sequences(genes: list[str]) -> dict[str, str]:
    """Fetch canonical UniProt protein sequence for each human gene symbol."""
    try:
        from bioservices import UniProt
    except ImportError:
        raise ImportError(
            "bioservices is required for sequence fetching: pip install bioservices"
        )

    client = UniProt(verbose=False)
    seqs: dict[str, str] = {}
    missing: list[str] = []

    for gene in genes:
        try:
            result = client.search(
                f'gene_exact:{gene} AND organism_id:9606 AND reviewed:true',
                frmt="fasta",
                limit=1,
            )
            if result and isinstance(result, str) and result.startswith(">"):
                seq = "".join(result.split("\n")[1:])
                if seq:
                    seqs[gene] = seq
                    continue
        except Exception as e:
            warnings.warn(
                f"UniProt fetch failed for {gene!r}: {e!r}",
                stacklevel=2,
            )
        missing.append(gene)

    if missing:
        warnings.warn(
            f"{len(missing)} gene(s) have no UniProt canonical sequence and will be excluded: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            stacklevel=2,
        )

    return seqs


def _embed_sequences(
    gene_seqs: dict[str, str],
    device: str,
) -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL)
    model = EsmModel.from_pretrained(_HF_MODEL).to(device).eval()

    genes = list(gene_seqs.keys())
    seqs = list(gene_seqs.values())
    embeddings: dict[str, np.ndarray] = {}

    for start in range(0, len(genes), _BATCH):
        batch_genes = genes[start : start + _BATCH]
        batch_seqs = seqs[start : start + _BATCH]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)

        # mean-pool over sequence tokens (exclude padding)
        mask = inputs["attention_mask"].float().unsqueeze(-1)
        last = out.last_hidden_state  # (B, L, 1280)
        pooled = (last * mask).sum(1) / mask.sum(1)  # (B, 1280)

        for gene, emb in zip(batch_genes, pooled.cpu().float().numpy()):
            embeddings[gene] = emb

    return embeddings


def compute_esm2_embeddings(
    genes_path: Path,
    output: Path,
    device: str = "cpu",
    fasta_path: Path | None = None,
) -> Path:
    genes = _load_gene_list(genes_path)
    print(f"Processing {len(genes)} gene symbols.")

    if fasta_path is not None:
        print(f"Reading sequences from {fasta_path} (skipping UniProt fetch).")
        gene_seqs = _read_fasta(fasta_path, genes)
    else:
        gene_seqs = _fetch_sequences(genes)
    no_seq = [g for g in genes if g not in gene_seqs]
    print(f"Sequences resolved for {len(gene_seqs)}/{len(genes)} genes; {len(no_seq)} excluded.")

    if not gene_seqs:
        raise RuntimeError("No sequences fetched — all genes have missing UniProt entries.")

    embeddings = _embed_sequences(gene_seqs, device)

    rows = [{"gene_symbol": gene, _COLUMN: emb.tolist()} for gene, emb in embeddings.items()]
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {len(df)} rows → {output}")

    no_seq_path = output.with_suffix(".no_sequence.txt")
    no_seq_path.write_text("\n".join(no_seq) + ("\n" if no_seq else ""))
    if no_seq:
        print(f"  {len(no_seq)} gene(s) excluded (no UniProt sequence) → {no_seq_path.name}")

    return output


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--genes", type=Path, required=True, help="Text file with one gene symbol per line")
    p.add_argument("--output", type=Path, required=True, help="Output parquet path")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--fasta", type=Path, default=None,
        help="Pre-fetched FASTA file (skips UniProt API). Use the file produced by fetch_gene_sequences.py.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compute_esm2_embeddings(args.genes, args.output, device=args.device, fasta_path=args.fasta)


if __name__ == "__main__":
    main()
