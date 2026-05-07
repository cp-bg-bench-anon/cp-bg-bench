# Gene Embedding Coverage

Pre-computed ESM2 embeddings for the merged rxrx1 ∪ rxrx3_core gene universe
(1,814 unique HGNC symbols). Nine rxrx1 genes are excluded because they have
no canonical human protein in UniProt; those exclusions are documented here
and accepted as permanent limitations.

## Summary

| Model | Parquet | Genes covered | Genes excluded | Notes |
|-------|---------|---------------|----------------|-------|
| ESM2 | `esm2_1280.parquet` | 1,805 | 9 (all rxrx1) | Pseudogenes/lncRNAs with no UniProt canonical protein |

**All rxrx3_core CRISPR genes are fully covered.**

---

## ESM2 exclusions (`esm2_1280.no_sequence.txt`)

These 9 rxrx1 siRNA target genes have no canonical human protein in
UniProt Swiss-Prot (reviewed). They are pseudogenes, microRNA host genes,
or lncRNAs — no protein sequence exists to feed ESM2.

| Gene | Reason |
|------|--------|
| CNTNAP3P2 | Pseudogene |
| GUSBP17 | Pseudogene |
| HCG9 | Pseudogene |
| KLRA1P | Pseudogene |
| KRTAP93 | Pseudogene (keratin-associated protein) |
| MIR91HG | microRNA host gene (non-coding) |
| NA | Scraper artifact — not a valid HGNC symbol; treated as unresolved |
| PRORPPSMA6 | Pseudogene |
| UBE2E4P | Pseudogene |

**No fix is possible**: these genes have no protein. Wells targeting them
are dropped when generating ESM2 training configs via
`--embedding-parquet model/gene_embeddings/esm2_1280.parquet`.

---

## Operational impact

When generating a training config, always pass `--embedding-parquet`:

```bash
python scripts/make_rxrx1_training_config.py \
    --embedding-parquet model/gene_embeddings/esm2_1280.parquet \
    --out config/rxrx1_training.yml
```

The script prints the exact list of dropped genes and well counts.
Omitting `--embedding-parquet` produces a warning; the resulting config
will crash at training startup with a `KeyError` from `GeneLookupEncoder`
if any excluded gene appears in a training batch.
