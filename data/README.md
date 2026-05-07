# data

Snakemake pipeline that builds the four paired-view Cell Painting datasets
(C / S / CD / SD) from public sources for JUMP-CP, RxRx1, and RxRx3-core.

## Layout

```
data/
  pyproject.toml / pixi.toml      # env + tasks
  config/                         # per-source configs (jump.yml, rxrx1.yml, rxrx3_core.yml; smoke variants)
  src/cp_bg_bench/                # importable Python package
    calibrate / crops / datasets / download / io / segmentation / selection / transforms
  scripts/                        # one-off CLIs (training-config builders, gene resolvers, HF subset)
  snakemake/
    Snakefile
    rules/                        # one .smk per pipeline stage
    scripts/                      # one Python driver per rule
    profile/local                 # cores / retry profile
  tests/                          # pytest suite
  smoke_outputs/                  # tiny end-to-end sample of pipeline outputs
  data/meta/                      # gene-symbol whitelists, siRNA-ID lists
```

## Quickstart

```bash
cd data
pixi install
pixi run check              # ruff + pytest
pixi run smoke              # tiny single-plate JUMP run
```

## Pipeline stages

`download` → `resolve_metadata` → `segment` → `extract` → `quality_filter` →
`select` → `derive_variants` → `reshard` → `visualize`. Each stage is a
`*.smk` rule with a corresponding Python driver under `snakemake/scripts/`.
