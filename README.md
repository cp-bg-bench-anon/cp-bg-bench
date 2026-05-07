# cp-bg-bench

Code accompanying the paper. Layout:

| Directory | Contents |
|-----------|----------|
| `data/`   | Snakemake pipeline that builds the four paired-view Cell Painting datasets (C / S / CD / SD) from JUMP-CP, RxRx1, and RxRx3-core. |
| `model/`  | LoRA fine-tuning code for the three image encoders (DINOv3 ViT-B/16, OpenPhenom, SubCell) under a multi-positive contrastive objective conditioned on perturbation embeddings (ECFP4 for compounds, ESM2 for genetic perturbations). |
| `evals/`  | Four evaluation protocols, each in its own subdirectory. Each contains a combined notebook that produces the manuscript figure(s) for that section, the per-dataset notebooks that produce the underlying CSVs, and the resulting CSVs / figures. |
| `evals/baselines/` | Classical `cp_measure` baseline (CPM rows in Table A2). |
| `docs/`   | Sample paired-view visualisations. |

## Quickstart

Each top-level directory has its own `pixi.toml` and is run from that directory:

```bash
cd data/   && pixi install && pixi run check        # dataset-build pipeline + tests
cd model/  && pixi install && pixi run check        # training pipeline + tests
cd evals/<step>/ && pixi install                    # then open the notebook
```

## Manuscript figure -> notebook map

| Figure | Notebook |
|--------|----------|
| Figure 2 (`fig_bg_exploitability_{pre,post}_harmony.pdf`) | `evals/03_phenotypic_activity/001_background_exploitability.ipynb` |
| Figure 3 (`fig_cpm_pred_post_harmony.pdf`)   | `evals/04_cp_feature_prediction/004_predict_cpm_features.ipynb` |
| Figure A1 (`fig_batch_integration_scatter.pdf`) | `evals/02_batch_integration/002_batch_integration.ipynb` |
| Figure A2 (`fig_cpm_pred_groups_by_encoder_post_harmony.pdf`) | `evals/04_cp_feature_prediction/004_predict_cpm_features.ipynb` |

Recall tables (Table 3, Table A6) come directly from the CSVs in
`evals/01_perturbation_recall/perturbation_recall_*.csv` and the corresponding
notebook `003_perturbation_recall.ipynb`.

## Reproducibility notes

The aggregated well-level embeddings (`evals/data/*.h5ad`, ~16 GB) and the 36
trained encoder checkpoints are available at https://huggingface.co/datasets/cp-bg-bench-anon/reproducibility. Once those are in place,
the four combined notebooks reproduce every manuscript figure end-to-end from
committed CSVs and h5ads in seconds-to-minutes.
