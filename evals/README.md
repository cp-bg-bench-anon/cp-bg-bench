# evals

Four evaluation protocols on the cp-bg-bench corpus, one per subdirectory.

## Layout

```
evals/
  01_perturbation_recall/        # cross-batch R@k retrieval (Tables 3, A6)
  02_batch_integration/          # scIB pre/post Harmony (Table 2, Figure A1)
  03_phenotypic_activity/        # cross-plate replicate mAP (Table A2, Figure 2)
  04_cp_feature_prediction/      # ridge probe to cp_measure features (Table 4, Figures 3 + A2)
  baselines/                     # cp_measure CPM baseline (Table A2 CPM rows)
```

Each subdirectory has:

- a `0NX_*.ipynb` combined notebook that produces the manuscript figure(s) for that protocol
- per-dataset notebooks (`{JUMP,Rxrx1,Rxrx3C}_*.ipynb`) that produce the underlying CSVs
- the resulting `*.csv` driver files, and the `fig_*.{pdf,png}` manuscript figures

## Manuscript figure → notebook

| Figure | Notebook |
|---|---|
| Figure 2 | `03_phenotypic_activity/001_background_exploitability.ipynb` |
| Figure 3 / A2 | `04_cp_feature_prediction/004_predict_cpm_features.ipynb` |
| Figure A1 | `02_batch_integration/002_batch_integration.ipynb` |

Tables 3, A6 (cross-batch recall) live in `01_perturbation_recall/perturbation_recall_*.csv` directly;
the combined notebook `003_perturbation_recall.ipynb` regenerates them.

## Quickstart

```bash
cd evals/03_phenotypic_activity
pixi install
pixi run jupyter execute 001_background_exploitability.ipynb
```

The notebooks read committed CSVs and aggregated h5ads
(`evals/data/*.h5ad`, gitignored — see top-level README for download
instructions).
