#!/usr/bin/env bash
# run_config.sh — train + embed + aggregate for one experiment config
#
# Usage (must be run from model/):
#   bash scripts/run_config.sh <CONFIG_ID> [MAX_EPOCHS]
#
#   CONFIG_ID   runs.csv experiment ID, e.g. 25_Rxrx3C_DINO_ESM2_C
#   MAX_EPOCHS  override training epochs (default: 25)
#
# Pipeline:
#   1. Train       → {PRETRAIN_DIR}/{EXPERIMENT_ID}_{timestamp}/exports/imgenc_finetuned.{pth,json,pt}
#   2. Inference   → {EMBED_DIR}/{CONFIG_ID}_{singlecell,aggregated}.h5ad
#                    (via run_inference.py which derives params from .hydra/config.yaml)
#
# After the script completes, update eval/runs.csv manually:
#   Trained:       path to last.ckpt  (printed at the end)
#   Inference+Agg: path to aggregated.h5ad  (printed at the end)

set -euo pipefail

# ── Paths (edit if cluster layout differs) ────────────────────────────────────
PATHS_ROOT="${DATA_ROOT}"
PRETRAIN_DIR="${PATHS_ROOT}/results/pretrain"
EMBED_DIR="${PATHS_ROOT}/results/embeddings"
# paths=local is the Hydra default (config/default.yaml); no override needed here.

# ── Arguments ─────────────────────────────────────────────────────────────────
CONFIG_ID="${1:?Usage: $0 <CONFIG_ID> [MAX_EPOCHS]}"
MAX_EPOCHS="${2:-25}"

# ── Parse CONFIG_ID: {NUM}_{DATASET}_{IMGENC}_{PERTENC}_{VIEW} ───────────────
IFS='_' read -r NUM DATASET IMGENC PERTENC VIEW <<< "$CONFIG_ID"

# ── Validate dataset ──────────────────────────────────────────────────────────
case "$DATASET" in
  JUMP|Rxrx1|Rxrx3C) ;;
  *)
    echo "ERROR: Unknown dataset '${DATASET}' in CONFIG_ID '${CONFIG_ID}'" >&2
    exit 1
    ;;
esac

EXPERIMENT_ID="$CONFIG_ID"

EXPERIMENT_YAML="config/experiment/${EXPERIMENT_ID}.yaml"
if [[ ! -f "$EXPERIMENT_YAML" ]]; then
  echo "ERROR: Experiment config not found: ${EXPERIMENT_YAML}" >&2
  echo "       (run from model/ directory)" >&2
  exit 1
fi

# ── Dataset path ─────────────────────────────────────────────────────────────
case "${DATASET}_${VIEW}" in
  JUMP_C)    DATA_PATH="${PATHS_ROOT}/jump_training/datasets/crops_resharded" ;;
  JUMP_CD)   DATA_PATH="${PATHS_ROOT}/jump_training/datasets/crops_density_resharded" ;;
  JUMP_S)    DATA_PATH="${PATHS_ROOT}/jump_training/datasets/seg_resharded" ;;
  JUMP_SD)   DATA_PATH="${PATHS_ROOT}/jump_training/datasets/seg_density_resharded" ;;
  Rxrx1_C)   DATA_PATH="${PATHS_ROOT}/rxrx1_training/datasets/crops_resharded" ;;
  Rxrx1_CD)  DATA_PATH="${PATHS_ROOT}/rxrx1_training/datasets/crops_density_resharded" ;;
  Rxrx1_S)   DATA_PATH="${PATHS_ROOT}/rxrx1_training/datasets/seg_resharded" ;;
  Rxrx1_SD)  DATA_PATH="${PATHS_ROOT}/rxrx1_training/datasets/seg_density_resharded" ;;
  Rxrx3C_C)  DATA_PATH="${PATHS_ROOT}/rxrx3_training/datasets/crops_resharded" ;;
  Rxrx3C_CD) DATA_PATH="${PATHS_ROOT}/rxrx3_training/datasets/crops_density_resharded" ;;
  Rxrx3C_S)  DATA_PATH="${PATHS_ROOT}/rxrx3_training/datasets/seg_resharded" ;;
  Rxrx3C_SD) DATA_PATH="${PATHS_ROOT}/rxrx3_training/datasets/seg_density_resharded" ;;
  *)
    echo "ERROR: Unknown dataset+view '${DATASET}_${VIEW}'" >&2
    exit 1
    ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────
cat <<EOF

╔══════════════════════════════════════════════════════════════════════════╗
║  run_config.sh                                                           ║
╟──────────────────────────────────────────────────────────────────────────╢
║  config_id   : ${CONFIG_ID}
║  dataset     : ${DATASET} / view=${VIEW}
║  max_epochs  : ${MAX_EPOCHS}
║  data_path   : ${DATA_PATH}
╚══════════════════════════════════════════════════════════════════════════╝

EOF

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo "=== [1/2] Training ==="
pixi run python scripts/train.py \
  "experiment=${EXPERIMENT_ID}" \
  "paths.data=${DATA_PATH}" \
  "training.lightning.trainer.max_epochs=${MAX_EPOCHS}"

# Hydra names the output dir: {PRETRAIN_DIR}/{EXPERIMENT_ID}_{YYYYMMDDHHMM}
# Pick the most recent matching directory.
RUN_DIR=$(ls -td "${PRETRAIN_DIR}/${EXPERIMENT_ID}_"???????????? 2>/dev/null | head -1 || true)
if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: No training output found under ${PRETRAIN_DIR}/${EXPERIMENT_ID}_*" >&2
  echo "       Check that training completed successfully." >&2
  exit 1
fi

LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"

# Prefer new .pth export; fall back to legacy .pt
if [[ -f "${RUN_DIR}/exports/imgenc_finetuned.pth" ]]; then
  CHECKPOINT="${RUN_DIR}/exports/imgenc_finetuned.pth"
elif [[ -f "${RUN_DIR}/exports/imgenc_finetuned.pt" ]]; then
  CHECKPOINT="${RUN_DIR}/exports/imgenc_finetuned.pt"
else
  echo "ERROR: Encoder not found at ${RUN_DIR}/exports/imgenc_finetuned.{pth,pt}" >&2
  exit 1
fi

echo "Training output : ${RUN_DIR}"
echo "Encoder         : ${CHECKPOINT}"

# ── Step 2: Inference + Aggregation ───────────────────────────────────────────
echo ""
echo "=== [2/2] Inference + Aggregation ==="
pixi run python scripts/run_inference.py \
  --run-dir    "${RUN_DIR}" \
  --output-dir "${EMBED_DIR}"

AGGREGATED_H5AD="${EMBED_DIR}/${CONFIG_ID}_aggregated.h5ad"

# ── Done ──────────────────────────────────────────────────────────────────────
cat <<EOF

╔══════════════════════════════════════════════════════════════════════════╗
║  Done — update eval/runs.csv with the following paths:                  ║
╟──────────────────────────────────────────────────────────────────────────╢
║  Trained       : ${LAST_CKPT}
║  Inference+Agg : ${AGGREGATED_H5AD}
╚══════════════════════════════════════════════════════════════════════════╝

EOF
