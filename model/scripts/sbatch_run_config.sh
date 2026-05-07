#!/usr/bin/env bash
# Submit run_config.sh (train + infer + aggregate) as an sbatch job.
#
# Usage:
#   bash scripts/sbatch_run_config.sh <CONFIG_ID> [MAX_EPOCHS]
#   bash scripts/sbatch_run_config.sh 25_Rxrx3C_DINO_ESM2_C
#   bash scripts/sbatch_run_config.sh 25_Rxrx3C_DINO_ESM2_C 10
#
# Submit all 36 configs:
#   for f in config/experiment/*.yaml; do
#     bash scripts/sbatch_run_config.sh "$(basename "${f%.yaml}")"
#   done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_ID="${1:?Usage: $0 <CONFIG_ID> [MAX_EPOCHS]}"
MAX_EPOCHS="${2:-25}"

# Cluster-specific values exposed as env vars; override to match your site.
#   SLURM_PARTITION=gpu SLURM_QOS=normal SLURM_GPU_CONSTRAINT=a100 \
#     bash scripts/sbatch_run_config.sh <CONFIG_ID>
sbatch \
  --job-name="run_${CONFIG_ID}" \
  --output="${MODEL_DIR}/logs/slurm_%j_${CONFIG_ID}.out" \
  --error="${MODEL_DIR}/logs/slurm_%j_${CONFIG_ID}.err" \
  --partition="${SLURM_PARTITION:-gpu}" \
  --qos="${SLURM_QOS:-normal}" \
  --ntasks=1 \
  --gpus=1 \
  --constraint="${SLURM_GPU_CONSTRAINT:-}" \
  --cpus-per-task=20 \
  --mem-per-cpu=14G \
  --time=24:00:00 \
  --wrap="source ~/.bashrc && cd ${MODEL_DIR} && bash scripts/run_config.sh ${CONFIG_ID} ${MAX_EPOCHS}"

echo "Submitted ${CONFIG_ID} (${MAX_EPOCHS} epochs)"
