#!/bin/bash
#
# SLURM batch wrapper for `scripts/train.py`. Cluster-specific values
# (partition, QoS, GPU constraint) are exposed as environment variables —
# override before submitting to match your site.
#
#   SLURM_PARTITION=gpu SLURM_QOS=normal SLURM_GPU_CONSTRAINT=a100 \
#     sbatch scripts/train.sh

#SBATCH -e "slurm_%j.err"
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --qos=${SLURM_QOS:-long}
#SBATCH --cpus-per-task=10
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --nice=10000

source ~/.bashrc
# Run via pixi from the model/ directory
cd "$(dirname "$0")/.."

export HYDRA_FULL_ERROR=1
unset SLURM_CPU_BIND
srun pixi run python scripts/train.py
