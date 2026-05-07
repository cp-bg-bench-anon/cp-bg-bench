#!/bin/bash
source ~/.bashrc
cd "$(dirname "$0")/.."

export HYDRA_FULL_ERROR=1
unset SLURM_CPU_BIND
pixi run python scripts/train.py --multirun launcher=cluster
