#!/usr/bin/env bash
# Run inside a Slurm GPU allocation, e.g.:
# srun -p ice-gpu --gres=gpu:a100:1 -c 4 --mem=24G -t 00:08:00 bash scripts/gpu_smoke.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/activate.sh rl
python scripts/check_quota.py
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0
nvidia-smi
python -u scripts/check_environment.py --gpu
