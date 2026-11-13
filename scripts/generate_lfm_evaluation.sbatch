#!/usr/bin/env bash
#SBATCH --job-name=lfm3-inference
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/setup/lfm3-generate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh lfm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
python scripts/check_quota.py
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u scripts/generate_lfm_evaluation.py "${1:?small or full}"
