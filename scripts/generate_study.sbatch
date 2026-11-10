#!/usr/bin/env bash
#SBATCH --job-name=a1-study-generate
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/setup/study-generate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
python scripts/check_quota.py
python -u scripts/generate_study.py "${1:?variant}" coder
python -u scripts/generate_study.py "$1" tester
