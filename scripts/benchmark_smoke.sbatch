#!/usr/bin/env bash
#SBATCH --job-name=a1-bench-smoke
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/setup/benchmark-smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/activate.sh rl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
python -u scripts/check_quota.py
module load python/3.11.9
source scripts/start_sandbox.sh
trap stop_sandbox EXIT
start_sandbox
"$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/benchmark_smoke.py
