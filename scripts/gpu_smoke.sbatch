#!/usr/bin/env bash
# Submit from the project root: sbatch scripts/gpu_smoke.sbatch
#SBATCH --job-name=a1-gpu-smoke
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/setup/gpu-smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
bash scripts/gpu_smoke.sh
