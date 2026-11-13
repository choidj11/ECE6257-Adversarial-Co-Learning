#!/usr/bin/env bash
#SBATCH --job-name=lfm3-report
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/setup/lfm3-report-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
export WANDB_MODE=disabled
python scripts/check_quota.py
python -u scripts/report_lfm_evaluation.py
