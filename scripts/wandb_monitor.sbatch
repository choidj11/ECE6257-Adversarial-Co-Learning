#!/usr/bin/env bash
#SBATCH --job-name=a1-wandb-monitor
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:15:00
#SBATCH --output=artifacts/setup/wandb-monitor-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
python -u scripts/wandb_monitor.py --mode "${1:-offline}" --max-hours 4
