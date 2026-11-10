#!/usr/bin/env bash
#SBATCH --job-name=a1-sandbox-smoke
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/setup/sandbox-smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/activate.sh rl
python scripts/check_quota.py
module load python/3.11.9
source scripts/start_sandbox.sh
trap stop_sandbox EXIT
start_sandbox
"$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/check_sandbox.py
