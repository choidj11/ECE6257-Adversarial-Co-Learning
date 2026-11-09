#!/usr/bin/env bash
#SBATCH --job-name=a1-reference-tests
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/setup/study-calibrate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
export SANDBOX_RUNTIME="$PROJECT_ROOT/.venv-rl"
export SANDBOX_MAX_TIMEOUT=120 SANDBOX_AS_GIB=16
module load python/3.11.9
source scripts/start_sandbox.sh
trap stop_sandbox EXIT
start_sandbox
"$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/calibrate_study.py
