#!/usr/bin/env bash
#SBATCH --job-name=a1-study-checks
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/setup/study-checks-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
module load python/3.11.9
source scripts/start_sandbox.sh
trap stop_sandbox EXIT
start_sandbox
"$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/check_feedback.py
"$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/check_study_evaluation.py
