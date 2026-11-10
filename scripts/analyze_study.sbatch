#!/usr/bin/env bash
#SBATCH --job-name=a1-study-report
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/setup/study-report-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
python -u scripts/record_study_resources.py
python -u scripts/analyze_study.py
python -u scripts/finalize_study.py
