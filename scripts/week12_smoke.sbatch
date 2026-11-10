#!/usr/bin/env bash
#SBATCH --job-name=a1-week12-smoke
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/setup/week12-smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/activate.sh rl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
export PYTHONPATH="$PROJECT_ROOT/external/Code-A1/Code-A1/code/rl${PYTHONPATH:+:$PYTHONPATH}"
# Keep Ray's Unix sockets off long Scratch paths and out of persistent storage.
export RAY_TMPDIR="/tmp/a1-ray-$USER-$SLURM_JOB_ID"
source scripts/start_sandbox.sh
trap stop_sandbox EXIT
python -u scripts/check_quota.py
python -u scripts/prepare_week12.py
module load python/3.11.9
start_sandbox
"$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/check_sandbox.py
export PREFIX="$PROJECT_ROOT/external/Code-A1/Code-A1"
"$PROJECT_ROOT/.venv-rl/bin/python" -u "$PREFIX/code/rl/main_a1.py" "$PROJECT_ROOT/artifacts/week12/smoke.yaml"
"$PROJECT_ROOT/.venv-rl/bin/python" scripts/summarize_rl_smoke.py "artifacts/setup/week12-smoke-$SLURM_JOB_ID.log"
