#!/usr/bin/env bash
# sbatch scripts/train_study.sbatch E0
#SBATCH --job-name=a1-study
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/setup/study-train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/activate.sh rl
nvidia-smi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
export PYTHONPATH="$PROJECT_ROOT/scripts:$PROJECT_ROOT/external/Code-A1/Code-A1/code/rl${PYTHONPATH:+:$PYTHONPATH}"
export RAY_TMPDIR="/tmp/a1-study-$USER-$SLURM_JOB_ID"
export PREFIX="$PROJECT_ROOT/external/Code-A1/Code-A1"
variant="${1:?Specify E0, E1, E2 or E3}"
case "$variant" in E0|E1|E2|E3) ;; *) exit 2 ;; esac
python scripts/check_quota.py --min-free-gib 32
module load python/3.11.9
source scripts/start_sandbox.sh
trap stop_sandbox EXIT
start_sandbox
if [[ "$variant" == E0 ]]; then
    "$PROJECT_ROOT/.venv-rl/bin/python" -u "$PREFIX/code/rl/main_a1.py" "$PROJECT_ROOT/artifacts/study/$variant/config.yaml"
else
    "$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/train_feedback.py "$PROJECT_ROOT/artifacts/study/$variant/config.yaml"
fi
