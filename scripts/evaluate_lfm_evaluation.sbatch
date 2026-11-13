#!/usr/bin/env bash
#SBATCH --job-name=lfm3-evaluate
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/setup/lfm3-evaluate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
export WANDB_MODE=disabled
stage="${1:?small or full}"
stream_args=()
if [[ "$stage" == full ]]; then
  stream_args=(--stream-coder)
  python scripts/gate_lfm_evaluation.py --require-passed
  for role in coder tester; do
    target="$PROJECT_ROOT/artifacts/lfm3-evaluation/full/$role-results.jsonl"
    if [[ ! -f "$target" ]]; then
      cp "$PROJECT_ROOT/artifacts/lfm3-evaluation/small/$role-results.jsonl" "$target"
    fi
  done
fi
export SANDBOX_RUNTIME="$PROJECT_ROOT/.venv-rl"
export SANDBOX_MAX_TIMEOUT=120 SANDBOX_AS_GIB=16
module load python/3.11.9
source scripts/start_sandbox.sh
trap stop_sandbox EXIT
start_sandbox
"$PROJECT_ROOT/.venv-rl/bin/python" -u scripts/evaluate_study.py \
  --run-dir "$PROJECT_ROOT/artifacts/lfm3-evaluation/$stage" --label "LFM3-1.7B-$stage" "${stream_args[@]}"
if [[ "$stage" == small ]]; then
  "$PROJECT_ROOT/.venv-rl/bin/python" scripts/gate_lfm_evaluation.py
fi
