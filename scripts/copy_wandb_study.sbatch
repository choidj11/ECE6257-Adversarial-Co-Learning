#!/usr/bin/env bash
#SBATCH --job-name=a1-wandb-copy
#SBATCH --partition=ice-cpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/setup/wandb-copy-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh rl
target_config="${1:?Provide an authorized target config for a new project}"
status_file="${2:?Provide a separate status output path}"
verification_file="${3:?Provide a separate verification output path}"
# Backfilling is for an empty destination. Refuse to replay over existing runs.
python - "$target_config" <<'PY'
import json
import os
from pathlib import Path
import sys
from dotenv import dotenv_values
import wandb
target=json.loads(Path(sys.argv[1]).read_text())
assert target.get('authorized') is True and target.get('scope')=='metrics'
os.environ['WANDB_API_KEY']=dotenv_values('.env')['WANDB_API_KEY']
api=wandb.Api(timeout=30)
assert not list(api.runs(f"{target['entity']}/{target['project']}")), 'Destination must be empty; resume a partial copy separately.'
PY
python -u scripts/wandb_monitor.py --mode online --max-hours 0.4 \
  --target-config "$target_config" --status-file "$status_file"
python -u scripts/publish_wandb_summary.py \
  --target-config "$target_config" --verification-file "$verification_file"
