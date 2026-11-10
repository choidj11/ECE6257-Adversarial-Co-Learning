#!/usr/bin/env bash
# Reinstall the pinned experiment environment without using home-directory caches.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
if type module >/dev/null 2>&1; then
    module load python/3.10.10
fi
command -v uv >/dev/null || { echo 'Missing project-local uv in .tools/bin' >&2; exit 1; }
[[ -d external/Code-A1/Code-A1/verl ]] || { echo 'Missing external/Code-A1 checkout' >&2; exit 1; }
if [[ ! -x .venv-rl/bin/python ]]; then
    uv venv --python 3.10 .venv-rl
fi
uv pip install --python .venv-rl/bin/python -r requirements/experiment.lock
uv pip install --python .venv-rl/bin/python --no-deps -e external/Code-A1/Code-A1/verl
wheel='flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
mkdir -p .cache/wheels
if [[ ! -s ".cache/wheels/$wheel" ]]; then
    curl -fL --retry 2 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$wheel" \
        -o ".cache/wheels/$wheel.part"
    mv ".cache/wheels/$wheel.part" ".cache/wheels/$wheel"
fi
uv pip install --python .venv-rl/bin/python ".cache/wheels/$wheel"
uv pip check --python .venv-rl/bin/python
echo 'Environment installed. Activate with: source scripts/activate.sh rl'
