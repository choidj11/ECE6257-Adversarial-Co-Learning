#!/usr/bin/env bash
# Install only the dependencies used by our Python-only SandboxFusion adapter.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/activate.sh tools
python scripts/check_quota.py
sandbox_revision=469de0845cb7d665bc68044d541aa3aa28c5b30a
if [[ ! -d external/SandboxFusion ]]; then
    git clone --no-checkout --filter=blob:none https://github.com/bytedance/SandboxFusion.git external/SandboxFusion
    git -C external/SandboxFusion checkout --detach "$sandbox_revision"
fi
if [[ "$(git -C external/SandboxFusion rev-parse HEAD)" != "$sandbox_revision" ]]; then
    printf 'SandboxFusion revision differs from the tested revision %s\n' "$sandbox_revision" >&2
    exit 1
fi
module load python/3.11.9
if [[ ! -d .venv-sandbox ]]; then
    uv venv --python 3.11.9 --no-python-downloads .venv-sandbox
fi
uv pip sync --python .venv-sandbox/bin/python requirements/sandbox.lock
