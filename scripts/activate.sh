#!/usr/bin/env bash
# Usage: source scripts/activate.sh [rl|tools|lfm]
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
_project_env_kind="${1:-rl}"
case "$_project_env_kind" in
    rl|tools|lfm) ;;
    *) printf 'Usage: source scripts/activate.sh [rl|tools|lfm]\n' >&2; return 2 ;;
esac
if type module >/dev/null 2>&1; then
    if [[ "$_project_env_kind" == lfm ]]; then
        module load python/3.11.9
    else
        module load python/3.10.10
    fi
fi
if [[ ! -f "$PROJECT_ROOT/.venv-$_project_env_kind/bin/activate" ]]; then
    printf 'Missing environment: .venv-%s\n' "$_project_env_kind" >&2
    return 1
fi
source "$PROJECT_ROOT/.venv-$_project_env_kind/bin/activate"
unset _project_env_kind
