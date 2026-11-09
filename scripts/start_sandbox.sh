#!/usr/bin/env bash
# Source from a compute job; caller owns cleanup via stop_sandbox.
start_sandbox() {
    : "${SLURM_JOB_ID:?Sandbox execution must run in a Slurm allocation}"
    local sandbox_python="$PROJECT_ROOT/.venv-sandbox/bin/python"
    export SANDBOX_PORT="${SANDBOX_PORT:-$((18080 + SLURM_JOB_ID % 10000))}"
    export SANDBOX_FUSION_ENDPOINT="http://127.0.0.1:$SANDBOX_PORT"
    "$sandbox_python" -u "$PROJECT_ROOT/scripts/sandbox_service.py" \
        > "$PROJECT_ROOT/artifacts/setup/sandbox-service-$SLURM_JOB_ID.log" 2>&1 &
    SANDBOX_SERVICE_PID=$!
    for ((attempt=0; attempt<90; attempt++)); do
        if curl -fsS "$SANDBOX_FUSION_ENDPOINT/v1/ping" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$SANDBOX_SERVICE_PID" 2>/dev/null; then
            cat "$PROJECT_ROOT/artifacts/setup/sandbox-service-$SLURM_JOB_ID.log" >&2
            return 1
        fi
        sleep 1
    done
    printf 'Sandbox readiness timeout\n' >&2
    stop_sandbox
    return 1
}

stop_sandbox() {
    if [[ -n "${SANDBOX_SERVICE_PID:-}" ]]; then
        kill "$SANDBOX_SERVICE_PID" 2>/dev/null || true
        wait "$SANDBOX_SERVICE_PID" 2>/dev/null || true
    fi
}
