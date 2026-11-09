#!/usr/bin/env python3
"""Official SandboxFusion API with a Python-only, rootless Bubblewrap runner.

Run on a compute allocation, bound to loopback. Upstream source stays untouched.
Each request has private PID/network/mount namespaces and no project/home mounts.
This adapter intentionally supports code/stdin only, not file transfer or Jupyter.
"""
import asyncio
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/SandboxFusion"))

from fastapi import FastAPI
from sandbox.runners import CODE_RUNNERS
from sandbox.runners.types import CodeRunResult, CommandRunResult, CommandRunStatus
from sandbox.server.sandbox_api import sandbox_router

if not shutil.which("bwrap"):
    raise RuntimeError("Bubblewrap is required; refusing unisolated execution")

LIMIT = asyncio.Semaphore(4)
RUNTIME = Path(os.environ.get("SANDBOX_RUNTIME", str(ROOT / ".venv-sandbox"))).resolve()
MAX_TIMEOUT = float(os.environ.get("SANDBOX_MAX_TIMEOUT", "30"))
ADDRESS_SPACE = int(os.environ.get("SANDBOX_AS_GIB", "2")) * 2**30


def command_for(input_dir):
    command = [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        "--cap-drop", "ALL", "--clearenv",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/storage/pace-apps", "/storage/pace-apps",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--tmpfs", "/work",
        "--ro-bind", str(RUNTIME), "/opt/runtime",
        "--ro-bind", str(input_dir), "/input",
        "--chdir", "/work", "--setenv", "PATH", "/opt/runtime/bin:/usr/bin",
        "--setenv", "OPENBLAS_NUM_THREADS", "1",
        "--setenv", "OMP_NUM_THREADS", "1",
        "--setenv", "MPLBACKEND", "Agg",
        "--setenv", "TF_NUM_INTRAOP_THREADS", "1",
        "--setenv", "TF_NUM_INTEROP_THREADS", "1",
        "--setenv", "XDG_CACHE_HOME", "/tmp/cache",
        "--setenv", "MPLCONFIGDIR", "/tmp/matplotlib",
    ]
    for system_path in ("/etc/passwd", "/etc/group", "/etc/fonts", "/etc/ld.so.cache"):
        if Path(system_path).exists():
            command += ["--ro-bind", system_path, system_path]
    if (ROOT / "data/nltk").exists():
        command += ["--ro-bind", str(ROOT / "data/nltk"), "/opt/nltk",
                    "--setenv", "NLTK_DATA", "/opt/nltk"]
    # Module-provided Python links against libraries in PACE's software tree.
    paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
             if p.startswith("/usr/local/pace-apps/")]
    if paths:
        command += ["--setenv", "LD_LIBRARY_PATH", ":".join(paths)]
    return command + [
        "/usr/bin/prlimit", f"--as={ADDRESS_SPACE}", "--fsize=1048576", "--nofile=128",
        f"--cpu={int(MAX_TIMEOUT)+1}", "--", "/opt/runtime/bin/python", "-I", "-B", "/input/main.py",
    ]


async def run_python(args):
    if args.files or args.fetch_files:
        raise ValueError("This Python-only adapter does not support file transfer")
    if not 0 < args.run_timeout <= MAX_TIMEOUT:
        raise ValueError(f"run_timeout must be in (0, {MAX_TIMEOUT}] seconds")
    if len(args.code.encode()) > 131072 or len((args.stdin or "").encode()) > 65536:
        raise ValueError("Request exceeds code/stdin size limits")
    async with LIMIT:
        with tempfile.TemporaryDirectory(prefix="a1-code-") as directory:
            source = Path(directory) / "main.py"
            source.write_text(args.code)
            # Output files stay outside the child's mounts. RLIMIT_FSIZE caps them.
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                started = time.monotonic()
                process = await asyncio.create_subprocess_exec(
                    *command_for(directory), stdin=asyncio.subprocess.PIPE,
                    stdout=stdout, stderr=stderr, start_new_session=True,
                )
                status = CommandRunStatus.Finished
                try:
                    await asyncio.wait_for(process.communicate((args.stdin or "").encode()),
                                           timeout=args.run_timeout)
                except asyncio.TimeoutError:
                    status = CommandRunStatus.TimeLimitExceeded
                finally:
                    # Killing Bubblewrap also reaps its private PID namespace.
                    if process.returncode is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    await process.wait()
                stdout.seek(0)
                stderr.seek(0)
                result = CommandRunResult(
                    status=status, execution_time=time.monotonic() - started,
                    return_code=process.returncode,
                    stdout=stdout.read(1048576).decode(errors="replace"),
                    stderr=stderr.read(1048576).decode(errors="replace"),
                )
                return CodeRunResult(run_result=result)


# Keep the official request/response models, status handling and /run_code route.
CODE_RUNNERS.clear()
CODE_RUNNERS["python"] = run_python
app = FastAPI()


@app.middleware("http")
async def supported_routes(request, call_next):
    if request.url.path not in {"/run_code", "/v1/ping"}:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Python /run_code only"}, status_code=404)
    if int(request.headers.get("content-length", "0")) > 262144:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    return await call_next(request)


app.include_router(sandbox_router)


@app.get("/v1/ping")
async def ping():
    return {"status": "ready", "runner": "bubblewrap-python", "concurrency": 4}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("SANDBOX_PORT", "8080")),
                log_level="warning")
