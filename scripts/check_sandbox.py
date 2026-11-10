#!/usr/bin/env python3
"""Exercise the real SandboxFusion SDK against an allocated service."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from sandbox_fusion import run_code, RunCodeRequest, RunStatus, CommandRunStatus

ROOT = Path(__file__).resolve().parents[1]
endpoint = os.environ["SANDBOX_FUSION_ENDPOINT"]
checks = {}
cases = {
    "success": ("import numpy as np; assert int(np.sum([20,22])) == 42; print('ok')", 5),
    "failure": ("assert 1 == 2", 5),
    "timeout": ("while True: pass", 1),
    "isolation": (
        "import os, socket\n"
        "assert not os.path.exists('/storage/ice1')\n"
        "assert not os.path.exists('/home/hice1')\n"
        "assert not os.environ.get('SANDBOX_FUSION_ENDPOINT')\n"
        "assert len(os.listdir('/proc/self/ns')) > 0\n"
        "s=socket.socket(); s.settimeout(0.2)\n"
        "try: s.connect(('1.1.1.1', 80))\n"
        "except OSError: print('isolated')\n"
        "else: raise AssertionError('network was reachable')", 5),
}
ult = json.loads((ROOT / "data/UnLeakedTestBench/ULT_Lite.jsonl").read_text())[0]
probe = ult["code"] + f"\n\ndef test_fut_available():\n    assert callable({ult['func_name']})\n"
cases["ult_pytest_probe"] = (
    "import pathlib, pytest\n" + f"pathlib.Path('test_probe.py').write_text({probe!r})\n" +
    "raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', 'test_probe.py']))", 10)
for name, (code, timeout) in cases.items():
    response = run_code(RunCodeRequest(code=code, language="python", run_timeout=timeout),
                        endpoint=endpoint, max_attempts=1, client_timeout=15)
    run = response.run_result
    valid = {
        "success": response.status == RunStatus.Success and run.return_code == 0 and run.stdout.strip() == "ok",
        "failure": response.status == RunStatus.Failed and run.return_code != 0 and "AssertionError" in run.stderr,
        "timeout": response.status == RunStatus.Failed and run.status == CommandRunStatus.TimeLimitExceeded,
        "isolation": response.status == RunStatus.Success and run.stdout.strip() == "isolated",
        "ult_pytest_probe": response.status == RunStatus.Success and "1 passed" in run.stdout,
    }[name]
    checks[name] = {"passed": valid, "response": response.dict()}
    print(f"{'PASS' if valid else 'FAIL'} sandbox {name}: {response.status}", flush=True)
report = {"checked_at": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint,
          "checks": checks, "passed": all(c["passed"] for c in checks.values())}
(ROOT / "artifacts/setup/sandbox-check.json").write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if report["passed"] else 1)
