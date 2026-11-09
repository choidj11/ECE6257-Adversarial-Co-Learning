#!/usr/bin/env python3
"""Require a completed dual-model step and finite gradient metrics in its log."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("log", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
log = re.sub(r"\x1b\[[0-9;]*m", "", args.log.read_text(errors="replace"))
gradients = {}
for role in ("a", "b2"):
    matches = re.findall(rf"actor_{role}/grad_norm:([^\s]+)", log)
    gradients[role] = float(matches[-1]) if matches else None
passed = ("Dual model collaborative training completed!" in log
          and "Asynchronously update actor..." in log
          and all(g is not None and math.isfinite(g) for g in gradients.values()))
report = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": passed,
          "log": str(args.log), "gradient_norms_console_precision": gradients,
          "nonzero_gradients": all(g is not None and g > 0 for g in gradients.values()),
          "training_steps": 1 if passed else None,
          "scope": "One-step infrastructure validation. No formal baseline scores or saved checkpoint."}
(root / "artifacts/setup/rl-smoke-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if passed else 1)
