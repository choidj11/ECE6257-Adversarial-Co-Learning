#!/usr/bin/env python3
"""Read Lustre quotas directly and enforce headroom before compute jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--min-free-gib", type=float, default=20)
parser.add_argument("--min-free-files", type=int, default=20000)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
result = subprocess.run(["lfs", "quota", "-u", str(os.getuid()), str(root)],
                        check=True, capture_output=True, text=True, timeout=30)
# A long filesystem name can occupy its own line. Join whitespace before parsing.
fields = result.stdout.split("igrace", 1)[1].split()
used_kib, soft_kib, hard_kib = [int(x.rstrip("*")) for x in fields[1:4]]
used_files, soft_files, hard_files = [int(x.rstrip("*")) for x in fields[5:8]]
limit_kib = min(x for x in (soft_kib, hard_kib) if x > 0)
limit_files = min(x for x in (soft_files, hard_files) if x > 0)
free_gib = (limit_kib - used_kib) / 2**20
free_files = limit_files - used_files
report = {"checked_at": datetime.now(timezone.utc).isoformat(), "scope": "user Scratch quota",
          "used_gib": used_kib / 2**20, "limit_gib": limit_kib / 2**20,
          "free_gib": free_gib, "used_files": used_files, "limit_files": limit_files,
          "free_files": free_files, "required_free_gib": args.min_free_gib,
          "required_free_files": args.min_free_files,
          "passed": free_gib >= args.min_free_gib and free_files >= args.min_free_files}
(root / "artifacts/setup/quota-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
