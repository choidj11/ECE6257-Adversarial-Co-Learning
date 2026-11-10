#!/usr/bin/env python3
"""Delete only the explicitly audited Triton compilation cache paths."""
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

project = Path(__file__).resolve().parents[1]
plan_path = project / "artifacts/setup/triton-cleanup-plan.json"
log_path = project / "artifacts/setup/triton-cleanup.json"
plan = json.loads(plan_path.read_text())
root = Path("/storage/ice1/5/4/tzhao348")
assert Path(plan["root"]) == root, "Unexpected audit root"


def validate(path):
    assert path.is_absolute() and root in path.parents, path
    assert path.resolve() == path and not path.is_symlink(), path
    assert not {"site-packages", "dist-packages"}.intersection(path.parts), path
    valid = (path in {root / "agentic-rl/.cache/triton", root / "triton_cache"}
             or (path.parent.name == "runs" and re.fullmatch(r"triton-(?:rl-|eval-)?\d+", path.name)))
    assert valid, f"Not an approved compilation cache path: {path}"
    if path.exists():
        assert path.is_dir() and path.stat().st_uid == os.getuid(), path


targets = [Path(item["path"]) for item in plan["targets"]]
assert len(targets) == len(set(targets)) == 33, "Unexpected target list"
for path in targets:
    validate(path)

report = {"started_at": datetime.now(timezone.utc).isoformat(), "targets": [],
          "excluded_software_metadata": plan["excluded_software_metadata"],
          "quota_before": subprocess.check_output(["lfs", "quota", "-u", str(os.getuid()), "-h", str(root)], text=True)}


def save():
    temporary = log_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(log_path)


save()
for path in targets:
    validate(path)
    print(f"Removing compilation cache: {path}", flush=True)
    row = {"path": str(path), "started_at": datetime.now(timezone.utc).isoformat()}
    report["targets"].append(row)
    save()
    if path.exists():
        shutil.rmtree(path)
        row["state"] = "deleted"
    else:
        row["state"] = "already_absent"
    row["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"Finished: {path.name}", flush=True)

assert all(not path.exists() for path in targets), "A cache directory remains"
for item in plan["excluded_software_metadata"]:
    assert (root / item["path"]).is_dir(), "Software metadata unexpectedly missing"
report["finished_at"] = datetime.now(timezone.utc).isoformat()
report["quota_after"] = subprocess.check_output(["lfs", "quota", "-u", str(os.getuid()), "-h", str(root)], text=True)
report["software_metadata_preserved"] = True
report["completed"] = True
save()
print(report["quota_after"], flush=True)
