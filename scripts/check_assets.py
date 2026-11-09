#!/usr/bin/env python3
"""Offline checks for project-local model, datasets, and caches (no test execution)."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
report = {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": [], "datasets": {}}


def check(name, valid, details):
    report["checks"].append({"name": name, "passed": bool(valid), "details": details})
    print(f"{'PASS' if valid else 'FAIL'} {name}: {details}", flush=True)


for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "UV_CACHE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
    p = Path(os.environ.get(key, "/missing")).resolve()
    check(key, ROOT in p.parents, str(p))

model = ROOT / "models/Qwen2.5-Coder-1.5B-Instruct"
for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"):
    p = model / name
    check(name, p.is_file() and ROOT in p.resolve().parents, str(p.resolve()))
weight = model / "model.safetensors"
if weight.is_file():
    expected = weight.resolve().name
    h = hashlib.sha256()
    with weight.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    check("model SHA256", len(expected) == 64 and h.hexdigest() == expected, h.hexdigest())

groups = {
    "KodCode-V1 train": ("data/KodCode-V1/data/train-*.parquet", 15, None),
    "HumanEval": ("data/HumanEval/openai_humaneval/*.parquet", 1, 164),
    "BigCodeBench v0.1.4": ("data/BigCodeBench/data/v0.1.4-*.parquet", 1, 1140),
    "Code-A1 train": ("data/Code-A1/train_data/*.parquet", 1, 9688),
    "Code-A1 validation": ("data/Code-A1/val_data/*.parquet", 1, 542),
}
for name, (pattern, expected_files, expected_rows) in groups.items():
    paths = sorted(ROOT.glob(pattern))
    rows = 0
    for p in paths:
        check(f"{name} path", ROOT in p.resolve().parents, str(p.relative_to(ROOT)))
        pf = pq.ParquetFile(p)
        rows += pf.metadata.num_rows
        # Decode a small batch as well as reading the footer/schema.
        next(pf.iter_batches(batch_size=1), None)
    valid = len(paths) == expected_files and (expected_rows is None or rows == expected_rows)
    check(name, valid, f"files={len(paths)}, rows={rows}")
    report["datasets"][name] = {"files": len(paths), "rows": rows}

for name, expected_rows in (("ULT.jsonl", 3909), ("ULT_Lite.jsonl", 200)):
    p = ROOT / "data/UnLeakedTestBench" / name
    # Upstream .jsonl files contain JSON arrays, despite their extension.
    content = json.loads(p.read_text()) if p.is_file() else []
    check(name, len(content) == expected_rows and ROOT in p.resolve().parents, f"records={len(content)}")
    report["datasets"][name] = {"rows": len(content)}

report["passed"] = all(c["passed"] for c in report["checks"])
out = ROOT / "artifacts/setup/asset-check.json"
out.write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if report["passed"] else 1)
