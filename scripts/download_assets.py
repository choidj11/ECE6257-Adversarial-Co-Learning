#!/usr/bin/env python3
"""Download pinned project assets; refuse any cache outside this project."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "TMPDIR"):
    value = Path(os.environ.get(key, "/missing" )).resolve()
    if ROOT not in value.parents:
        raise SystemExit(f"Source scripts/env.sh first: {key} must be inside {ROOT}")

from huggingface_hub import snapshot_download

ASSETS = [
    ("model", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a", "models/Qwen2.5-Coder-1.5B-Instruct"),
    ("dataset", "KodCode/KodCode-V1", "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5", "data/KodCode-V1"),
    ("dataset", "openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "data/HumanEval"),
    ("dataset", "bigcode/bigcodebench", "b74c0d0bf70d2c0bc459be537895cca163007f1a", "data/BigCodeBench"),
]
manifest_path = ROOT / "artifacts/setup/assets.json"
manifest = {"updated_at": None, "assets": []}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())


def record(item):
    manifest["assets"] = [x for x in manifest["assets"] if x["path"] != item["path"]] + [item]
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)


def link_snapshot(target, destination):
    target = Path(target).resolve()
    if ROOT not in target.parents:
        raise RuntimeError(f"Asset outside project: {target}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == target:
            return
        raise RuntimeError(f"Existing link points to another version: {destination}")
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing path: {destination}")
    destination.symlink_to(os.path.relpath(target, destination.parent), target_is_directory=target.is_dir())


for kind, repo, revision, relative in ASSETS:
    print(f"Downloading {repo}@{revision} → {relative}", flush=True)
    target = snapshot_download(repo_id=repo, repo_type=kind, revision=revision,
                               cache_dir=os.environ["HF_HUB_CACHE"], token=False, max_workers=4)
    link_snapshot(target, ROOT / relative)
    files = [p for p in Path(target).rglob("*") if p.is_file()]
    record({"repo_id": repo, "repo_type": kind, "revision": revision, "path": relative,
            "cache_snapshot": str(Path(target).relative_to(ROOT)), "files": len(files),
            "bytes": sum(p.stat().st_size for p in files), "state": "downloaded"})
    print(f"Finished {repo}: {len(files)} files", flush=True)

for repo, relative, source in [
    ("Code-A1", "data/Code-A1", "external/Code-A1/Code-A1/code/rl"),
    ("UnLeakedTestBench", "data/UnLeakedTestBench", "external/UnLeakedTestBench/datasets"),
]:
    target = ROOT / source
    if repo == "Code-A1":
        for split in ("train_data", "val_data"):
            link_snapshot(target / split, ROOT / relative / split)
    else:
        link_snapshot(target, ROOT / relative)
    clone = ROOT / "external" / repo
    revision = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
    record({"repo_id": repo, "repo_type": "github", "revision": revision, "path": relative,
            "state": "downloaded", "source": source})

print(f"Download manifest: {manifest_path}", flush=True)
