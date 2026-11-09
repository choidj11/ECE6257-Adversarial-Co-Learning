#!/usr/bin/env python3
"""Read-only Scratch inode audit; do not follow links or double-count hardlinks."""
import argparse
import json
import os
import stat
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--lustre", action="store_true", help="Use Lustre MDT lazy metadata and FIDs")
parser.add_argument("--dirents", action="store_true", help="Fast inode counts from directory entries; no size/owner stats")
args = parser.parse_args()
root = args.root.resolve()
uid = os.getuid()
device = root.stat().st_dev
rows = {}
hardlinks = {}
errors = []
started = time.monotonic()
last_update = started
visited = 0
fields = ("entries", "files", "small_files", "directories", "symlinks", "own_inode_entries", "allocated_bytes")


seen_file_inodes = {}
suffixes = Counter()


def walk_dirents(path, relative):
    global visited, last_update
    totals = dict.fromkeys(fields, 0)
    rows[relative] = totals
    totals["directories"] = totals["own_inode_entries"] = 1
    try:
        with os.scandir(path) as it:
            for entry in it:
                visited += 1
                child = entry.name if relative == "." else relative + "/" + entry.name
                if time.monotonic() - last_update > 15:
                    print(f"Scanned {visited:,} entries; current: {relative}", flush=True)
                    last_update = time.monotonic()
                try:
                    if entry.is_dir(follow_symlinks=False):
                        sub = walk_dirents(Path(entry.path), child)
                        for key in fields:
                            totals[key] += sub[key]
                        continue
                    totals["entries"] += 1
                    totals["own_inode_entries"] += 1
                    if entry.is_file(follow_symlinks=False):
                        totals["files"] += 1
                        suffixes[os.path.splitext(entry.name)[1].lower() or "[no extension]"] += 1
                        inode = entry.inode()
                        if inode in seen_file_inodes:
                            group = hardlinks.setdefault(inode, {"nlink": 1, "blocks": 0,
                                                                  "parents": [seen_file_inodes[inode]]})
                            group["parents"].append(relative)
                            group["nlink"] += 1
                        else:
                            seen_file_inodes[inode] = relative
                    elif entry.is_symlink():
                        totals["symlinks"] += 1
                except OSError as exc:
                    errors.append({"path": child, "error": str(exc)})
    except OSError as exc:
        errors.append({"path": str(path), "error": str(exc)})
    return totals


def walk_lustre():
    global visited, last_update
    error_path = args.output.with_suffix(".errors.log")
    with error_path.open("w") as error_file:
        proc = subprocess.Popen(["lfs", "find", str(root), "--lazy", "--printf",
                                 "%y\\t%LF\\t%n\\t%s\\t%b\\t%U\\t%p\\n"],
                                stdout=subprocess.PIPE, stderr=error_file, text=True)
        for line in proc.stdout:
            parts = line.rstrip("\n").split("\t", 6)
            if len(parts) != 7:
                errors.append({"error": "Unparseable Lustre record", "line": line[:200]})
                continue
            kind, fid, links, size, blocks, owner, absolute = parts
            relative = absolute[len(str(root)) + 1:] if absolute != str(root) else "."
            parent = relative.rsplit("/", 1)[0] if "/" in relative else "."
            directory = relative if kind == "d" else parent
            row = rows.setdefault(directory, dict.fromkeys(fields, 0))
            owned = int(owner) == uid
            if owned:
                row["own_inode_entries"] += 1
                row["allocated_bytes"] += int(blocks) * 512
            if kind == "d":
                row["directories"] += 1
            else:
                row["entries"] += 1
            if kind == "f":
                row["files"] += 1
                row["small_files"] += int(int(size) < 65536)
                if owned and int(links) > 1:
                    group = hardlinks.setdefault(fid, {"nlink": int(links), "blocks": int(blocks) * 512, "parents": []})
                    group["parents"].append(parent)
            elif kind == "l":
                row["symlinks"] += 1
            visited += 1
            if time.monotonic() - last_update > 15:
                print(f"Scanned {visited:,} entries; current: {directory}", flush=True)
                last_update = time.monotonic()
        if proc.wait():
            errors.append({"error": "lfs find returned errors", "log": str(error_path)})
    for directory in sorted(rows, key=lambda p: p.count("/") + (p != "."), reverse=True):
        if directory == ".":
            continue
        parent = directory.rsplit("/", 1)[0] if "/" in directory else "."
        target = rows.setdefault(parent, dict.fromkeys(fields, 0))
        for key in fields:
            target[key] += rows[directory][key]


def walk(path, relative):
    global visited, last_update
    totals = dict.fromkeys(fields, 0)
    rows[relative] = totals
    try:
        own = path.lstat()
        totals["directories"] = 1
        if own.st_uid == uid:
            totals["own_inode_entries"] = 1
            totals["allocated_bytes"] = own.st_blocks * 512
        with os.scandir(path) as it:
            for entry in it:
                visited += 1
                if time.monotonic() - last_update > 15:
                    print(f"Scanned {visited:,} entries; current: {relative}", flush=True)
                    last_update = time.monotonic()
                try:
                    st = entry.stat(follow_symlinks=False)
                    child = entry.name if relative == "." else relative + "/" + entry.name
                    if stat.S_ISDIR(st.st_mode):
                        if st.st_dev != device:
                            errors.append({"path": child, "error": "Skipped another filesystem"})
                            continue
                        sub = walk(Path(entry.path), child)
                        for key in fields:
                            totals[key] += sub[key]
                        continue
                    totals["entries"] += 1
                    owned = st.st_uid == uid
                    if owned:
                        totals["own_inode_entries"] += 1
                        totals["allocated_bytes"] += st.st_blocks * 512
                    if stat.S_ISREG(st.st_mode):
                        totals["files"] += 1
                        if st.st_size < 65536:
                            totals["small_files"] += 1
                        if owned and st.st_nlink > 1:
                            key = (st.st_dev, st.st_ino)
                            group = hardlinks.setdefault(key, {"nlink": st.st_nlink, "blocks": st.st_blocks * 512, "parents": []})
                            group["parents"].append(relative)
                    elif stat.S_ISLNK(st.st_mode):
                        totals["symlinks"] += 1
                except OSError as exc:
                    errors.append({"path": str(entry.path), "error": str(exc)})
    except OSError as exc:
        errors.append({"path": str(path), "error": str(exc)})
    return totals


print(f"Auditing {root}; uid={uid}; symlinks will not be followed", flush=True)
if args.dirents:
    walk_dirents(root, ".")
elif args.lustre:
    walk_lustre()
else:
    walk(root, ".")
print(f"Traversal complete: {visited:,} entries, {len(rows):,} directories. Accounting for hardlinks.", flush=True)
for row in rows.values():
    row["unique_owned_inodes"] = row["own_inode_entries"]
    row["potentially_freed_inodes"] = row["own_inode_entries"]
    row["unique_allocated_bytes"] = row["allocated_bytes"]

for group in hardlinks.values():
    counts = Counter()
    for parent in group["parents"]:
        while True:
            counts[parent] += 1
            if parent == ".":
                break
            parent = parent.rsplit("/", 1)[0] if "/" in parent else "."
    for parent, count in counts.items():
        row = rows[parent]
        row["unique_owned_inodes"] -= count - 1
        # A file's inode survives if another hardlink exists outside this subtree.
        row["potentially_freed_inodes"] -= count - int(count == group["nlink"])
        row["unique_allocated_bytes"] -= (count - 1) * group["blocks"]

result = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "root": str(root), "uid": uid,
    "small_file_threshold_bytes": 65536,
    "elapsed_seconds": round(time.monotonic() - started, 1),
    "hardlinked_file_inodes": len(hardlinks),
    "backend": "directory entries (all owners; sizes not measured)" if args.dirents else ("Lustre FIDs / lazy MDT metadata" if args.lustre else "POSIX stat"),
    "ownership_filtered": not args.dirents,
    "sizes_measured": not args.dirents,
    "reclaim_estimate_accounts_for_links_outside_root": not args.dirents,
    "file_extensions": dict(suffixes.most_common()),
    "notes": ["Directory and symlink inodes included; links not followed.",
              "Per-directory totals overlap for parent/child directories; do not add these rows.",
              "Potentially freed inodes are a count estimate, not a deletion recommendation.",
              "Files can change during the scan; results are a point-in-time approximation."],
    "errors": errors,
    "directories": rows,
}
if args.dirents:
    result["directories"] = {
        path: {"entries": row["entries"], "files": row["files"],
               "directories": row["directories"], "symlinks": row["symlinks"],
               "unique_inodes": row["unique_owned_inodes"],
               "potentially_freed_inodes_upper_bound": row["potentially_freed_inodes"]}
        for path, row in rows.items()
    }
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(f"Saved {args.output}; elapsed={result['elapsed_seconds']}s; errors={len(errors)}", flush=True)
print("TOP-LEVEL: unique inodes | file entries | directories | symlinks | path" if args.dirents
      else "TOP-LEVEL: unique owned inodes | file entries | files <64 KiB | allocated GiB | path", flush=True)
top = [(path, row) for path, row in rows.items() if path != "." and "/" not in path]
for path, row in sorted(top, key=lambda item: item[1]["unique_owned_inodes"], reverse=True):
    if args.dirents:
        print(f"{row['unique_owned_inodes']:>10,} | {row['files']:>10,} | {row['directories']:>10,} | {row['symlinks']:>8,} | {path}", flush=True)
    else:
        print(f"{row['unique_owned_inodes']:>10,} | {row['files']:>10,} | {row['small_files']:>10,} | {row['unique_allocated_bytes']/2**30:>7.2f} | {path}", flush=True)
print("TOTAL", rows["."], flush=True)
