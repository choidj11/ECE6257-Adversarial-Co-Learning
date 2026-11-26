#!/usr/bin/env python3
"""Full KodCode training file in the study's order: same <=1024-token prompt filter (Qwen2.5-Coder tokenizer)
and random.Random(42) shuffle as prepare_experiments.py, all eligible rows kept. The first 128 rows equal
artifacts/study/train.parquet."""
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/study/train_full.parquet"
# argv[1]: a Qwen2.5-Coder-1.5B-Instruct tokenizer directory (the model itself is no longer in models/)
tok_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models/Qwen2.5-Coder-1.5B-Instruct")
tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
table = pq.read_table(next((ROOT / "data/Code-A1/train_data").glob("*.parquet")))
rows = table.to_pylist()
eligible = [i for i, r in enumerate(rows)
            if len(tok.apply_chat_template(r["prompt"], add_generation_prompt=True)) <= 1024]
random.Random(42).shuffle(eligible)
full = table.take(pa.array(eligible))
pq.write_table(full, OUT)
small = pq.read_table(ROOT / "artifacts/study/train.parquet")
assert full.slice(0, 128).column("extra_info").to_pylist() == small.column("extra_info").to_pylist()
print(json.dumps({"source_rows": len(rows), "eligible_rows": len(eligible), "out": str(OUT),
                  "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))
