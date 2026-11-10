#!/usr/bin/env python3
"""Prepare tiny official-data slices and resolved local Code-A1 configurations."""
import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/Code-A1/Code-A1"
OUT = ROOT / "artifacts/week12"
OUT.mkdir(parents=True, exist_ok=True)
model = str(ROOT / "models/Qwen2.5-Coder-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
train = next((ROOT / "data/Code-A1/train_data").glob("*.parquet"))
val = next((ROOT / "data/Code-A1/val_data").glob("*.parquet"))
table = pq.read_table(train)
rows = table.to_pylist()
# Preserve source columns and labels; select small training examples for a short check.
lengths = [(len(tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True)), i)
           for i, r in enumerate(rows)]
selected = [i for length, i in sorted(lengths) if length <= 768 and rows[i]["GT_code"]][:4]
assert len(selected) == 4
pq.write_table(table.take(pa.array(selected)), OUT / "train.parquet")
pq.write_table(pq.read_table(val).slice(0, 2), OUT / "val.parquet")

base = OmegaConf.load(UPSTREAM / "verl/verl/trainer/config/ppo_trainer.yaml")
official = OmegaConf.load(UPSTREAM / "code/rl/config/1.5B_A1.yaml")
local = {
    "prefix": str(UPSTREAM),
    "data": {"train_files": str(train), "val_files": str(val),
             "cache_dir": str(ROOT / ".cache/verl"), "filter_overlong_prompts_workers": 2,
             "dataloader_num_workers": 0, "trust_remote_code": False},
    "actor_rollout_ref": {"model": {"path": model}},
    "trainer": {"logger": ["console"], "default_local_dir": str(ROOT / "artifacts/baseline/coder"),
                "rollout_data_dir": str(ROOT / "artifacts/baseline/coder-rollouts"),
                "validation_data_dir": str(ROOT / "artifacts/baseline/coder-validation"),
                "max_actor_ckpt_to_keep": 1, "max_critic_ckpt_to_keep": 1},
    "other": {"actor_rollout_ref": {"model": {"path": model}},
              "trainer": {"default_local_dir": str(ROOT / "artifacts/baseline/tester"),
                          "rollout_data_dir": str(ROOT / "artifacts/baseline/tester-rollouts"),
                          "validation_data_dir": str(ROOT / "artifacts/baseline/tester-validation")}},
}
baseline = OmegaConf.merge(base, official, local)
OmegaConf.resolve(baseline)
OmegaConf.save(baseline, OUT / "baseline-local.yaml")
smoke = OmegaConf.merge(baseline, {
    "data": {"train_files": str(OUT / "train.parquet"), "val_files": str(OUT / "val.parquet"),
             "max_prompt_length": 1024, "max_response_length": 512,
             "train_batch_size": 2, "val_batch_size": 2, "shuffle": False},
    "actor_rollout_ref": {
        "actor": {"ppo_mini_batch_size": 2, "ppo_max_token_len_per_gpu": 4096,
                  "optim": {"lr_warmup_steps": 0}},
        "ref": {"log_prob_micro_batch_size_per_gpu": 1, "log_prob_max_token_len_per_gpu": 4096},
        "rollout": {"n": 2, "prompt_length": 1024, "response_length": 512,
                    "max_model_len": 4096, "max_num_batched_tokens": 4096,
                    "gpu_memory_utilization": 0.25, "enforce_eager": True,
                    "free_cache_engine": True, "enable_chunked_prefill": False,
                    "log_prob_micro_batch_size_per_gpu": 1, "log_prob_max_token_len_per_gpu": 4096}},
    "trainer": {"n_gpus_per_node": 1, "total_epochs": 1, "total_training_steps": 1,
                "experiment_name": "week12-smoke", "save_freq": -1, "test_freq": -1,
                "val_before_train": False, "resume_mode": "disable",
                "rollout_data_dir": str(OUT / "coder-rollouts"),
                "validation_data_dir": str(OUT / "coder-validation"),
                "default_local_dir": str(OUT / "coder-checkpoints")},
    "other": {"data": {"max_prompt_length": 2048},
              "actor_rollout_ref": {"rollout": {"n": 2}}, "update_n": 1,
              "trainer": {"n_gpus_per_node": 1, "rollout_data_dir": str(OUT / "tester-rollouts"),
                          "validation_data_dir": str(OUT / "tester-validation"),
                          "default_local_dir": str(OUT / "tester-checkpoints")}},
})
OmegaConf.save(smoke, OUT / "smoke.yaml")

# Validate the actual verl dataset and collator, not only Parquet metadata.
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl import DataProto
dataset = RLHFDataset(str(OUT / "train.parquet"), tokenizer, smoke.data)
batch = DataProto.from_single_dict(collate_fn([dataset[0], dataset[1]]))
assert len(dataset) == 4 and len(batch) == 2
assert "GT_code" in batch.non_tensor_batch and "GT_test" in batch.non_tensor_batch
report = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": True,
          "official_train_rows": len(rows), "selected_source_rows": selected,
          "question_ids": [str(rows[i].get("question_id") or rows[i].get("extra_info", {}).get("index") or i)
                           for i in selected],
          "batch_size": len(batch), "tensor_keys": list(batch.batch.keys()),
          "non_tensor_keys": list(batch.non_tensor_batch),
          "config": str(OUT / "smoke.yaml"),
          "note": "One-step infrastructure check; not a baseline training result. Checkpoint saving disabled."}
(ROOT / "artifacts/setup/trainer-data-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
