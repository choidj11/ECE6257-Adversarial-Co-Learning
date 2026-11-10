#!/usr/bin/env python3
"""Check real imports and offline tokenizer loading; optionally test a GPU rollout."""
import argparse
import importlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="Run only within a GPU allocation")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
report = {"checked_at": datetime.now(timezone.utc).isoformat(), "imports": {}, "gpu_requested": args.gpu}

for module, distribution in [
    ("torch", "torch"), ("transformers", "transformers"), ("datasets", "datasets"),
    ("ray", "ray"), ("vllm", "vllm"), ("verl", "verl"),
    ("flash_attn", "flash-attn"), ("sandbox_fusion", "sandbox-fusion"),
    ("bigcodebench", "bigcodebench"), ("evalplus", "evalplus"),
    ("cosmic_ray", "cosmic-ray"), ("coverage", "coverage"),
]:
    try:
        imported = importlib.import_module(module)
        version = importlib.metadata.version(distribution)
        report["imports"][module] = {"passed": True, "version": version}
        print(f"PASS import {module} {version}", flush=True)
    except Exception as exc:
        report["imports"][module] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"FAIL import {module}: {exc}", flush=True)

try:
    from huggingface_hub import constants as hub_constants
    import datasets.config as datasets_config
    for name, value in {
        "HF_HOME": hub_constants.HF_HOME,
        "HF_HUB_CACHE": hub_constants.HF_HUB_CACHE,
        "HF_DATASETS_CACHE": str(datasets_config.HF_DATASETS_CACHE),
    }.items():
        assert root in Path(value).resolve().parents, (name, value)
    from transformers import AutoConfig, AutoTokenizer
    model_path = root / "models/Qwen2.5-Coder-1.5B-Instruct"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": "Write a Python function that adds two integers."}],
                                             tokenize=True, add_generation_prompt=True)
    report["offline_tokenizer"] = {"passed": True, "model_type": config.model_type, "prompt_tokens": len(encoded)}
    print(f"PASS offline config/tokenizer: {config.model_type}, {len(encoded)} tokens", flush=True)
except Exception as exc:
    report["offline_tokenizer"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    print(f"FAIL offline config/tokenizer: {exc}", flush=True)

try:
    import torch
    report["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                       "devices": torch.cuda.device_count()}
    print(f"CUDA: {report['cuda']}", flush=True)
    if args.gpu:
        assert torch.cuda.is_available(), "No CUDA GPU visible in this allocation"
        report["cuda"]["device_name"] = torch.cuda.get_device_name(0)
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(model_path), tokenizer=str(model_path), dtype="bfloat16",
                  max_model_len=256, max_num_seqs=1, enforce_eager=True,
                  gpu_memory_utilization=0.35, disable_log_stats=True, tensor_parallel_size=1)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Write a Python function add(a, b) that returns their sum."}],
                                                tokenize=False, add_generation_prompt=True)
        results = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=48), use_tqdm=False)
        text = results[0].outputs[0].text
        assert text.strip(), "GPU generation returned no text"
        report["gpu_generation"] = {"passed": True, "text": text}
        print(f"PASS vLLM GPU generation: {text!r}", flush=True)
except Exception as exc:
    report["cuda_error"] = f"{type(exc).__name__}: {exc}"
    print(f"GPU/CUDA check: {exc}", flush=True)

report["passed"] = (all(x["passed"] for x in report["imports"].values())
                    and report["offline_tokenizer"]["passed"]
                    and (not args.gpu or report.get("gpu_generation", {}).get("passed", False)))
destination = root / "artifacts/setup" / ("environment-gpu-check.json" if args.gpu else "environment-check.json")
destination.write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if report["passed"] else 1)
