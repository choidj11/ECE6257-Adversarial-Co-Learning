#!/usr/bin/env python3
"""Publish an LFM3 dual (coder + tester, acg_llm) GRPO run (numeric metrics only) to Weights & Biases.

Uploads: per-step numeric metrics from metrics.jsonl (coder/*, tester/*, reward/*, perf/*), in-process
held-out evals (eval/*) at their step, the frozen protocol as config (without question ids), and optionally
an external eval summary (lfm_eval_vllm.py output). No prompts, completions, tests, data or weights.

Usage: .venv-rl/bin/python scripts/publish_lfm_dual_wandb.py --run artifacts/lfm-dual-llm-full \
         --name lfm3-dual-llm-full [--eval artifacts/lfm-dual-llm128-eval]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF = ("seconds", "gen_seconds", "reward_seconds", "train_seconds", "max_mem_gib", "device_mem_gib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval", type=Path, default=None)
    ap.add_argument("--tags", default="")
    ap.add_argument("--entity", default="6257-final-proj")
    ap.add_argument("--project", default="6257-adversarial-co-training")
    args = ap.parse_args()

    from dotenv import dotenv_values
    os.environ["WANDB_API_KEY"] = dotenv_values(ROOT / ".env")["WANDB_API_KEY"]
    os.environ.setdefault("WANDB_DIR", str(ROOT / "artifacts/wandb"))
    os.environ["WANDB_MODE"] = "online"
    import wandb

    protocol = json.loads((args.run / "protocol.json").read_text())
    protocol.pop("question_ids", None)
    records = [json.loads(l) for l in (args.run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    steps = {}
    for r in records:
        ev = r.get("event")
        if ev == "step":
            d = steps.setdefault(r["step"], {})
            for role in ("coder", "tester"):
                d.update({f"{role}/{k}": v for k, v in r[role].items() if isinstance(v, (int, float))})
            d.update({f"perf/{k}": r[k] for k in PERF if k in r})
            d.update({f"reward/{k}": v for k, v in r.items() if k not in PERF and k not in (
                "event", "step", "time", "coder", "tester", "micro_batch") and isinstance(v, (int, float))})
        elif ev == "eval":
            steps.setdefault(r["step"], {}).update({f"eval/{k}": v for k, v in r.items()
                                                    if k not in ("event", "step", "time") and isinstance(v, (int, float))})
    jobs = sorted({r["slurm_job_id"] for r in records if r.get("event") == "job_start"})
    final = max((r["step"] for r in records if r.get("event") == "step"), default=0)

    run = wandb.init(entity=args.entity, project=args.project, name=args.name, id=args.name,
                     resume="allow", job_type="train",
                     config={**protocol, "framework": "custom dual GRPO (TRL-equivalent loss) + vLLM 0.34.0 colocate",
                             "slurm_jobs": jobs,
                             "eval": "Jack held-out 64 (MBPP+ 36 / HumanEval+ 28), 1024 tokens; 4 HumanEval+ tasks "
                                     "exceed the 128 KB sandbox limit, so the ceiling is 60/64"},
                     tags=["lfm3", "acg_llm", "dual", "grpo", "black-box-tester"] + [t for t in args.tags.split(",") if t])
    run.define_metric("train_step")
    run.define_metric("*", step_metric="train_step")
    already = run.summary.get("train_step", -1)
    already = -1 if already is None else already
    for s in sorted(steps):
        if s > already:
            run.log({"train_step": s, **steps[s]})
    if args.eval is not None:
        for label in sorted(p.name for p in args.eval.iterdir() if (p / "summary.json").exists()):
            summ = json.loads((args.eval / label / "summary.json").read_text())
            for split in ("heldout", "train"):
                for k, v in summ[split].items():
                    if isinstance(v, (int, float)):
                        run.summary[f"eval_{label}/{split}_{k}"] = v
    run.summary["training/completed_steps"] = final
    run.finish()
    print(f"published {len(steps)} steps (final {final}) to {args.entity}/{args.project}/{args.name}")


if __name__ == "__main__":
    main()
