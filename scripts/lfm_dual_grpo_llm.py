#!/usr/bin/env python3
"""Coder + tester GRPO for LiquidAI/LFM3-1.7B with the Code-A1 acg_llm reward, shared black-box tests.

Per problem and step: the coder samples n codes, the tester samples n test sets from the *question only*
(no code in its prompt), and every code is run against every test set (n x n results). Everything else is
Code-A1's acg_llm:

  * tests are validated against GT_code (expected values are re-filled from GT_code, duplicates and
    non-executable asserts are invalid), via Code-A1 reward.evaluate;
  * r_a(i, j), r_b(i, j) from Code-A1 reward.compute_dual_model_reward (alpha = 0.5) for code i x test set j,
    with the Mistake Book's historical attack tests for the question;
  * coder reward  R_a(i) = mean_j r_a(i, j)                        (as acg_llm: mean over its tester samples)
  * tester reward R_b(j) = mean_{i: code i valid} r_b(i, j)        (all i if no code is valid)
  * Mistake Book update with Code-A1 reward.update_mistake_book (MistakeBook class copied verbatim from
    trainer_a1.py via ast).

Deviation from Code-A1: tester is black-box (Code-A1 puts one specific code in the tester prompt and samples n
test sets per code, n*n tester rollouts per problem, then trains on one code's n of them). Here the tester
generates n per problem and trains on all n.

The GRPO update mirrors the TRL 1.18.2 GRPOTrainer configuration used by lfm_grpo_gt.py, applied to each
model separately: group-normalised advantages (std + 1e-4), loss_type "dapo" (token-mean over the step),
clip 0.2 / 0.28, k3 KL beta 0.001 against a frozen copy of the base model, vLLM importance-sampling correction
(sequence_mask, max 3.0), 1 iteration per batch, AdamW lr 1e-6 cosine, wd 0.1, grad clip 1.0, fp32 master
weights and moments with bf16 autocast, gradient checkpointing. Both models share one colocated vLLM engine
(sleep mode level 2): before each generation the engine is loaded with that model's current weights.

Held-out eval runs in-process on the same engine every --eval-every steps (plus step 0 and the last step),
with the protocol of lfm_eval_vllm.py: Jack's 64 tasks, greedy pass@1, and 8 samples (T 0.7, top-p 0.95,
seed 2026) for sampled pass@1 / pass@8, 1024 tokens. Device memory (all processes, incl. vLLM) is sampled
through NVML and logged per step as device_mem_gib.
"""
import argparse
import ast
import json
import math
import os
import random
import re
import shutil
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from lfm_grpo_gt import A1, MODEL, ROOT, TRAIN

TESTER_SYSTEM = "You are a helpful test case generation assistant."
# Code-A1's non-SFT tester prompt (trainer_a1.py) with the Buggy Code section removed (black-box).
TESTER_USER = """
# Role
You are specializing in finding specific inputs that cause a candidate solution to behave differently from the requirements (`Question`).

# Task
Generate 8 assertion-based test cases to detect bugs in solutions to the `Question`.

# Strategy
1. **Attack Logic Gaps**: Analyze where a solution's logic might be "too simple" compared to the `Question`. Construct input `parameters` that hit these blind spots (e.g., missing constraints, misinterpreted rules, over-simplified logic).
2. **Prioritize Complexity**: Prefer complex input `parameters` (e.g., boundary values, nested loops, compound conditions, rare branches) over simple ones. Ensure every logical branch is stressed and every potential issue is covered.
3. **Zero Redundancy**: Do not brute-force generating trivial or repetitive tests. Only the first few generated tests will be evaluated, so quality and ordering matter more than quantity.

# Context
Question:
{question}

Function signature:
```python
{signature}
```

# Output Format
Output ALL the assert statements inside ONE '```python ... ```' code block.
Format: 'assert function_name(parameters) == answer'
"""


def load_mistake_book_class():
    """The MistakeBook class from Code-A1 trainer_a1.py, unchanged (its module needs verl/ray)."""
    src = (ROOT / "Code-A1/code/rl/trainer_a1.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "MistakeBook")
    ns = {"os": os, "json": json, "socket": socket, "List": List}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "trainer_a1.py", "exec"), ns)
    return ns["MistakeBook"]


def tester_messages(row):
    info = row["extra_info"]
    return [{"role": "system", "content": TESTER_SYSTEM},
            {"role": "user", "content": TESTER_USER.format(question=row["clean_description"],
                                                           signature=info["function_declaration"])}]


def dual_rewards(rows, coder_texts, tester_texts, n, mistake_book):
    """acg_llm rewards on the n x n (code, shared test set) grid of each problem."""
    P = len(rows)
    codes, code_info = A1.extract_code_from_response(coder_texts)
    tests, test_info = A1.extract_testcase_from_response(tester_texts)
    qids = [r["extra_info"]["question_id"] for r in rows]
    # code i (problem i // n) is evaluated with the n test sets of its problem: test_idx = i * n + j
    grid_tests = [tests[(i // n) * n + j] for i in range(P * n) for j in range(n)]
    gt_codes = [rows[i // n]["GT_code"] for i in range(P * n)]
    attack = A1.get_attack_testcases_from_mistake_book(mistake_book, qids, n)
    (valid, valid_tests, invalid_cnt, total_cnt, _, _, _, gen_passed, gen_total, gen_failed_msgs,
     _stdout, timeouts, _err, attack_total, attack_passed_tcs, gen_failed_tcs) = A1.evaluate(
        codes, grid_tests, gt_codes, [], attack)
    ra = [[0.0] * n for _ in range(P * n)]
    rb = [[0.0] * n for _ in range(P * n)]
    extra = []
    for i in range(P * n):
        for j in range(n):
            t, b = i * n + j, (i // n) * n + j
            out = A1.compute_dual_model_reward(
                is_code_valid=valid[i], gt_passed=-1, gt_total=-1, gt_failed_msgs=[],
                valid_generated_tests=valid_tests[t], invalid_count=invalid_cnt[t],
                total_generated_count=total_cnt[t], gen_passed=gen_passed[t], gen_failed_msgs=gen_failed_msgs[t],
                test_phase="adversarial",
                test_nonprintable_ratio=test_info["nonprintable_ratio"][b], test_space_ratio=test_info["space_ratio"][b],
                test_nonrelated_ratio=test_info["nonrelated_ratio"][b], test_overlong_ratio=test_info["overlong_ratio"][b],
                code_nonrelated_ratio=code_info["nonrelated_ratio"][i],
                attack_gen_passed=len(attack_passed_tcs[i]) if attack_passed_tcs else 0,
                attack_gen_total=attack_total[i] if attack_total else 0, alpha=0.5)
            ra[i][j], rb[i][j] = out["reward_a"], out["reward_b"]
            extra.append(out["reward_extra_info"])
    coder_r = [sum(ra[i]) / n for i in range(P * n)]
    tester_r = []
    for p in range(P):
        idx = [i for i in range(p * n, (p + 1) * n) if valid[i]] or list(range(p * n, (p + 1) * n))
        tester_r += [sum(rb[i][j] for i in idx) / len(idx) for j in range(n)]
    A1.update_mistake_book(mistake_book, question_ids=qids, n_samples_a=n, n_samples_b=n,
                           generated_code_list=codes, is_code_valid_list=valid, attack_gen_total=attack_total,
                           attack_gen_passed_tcs=attack_passed_tcs, gen_adv_total_phase2=gen_total,
                           gen_adv_passed_phase2=gen_passed, gen_failed_tcs_phase2=gen_failed_tcs)
    # tester-side stats per test set (first code of the problem: validity does not depend on the code)
    first = [(b // n) * n * n + (b % n) for b in range(P * n)]
    gen = [e.get("generated_test_results", {}) for e in extra]
    stats = {
        "code_valid": sum(valid) / len(valid),
        "sandbox_timeouts": sum(timeouts),
        "tests_total": sum(total_cnt[t] for t in first) / len(first),
        "tests_valid": sum(len(valid_tests[t]) for t in first) / len(first),
        "tests_invalid": sum(invalid_cnt[t] for t in first) / len(first),
        "gen_pass_rate": sum(g.get("gen_pass_rate", 0) for g in gen) / len(gen),
        "attack_rate": sum(g.get("final_attack_rate", 0) for g in gen) / len(gen),
        "mistake_book_attack_tests": sum(attack_total) / len(attack_total) if attack_total else 0,
        "mistake_book_attack_pass_rate": (sum(len(x) for x in attack_passed_tcs) / max(1, sum(attack_total))),
    }
    return coder_r, tester_r, stats


def group_advantages(rewards, n):
    import torch
    r = torch.tensor(rewards, dtype=torch.float32).view(-1, n)
    std = r.std(dim=1, keepdim=True)
    adv = ((r - r.mean(dim=1, keepdim=True)) / (std + 1e-4)).flatten()
    zero = torch.isclose(std, torch.zeros_like(std)).float().mean().item()
    return adv, zero


class Policy:
    """One trainable model: fp32 weights + AdamW, frozen fp32 reference, cosine LR."""

    def __init__(self, name, args, max_steps):
        import torch
        from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
        kw = dict(dtype=torch.float32, trust_remote_code=True, attn_implementation="sdpa")
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(str(MODEL), **kw).cuda()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        self.ref = AutoModelForCausalLM.from_pretrained(str(MODEL), **kw).cuda().eval().requires_grad_(False)
        # HF Trainer default: no weight decay on biases and norm weights
        decay = [p for n_, p in self.model.named_parameters() if p.ndim >= 2 and "norm" not in n_.lower()]
        no_decay = [p for n_, p in self.model.named_parameters() if not (p.ndim >= 2 and "norm" not in n_.lower())]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                      {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.999),
                                     eps=1e-8)
        self.sched = get_cosine_schedule_with_warmup(self.opt, 0, max_steps)

    def named_params(self):
        for n_, p in self.model.named_parameters():
            yield n_.replace("_checkpoint_wrapped_module.", ""), p.data


def per_token_logps(model, ids, attn, K, micro):
    import torch
    out = []
    for s in range(0, ids.size(0), micro):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids[s:s + micro], attention_mask=attn[s:s + micro],
                           logits_to_keep=K + 1).logits[:, :-1].float()
        tgt = ids[s:s + micro, -K:]
        out.append(torch.gather(logits.log_softmax(-1), -1, tgt.unsqueeze(-1)).squeeze(-1))
        del logits
    return torch.cat(out)


def grpo_update(pol, prompt_ids, comp_ids, vllm_logps, advantages, args, pad_id):
    """One TRL-equivalent GRPO optimizer step (dapo, 1 iteration, vLLM IS sequence_mask)."""
    import torch
    B = len(comp_ids)
    P = max(len(p) for p in prompt_ids)
    K = max(len(c) for c in comp_ids)
    ids = torch.full((B, P + K), pad_id, dtype=torch.long)
    attn = torch.zeros((B, P + K), dtype=torch.long)
    mask = torch.zeros((B, K))
    samp = torch.zeros((B, K))
    for b, (p, c, lp) in enumerate(zip(prompt_ids, comp_ids, vllm_logps)):
        ids[b, P - len(p):P] = torch.tensor(p)  # left-pad prompts, right-pad completions
        ids[b, P:P + len(c)] = torch.tensor(c)
        attn[b, P - len(p):P + len(c)] = 1
        mask[b, :len(c)] = 1
        samp[b, :len(c)] = torch.tensor([float("nan") if x is None else x for x in lp])
    ids, attn, mask, samp = ids.cuda(), attn.cuda(), mask.cuda(), samp.cuda()
    adv = advantages.cuda().unsqueeze(1)
    micro = args.micro_batch
    pol.model.eval()
    with torch.no_grad():
        old = per_token_logps(pol.model, ids, attn, K, micro)
        ref = per_token_logps(pol.ref, ids, attn, K, micro)
    diff = torch.nan_to_num((old - samp) * mask, nan=0.0).sum(-1, keepdim=True)
    is_ratio = torch.exp(diff)
    is_ratio = is_ratio.masked_fill(is_ratio > 3.0, 0.0)
    pol.model.train()
    n_tok = mask.sum()
    stats = {"kl": 0.0, "clip_frac": 0.0}
    for s in range(0, B, micro):
        sl = slice(s, s + micro)
        logp = per_token_logps(pol.model, ids[sl], attn[sl], K, micro)
        coef1 = torch.exp(logp - old[sl])
        coef2 = torch.clamp(coef1, 1 - 0.2, 1 + 0.28)
        loss_tok = -torch.min(coef1 * adv[sl], coef2 * adv[sl]) * is_ratio[sl]
        kl = torch.exp(ref[sl] - logp) - (ref[sl] - logp) - 1
        loss_tok = loss_tok + args.beta * kl
        loss = (loss_tok * mask[sl]).sum() / n_tok
        loss.backward()
        stats["kl"] += ((kl * mask[sl]).sum() / n_tok).item()
        stats["clip_frac"] += (((coef1 != coef2).float() * mask[sl]).sum() / n_tok).item()
    gn = torch.nn.utils.clip_grad_norm_(pol.model.parameters(), 1.0).item()
    pol.opt.step()
    pol.sched.step()
    pol.opt.zero_grad(set_to_none=True)
    stats.update(grad_norm=gn, lr=pol.sched.get_last_lr()[0], is_masked=(is_ratio == 0).float().mean().item(),
                 is_ratio_mean=is_ratio.mean().item())
    return stats


class Engine:
    """One colocated vLLM engine shared by both policies; weights are pushed before each generation."""

    def __init__(self, args):
        from vllm import LLM
        self.llm = LLM(model=str(MODEL), gpu_memory_utilization=args.vllm_mem, max_model_len=4096,
                       enable_sleep_mode=True, distributed_executor_backend="external_launcher", seed=0,
                       max_num_batched_tokens=4096, logprobs_mode="processed_logprobs", trust_remote_code=True)
        self.llm.sleep(level=2)

    def _wake(self, policy):
        import torch
        torch.cuda.empty_cache()
        self.llm.wake_up(tags=["weights"])
        runner = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        for name, p in policy.named_params():
            runner.load_weights([(name, p)])
        self.llm.wake_up(tags=["kv_cache"])
        self.llm.reset_prefix_cache()

    def generate_eval(self, policy, prompts, params_list):
        """Text prompts, one llm.generate per SamplingParams (as lfm_eval_vllm.py)."""
        self._wake(policy)
        outs = [self.llm.generate(prompts, sp, use_tqdm=False) for sp in params_list]
        self.llm.sleep(level=2)
        return outs

    def generate(self, policy, prompt_ids, n, max_tokens):
        from vllm import SamplingParams
        self._wake(policy)
        sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, logprobs=0)
        reqs = [{"prompt_token_ids": p} for p in prompt_ids for _ in range(n)]
        outs = self.llm.generate(reqs, sampling_params=sp, use_tqdm=False)
        self.llm.sleep(level=2)
        comp, logps, texts, trunc = [], [], [], []
        for o in outs:
            c = o.outputs[0]
            comp.append(list(c.token_ids))
            logps.append([next(iter(d.values())).logprob if d else None for d in c.logprobs])
            texts.append(c.text)
            trunc.append(c.finish_reason == "length")
        return comp, logps, texts, trunc


class DeviceMemory:
    """Peak device memory used (all processes on the GPU, incl. vLLM's pool), sampled via NVML."""

    def __init__(self, period=0.05):
        import pynvml
        pynvml.nvmlInit()
        import torch
        try:  # the GPU torch uses, matched by UUID (NVML ignores CUDA_VISIBLE_DEVICES)
            self.h = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
        except Exception:
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.nvml = pynvml
        self.total = pynvml.nvmlDeviceGetMemoryInfo(self.h).total / 2**30
        self.peak = 0.0
        threading.Thread(target=self._run, args=(period,), daemon=True).start()

    def _run(self, period):
        while True:
            self.peak = max(self.peak, self.nvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**30)
            time.sleep(period)

    def take(self):
        p, self.peak = self.peak, 0.0
        return p


def heldout_eval(engine, coder, tok, samples=8):
    from vllm import SamplingParams
    from lfm3_dual_baseline import extract_code, isolated_run
    from lfm3_passk_eval import pass_at_k
    from lfm_eval_vllm import heldout_rows
    held = heldout_rows()
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for r in held]
    greedy = SamplingParams(temperature=0.0, max_tokens=1024)
    sampled = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, n=samples, seed=2026)
    g, s = engine.generate_eval(coder, prompts, [greedy, sampled])
    jobs = []
    for row, go, so in zip(held, g, s):
        info = row["extra_info"]
        for text in [go.outputs[0].text] + [o.text for o in so.outputs]:
            jobs.append(extract_code(text) + "\n\ncandidate = " + info["function_name"] + "\n" + row["test"])
    with ThreadPoolExecutor(16) as ex:
        ok = [r[0] for r in ex.map(isolated_run, jobs)]
    k = 1 + samples
    n = len(held)
    gp = [ok[t * k] for t in range(n)]
    corr = [sum(ok[t * k + 1:(t + 1) * k]) for t in range(n)]
    return {"greedy_pass_at_1": sum(gp) / n,
            "sampled_pass_at_1": sum(pass_at_k(samples, c, 1) for c in corr) / n,
            "pass_at_8": sum(pass_at_k(samples, c, 8) for c in corr) / n,
            "correct_samples": sum(corr),
            "greedy_truncated": sum(go.outputs[0].finish_reason == "length" for go in g),
            "per_task": [{"task_id": r["extra_info"].get("question_id"), "greedy_pass": int(a), "sampled_correct": c}
                         for r, a, c in zip(held, gp, corr)]}


def complete_checkpoint(ckpt_dir: Path):
    found = []
    for p in ckpt_dir.glob("step-*"):
        m = re.fullmatch(r"step-(\d+)", p.name)
        if m and (p / "COMPLETE").exists():
            found.append((int(m.group(1)), p))
    return max(found)[1] if found else None


def save_checkpoint(ckpt_dir, step, coder, tester, mistake_book, keep):
    import torch
    d = ckpt_dir / f"step-{step}"
    if d.exists():
        shutil.rmtree(d)
    for pol in (coder, tester):
        pol.model.save_pretrained(d / pol.name, safe_serialization=True)
        for f in MODEL.glob("*"):  # tokenizer, chat template, remote-code files for standalone loading
            if f.suffix in (".py", ".json", ".jinja") and not (d / pol.name / f.name).exists():
                shutil.copy(f, d / pol.name / f.name)
        torch.save({"opt": pol.opt.state_dict(), "sched": pol.sched.state_dict()}, d / f"{pol.name}_optim.pt")
    (d / "mistake_book.json").write_text(json.dumps(mistake_book.mistake_book))
    torch.save({"step": step, "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                "py": random.getstate()}, d / "state.pt")
    (d / "COMPLETE").write_text(str(step))
    old = sorted((int(p.name.split("-")[1]), p) for p in ckpt_dir.glob("step-*") if p != d)
    for _, p in old[:max(0, len(old) - (keep - 1))]:
        shutil.rmtree(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-file", type=Path, default=TRAIN)
    ap.add_argument("--problems", type=int, default=128)
    ap.add_argument("--rollout-n", type=int, default=8)
    ap.add_argument("--problems-per-step", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--save-at", default="half,last",
                    help="comma list of steps, 'half', 'last', or 'none'")
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=100, help="0 disables held-out eval")
    ap.add_argument("--stop-after-step", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.001)
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    args = ap.parse_args()
    for k, v in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"), ("MASTER_ADDR", "127.0.0.1"),
                 ("MASTER_PORT", str(29500 + os.getpid() % 1000))):
        os.environ.setdefault(k, v)

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    random.seed(42)
    torch.manual_seed(42)
    rows = pq.read_table(args.train_file).slice(0, args.problems).to_pylist()
    assert len(rows) == args.problems
    out, ckpt_dir = args.output, args.output / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    protocol = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k not in ("output", "stop_after_step", "save_at", "keep", "eval_every", "micro_batch")}
    protocol.update(model=str(MODEL), question_ids=[r["extra_info"]["question_id"] for r in rows],
                    reward="Code-A1 acg_llm, alpha 0.5, black-box tester, shared n x n grid, Mistake Book",
                    precision="fp32 master+AdamW, bf16 autocast")
    proto_path = out / "protocol.json"
    if proto_path.exists():
        assert json.loads(proto_path.read_text()) == protocol, "protocol changed; refusing to resume"
    else:
        proto_path.write_text(json.dumps(protocol, indent=2) + "\n")
    metrics = open(out / "metrics.jsonl", "a", buffering=1)
    log = lambda **kw: metrics.write(json.dumps({"time": time.time(), **kw}) + "\n")
    t_job = time.time()
    log(event="job_start", slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name())

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    def encode(msgs):
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                   add_special_tokens=False)["input_ids"]
    coder_prompts = [encode(r["prompt"]) for r in rows]
    tester_prompts = [encode(tester_messages(r)) for r in rows]

    coder, tester = Policy("coder", args, args.max_steps), Policy("tester", args, args.max_steps)
    MistakeBook = load_mistake_book_class()
    mistake_book = MistakeBook(str(out / "mistake_book_unused"), 0)
    start = 0
    resume = complete_checkpoint(ckpt_dir)
    if resume is not None:
        from safetensors.torch import load_file
        for pol in (coder, tester):
            sd = {}
            for f in sorted((resume / pol.name).glob("*.safetensors")):
                sd.update(load_file(str(f)))
            pol.model.load_state_dict(sd, strict=False)
            st = torch.load(resume / f"{pol.name}_optim.pt", weights_only=False)
            pol.opt.load_state_dict(st["opt"])
            pol.sched.load_state_dict(st["sched"])
        mistake_book.mistake_book = json.loads((resume / "mistake_book.json").read_text())
        st = torch.load(resume / "state.pt", weights_only=False)
        start = st["step"]
        torch.set_rng_state(st["torch"]); torch.cuda.set_rng_state(st["cuda"]); random.setstate(st["py"])
    log(event="resume_from", path=str(resume), step=start,
        param_dtype=str(next(coder.model.parameters()).dtype))
    save_at = set()
    for x in args.save_at.split(","):
        x = x.strip()
        if x == "half":
            save_at.add(args.max_steps // 2)
        elif x == "last":
            save_at.add(args.max_steps)
        elif x and x != "none":
            save_at.add(int(x))
    engine = Engine(args)
    devmem = DeviceMemory()
    log(event="setup_done", seconds=time.time() - t_job, max_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
        device_mem_gib=devmem.take(), device_total_gib=devmem.total, save_at=sorted(save_at))

    def run_eval(step):
        te = time.time()
        res = heldout_eval(engine, coder, tok)
        (out / "eval").mkdir(exist_ok=True)
        (out / "eval" / f"step-{step}.json").write_text(json.dumps(res, indent=1) + "\n")
        log(event="eval", step=step, seconds=time.time() - te, **{k: v for k, v in res.items() if k != "per_task"})

    if args.eval_every and start == 0:
        run_eval(0)

    n, pps = args.rollout_n, args.problems_per_step
    for step in range(start + 1, args.max_steps + 1):
        t0 = time.time()
        lo = ((step - 1) * pps) % len(rows)
        idx = [(lo + k) % len(rows) for k in range(pps)]
        batch = [rows[k] for k in idx]
        cp, tp = [coder_prompts[k] for k in idx], [tester_prompts[k] for k in idx]
        c_comp, c_lp, c_txt, c_tr = engine.generate(coder, cp, n, args.max_completion_length)
        t_comp, t_lp, t_txt, t_tr = engine.generate(tester, tp, n, args.max_completion_length)
        t_gen = time.time()
        coder_r, tester_r, rstats = dual_rewards(batch, c_txt, t_txt, n, mistake_book)
        t_rew = time.time()
        c_adv, c_zero = group_advantages(coder_r, n)
        t_adv, t_zero = group_advantages(tester_r, n)
        rep = lambda ps: [p for p in ps for _ in range(n)]
        cs = grpo_update(coder, rep(cp), c_comp, c_lp, c_adv, args, pad_id)
        ts = grpo_update(tester, rep(tp), t_comp, t_lp, t_adv, args, pad_id)
        t_end = time.time()
        mean = lambda x: sum(x) / len(x)
        log(event="step", step=step, seconds=t_end - t0, gen_seconds=t_gen - t0, reward_seconds=t_rew - t_gen,
            train_seconds=t_end - t_rew, max_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
            device_mem_gib=devmem.take(), micro_batch=args.micro_batch,
            question_ids=[r["extra_info"]["question_id"] for r in batch],
            coder={"reward": mean(coder_r), "zero_std_groups": c_zero, "length": mean([len(c) for c in c_comp]),
                   "clipped_ratio": mean(c_tr), **cs},
            tester={"reward": mean(tester_r), "zero_std_groups": t_zero, "length": mean([len(c) for c in t_comp]),
                    "clipped_ratio": mean(t_tr), **ts},
            **rstats)
        torch.cuda.reset_peak_memory_stats()
        if step == 1:
            (out / "samples_step1.json").write_text(json.dumps(
                {"coder": c_txt[:2], "tester": t_txt[:2], "coder_reward": coder_r[:2], "tester_reward": tester_r[:2]},
                indent=1))
        stop = args.stop_after_step is not None and step >= args.stop_after_step
        if args.eval_every and (step % args.eval_every == 0 or step == args.max_steps):
            run_eval(step)
        if step in save_at:
            ts0 = time.time()
            save_checkpoint(ckpt_dir, step, coder, tester, mistake_book, args.keep)
            log(event="checkpoint", step=step, seconds=time.time() - ts0)
        if stop:
            break
    log(event="job_end", seconds=time.time() - t_job,
        optimizer_state_dtypes=sorted({str(v.dtype) for s in coder.opt.state.values() for v in s.values()
                                       if torch.is_tensor(v) and v.ndim}))


if __name__ == "__main__":
    main()
