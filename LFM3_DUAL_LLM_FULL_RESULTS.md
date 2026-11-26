# LFM3-1.7B coder + tester GRPO with acg_llm: full KodCode, 1 epoch

Date: 2026-11-24. Training job 7270663 (6 h 54 min, 1x H200). Job 7270641 failed before training: CUDA could not initialise on node atl1-1-03-014-16-0.
W&B: https://wandb.ai/6257-final-proj/6257-adversarial-co-training/runs/lfm3-dual-llm-full
(The 128-problem run is `lfm3-dual-llm128`; see `LFM3_DUAL_LLM_RESULTS.md` for its method and results.)

Slurm marks the job FAILED, but training finished. `job_end` was logged after step 1614 and both checkpoints are complete. The non-zero exit status comes from vLLM's `cumem_allocator` error at process teardown, the same one seen at the end of every earlier run.

## Held-out result (in-process eval, Jack's 64 tasks, 1024 tokens)

| step | greedy pass@1 | sampled pass@1 (8) | pass@8 | correct / 512 | greedy cut off |
|---:|---:|---:|---:|---:|---:|
| 0 (base) | 59.4% | 61.9% | 81.2% | 317 | 34 |
| 100 | 90.6% | 87.7% | 93.8% | 449 | 0 |
| 200 | 90.6% | 91.2% | 93.8% | 467 | 1 |
| 300 | 93.8% | 92.2% | 93.8% | 472 | 0 |
| 500 | 93.8% | 92.6% | 93.8% | 474 | 0 |
| 700 | 89.1% | 90.8% | 92.2% | 465 | 2 |
| 800 | 93.8% | 93.0% | 93.8% | 476 | 0 |
| 1000 | 93.8% | 92.8% | 93.8% | 475 | 0 |
| 1200 | 93.8% | 93.2% | 93.8% | 477 | 0 |
| 1400 | 92.2% | 92.8% | 93.8% | 475 | 1 |
| 1614 (final) | **92.2%** | **93.2%** | **93.8%** | 477 | 1 |

All 18 evals (every 100 steps) are in `artifacts/lfm-dual-llm-full/eval/step-*.json` and on W&B as `eval/*`.

For comparison, the 128-problem runs scored greedy 84.4% (both GT and dual). The step-0 in-process base gives sampled 61.9% and pass@8 81.2%, a little above the standalone eval (58.8% / 78.1%) because the sampling RNG differs. Compare within one series.

- **Final vs step 0, per task (sampled correct count):** up on 33, down on 0.
- **Final vs the 128-problem dual run:** up on 17, down on 0.
- **On tasks where base was cut off at 1024 tokens (30 tasks),** greedy passes went from 8 at step 0 to 22 for the 128-problem dual run and 27 at the final step.
- **On tasks where base finished (34),** greedy passes are 30 at step 0, 32 for the 128-problem dual run and 32 at the final step.

### The held-out set is saturated: its ceiling is 60/64

Four HumanEval+ tasks (21, 34, 38, 56) fail for every model, **including the GT solution**. The reason is that `isolated_run` in `scripts/lfm3_dual_baseline.py` rejects programs larger than 128 KB (`source_too_large`), and these four tasks' test files are larger than that.

- The highest score possible is 60/64 = 93.75%.
- pass@8 has sat at that ceiling since step 100.
- Greedy pass@1 is 59–60 of the 60 solvable tasks from step 300 on.

The same limit applies to every earlier result (base, GT run, 128-problem dual run), so those comparisons still hold. But **this eval can no longer tell later checkpoints apart.** A larger held-out set (the full MBPP+ / HumanEval+ validation sets, the 128 KB limit raised, and a 2048–4096-token budget) is needed.

## Training curve (averages over 200 steps)

| steps | coder reward | coder length | coder cut-off | coder no-signal | coder KL | tester reward | tester length | tester no-signal | tester KL | valid / total tests | attack rate | valid code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–200 | 0.707 | 386 | 8.2% | 0.25 | 0.059 | 0.424 | 196 | 0.04 | 0.10 | 7.3 / 7.6 | 0.235 | 90.6% |
| 201–400 | 0.786 | 368 | 2.2% | 0.33 | 0.093 | 0.464 | 158 | 0.09 | 0.22 | 7.9 / 8.1 | 0.192 | 96.9% |
| 401–600 | 0.812 | 371 | 1.7% | 0.41 | 0.107 | 0.450 | 167 | 0.19 | 0.28 | 8.7 / 9.1 | 0.157 | 97.3% |
| 601–800 | 0.826 | 350 | 3.5% | 0.49 | 0.117 | 0.461 | 156 | 0.36 | 0.31 | 8.4 / 8.6 | 0.151 | 95.5% |
| 801–1000 | 0.856 | 276 | 0.8% | 0.51 | 0.141 | 0.477 | 141 | 0.42 | 0.32 | 7.9 / 8.1 | 0.123 | 98.5% |
| 1001–1200 | 0.876 | 303 | 0.7% | 0.59 | 0.131 | 0.475 | 143 | 0.50 | 0.45 | 7.8 / 8.0 | 0.104 | 98.7% |
| 1201–1400 | 0.881 | 296 | 0.8% | 0.57 | 0.132 | 0.483 | 145 | 0.48 | 0.33 | 7.9 / 8.0 | 0.106 | 98.3% |
| 1401–1614 | 0.875 | 315 | 1.0% | 0.56 | 0.137 | 0.481 | 143 | 0.48 | 0.34 | 7.8 / 8.0 | 0.110 | 98.1% |

("No-signal" is the fraction of groups whose 8 rewards are all equal.)

Reading the curve:
- **The coder wins the adversarial game.** Its reward rises from 0.71 to 0.88 and it almost never runs out of tokens (cut-off < 1%). The tester's attack rate falls from 0.24 to about 0.10.
- **The tester mostly earns the validity half of its reward.** It writes about 8 asserts per set and about 98% of them are valid. With α = 0.5, its reward ≈ 0.5·valid + 0.5·attack ≈ 0.48.
- **Learning signal shrinks for both models.** By the second half, more than half of the coder groups (all 8 codes pass) and about half of the tester groups give zero advantage.
- **KL is much larger than in the 128-problem runs** (< 0.01 there): coder about 0.14, tester about 0.3–0.45. The tester drifts more from its base.
- **Mistake Book:** it ends with 76,133 failing tests over 5,339 of the 9,684 questions. None were replayed, because with 1 epoch each question is seen once. A second epoch would start from this buffer, which is saved in `ckpt/step-1614/mistake_book.json`.

## Setup

Same method as the 128-problem dual run (`LFM3_DUAL_LLM_RESULTS.md`: black-box tester, shared 8×8 grid, Code-A1 `acg_llm`, α = 0.5, TRL-equivalent GRPO, fp32 AdamW, lr 1e-6 cosine, β = 0.001). Changes for this run:

| | |
|---|---|
| Data | `artifacts/study/train_full.parquet`: all 9,684 KodCode problems with prompts ≤ 1024 tokens (Qwen2.5-Coder tokenizer), `random.Random(42)` order. Built by `scripts/prepare_full_kodcode.py` and checked: the first 128 rows equal `train.parquet`, and the excluded rows are 5370, 7641, 8633, 8853, matching the study protocol. |
| Batch | 6 problems per step, coder 48 + tester 48 completions; micro-batch 16 |
| Schedule | 1,614 steps = 1 epoch, cosine decay over the full run |
| Eval | in-process on the training vLLM engine at step 0, every 100 steps and the final step |
| Checkpoints | step 807 (half, deleted 2026-11-25 to free quota) and step 1614 (final, kept): coder, tester, both fp32 optimizer states, the Mistake Book and RNG state, about 41 GB: `artifacts/lfm-dual-llm-full/ckpt/step-1614/` |

## Cost

| | |
|---|---|
| Wall-clock | 6 h 54 min (24,842 s in the script) |
| Step time | mean 15.0 s, median 13.8 s, p95 24.5 s, max 44 s |
| Breakdown per step | generation 5.2 s, sandbox reward 2.6 s, update 7.2 s |
| Sandbox timeouts | 390, spread over 222 steps (about +10 s each) |
| Eval | 17 s on average, 18 evals |
| Peak device memory (NVML, including vLLM) | 100.9 GiB of 140.4 GiB |

This was faster than the 11.5 h estimate. Coder outputs shrank from about 690 to about 300 tokens and tester outputs from about 490 to about 140, so generation and updates got cheaper as training went on.

Storage: only the final checkpoint is kept (about 41 GB).

## Takeaways

1. Over one epoch of KodCode, dual training with `acg_llm` takes held-out greedy pass@1 from 59.4% to 92.2–93.8%, at the 60/64 ceiling of this eval. Most of the gain comes in the first 100–300 steps.
2. The tester is pushed toward valid but easy tests; its attack rate falls to about 0.10. With α = 0.5 that is a stable equilibrium in which the coder gets little signal (more than half of its groups are all-pass). Raising the weight on attack, or a second epoch that turns on the Mistake Book, are the obvious next experiments.
3. The held-out eval has to change before further comparisons: fix the 128 KB sandbox limit, use the full validation sets, and use a larger token budget.
