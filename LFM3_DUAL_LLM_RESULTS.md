# LFM3-1.7B coder + tester GRPO with the acg_llm reward: 128 problems, 1 epoch

Date: 2026-11-23. Training job 7268763 (12 min 30 s, 1x H200). Eval job 7269122 (step 32 coder).
Base and GT-reward numbers come from `LFM3_GT_CODER_RESULTS.md` (same eval script and backend).

## Result

| Held-out 64 (36 MBPP+, 28 HumanEval+) | base | GT coder-only, step 32 | **dual acg_llm, step 32** |
|---|---:|---:|---:|
| greedy pass@1 | 59.4% (38/64) | 84.4% (54/64) | **84.4%** (54/64) |
| sampled pass@1 (8 samples) | 58.8% | 83.2% | **82.6%** |
| pass@8 | 78.1% | 90.6% | **92.2%** |
| correct samples | 301/512 | 426/512 | 423/512 |
| greedy answers cut off at 1024 tokens | 30/64 | 6/64 | 6/64 |

| Train 128 (KodCode, scored with acg_gt) | base | GT, step 32 | dual, step 32 |
|---|---:|---:|---:|
| greedy mean reward (GT pass fraction) | 0.236 | 0.629 | 0.618 |
| greedy all GT tests pass | 19.5% | 51.6% | 50.8% |
| sampled mean reward | 0.235 | 0.590 | 0.579 |
| pass@8 | 42.2% | 69.5% | 68.0% |

- **Dual vs base:** per task, the sampled correct count went up on 33 of 64 tasks, stayed the same on 29 and went down on 2.
- **Dual vs GT:** up on 7, down on 9, same on 48. Greedy pass/fail agrees on 58 of 64 tasks. Within noise, the two runs are the same.
- So the coder trained only on tester-generated tests (checked against GT code, never against KodCode's GT tests) reaches the same held-out score as the coder trained on the GT tests.

### Where the gain comes from (same picture as the GT run)

| | tasks | base greedy pass | GT step 32 | dual step 32 |
|---|---:|---:|---:|---:|
| base answer **not** cut off | 34 | 32 | 31 | 32 |
| base answer cut off at 1024 | 30 | 6 | 23 | 22 |

Almost all of the gain is on tasks where base ran out of tokens. On tasks where base finished, all three models are tied. As in the GT run, the main thing learned under a 1024-token budget is to write shorter answers that finish inside the budget.

## Method

Code-A1 `acg_llm` reward (`Code-A1/code/rl/reward.py`), with one change: **the tester is black-box and its tests are shared across the problem's codes**.

| | Code-A1 (paper / upstream) | this run |
|---|---|---|
| Tester prompt | question + one specific candidate code (white-box) | question + function signature only (black-box); Code-A1's non-SFT prompt with the `Buggy Code` part removed |
| Tester rollouts per problem | 8 codes × 8 = 64 | 8 |
| Grid evaluated | each code × its own 8 test sets | each of the 8 codes × the same 8 test sets (8×8) |
| Coder reward R_a(i) | mean over its 8 tester samples of r_a | mean over the 8 shared test sets of r_a (unchanged formula) |
| Tester reward | r_b of (code, test set); update only the 8 samples of the code with the highest r_b std (`update_n=1`) | R_b(j) = mean of r_b(i, j) over valid codes i; all 8 samples are updated |
| Mistake Book | yes | yes (class copied verbatim from `trainer_a1.py`) |

Unchanged from Code-A1:
- **How tests are checked:** each tester assert's call is re-run on `GT_code`. A duplicate, a call that errors, or a predicted answer that differs from GT counts as invalid. The call is still run against the code using GT's answer.
- **Formulas:** `compute_dual_model_reward` with α = 0.5:
  - `r_a` = pass rate on the new tests, averaged with the historical pass rate when there are historical tests;
  - `r_b = 0.5·valid_ratio + 0.5·attack_rate − nonprintable_ratio`, and 0 for the valid/attack part when the code is invalid.

## Training parameters

| | |
|---|---|
| Models | coder and tester, both initialised from `models/LFM3-1.7B`, each with its own frozen fp32 reference model for KL |
| Data | `artifacts/study/train.parquet`: the same 128 KodCode problems as the GT run, in the same order |
| Batch | 4 problems per step: coder 4 × 8 = 32 completions, tester 4 × 8 = 32 completions |
| Schedule | 32 steps = 1 epoch, one optimizer step per model per step |
| GRPO | written to match TRL 1.18.2 `GRPOTrainer` as set up for the GT run: advantages normalised within each group (std + 1e-4), `dapo` token-mean loss, clip 0.2 / 0.28, k3 KL β = 0.001, vLLM importance-sampling correction (sequence_mask, max 3), 1 iteration per batch |
| Optimizer | AdamW, lr 1e-6, cosine decay, 0 warmup, weight decay 0.1 (none on norms or biases), grad clip 1.0, for each model |
| Precision | fp32 weights and AdamW moments (verified at job end), bf16 autocast, gradient checkpointing |
| Rollouts | one vLLM 0.34 engine on the same GPU, shared by both models (sleep level 2; the current weights are loaded before each generation); temperature 1.0, top-p 1.0, max 1024 new tokens for both models |
| Sandbox | Code-A1 `evaluate` via Sandbox Fusion, 10 s per program |
| Checkpoint | only at step 32: `artifacts/lfm-dual-llm128/ckpt/step-32/{coder,tester}` plus optimizer states, 39 GB |

Code: `scripts/lfm_dual_grpo_llm.py`, `scripts/lfm_dual_grpo_llm.sbatch`. Per-step metrics: `artifacts/lfm-dual-llm128/metrics.jsonl`.

## Training curve (averages over 8 steps)

| steps | coder reward | coder cut-off | coder length | coder no-signal groups | tester reward | tester length | tester no-signal groups | valid tests / set | attack rate | valid code | coder KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–8 | 0.351 | 48% | 685 | 0.47 | 0.203 | 487 | 0.28 | 5.2 | 0.34 | 47% | 0.001 |
| 9–16 | 0.448 | 31% | 573 | 0.31 | 0.240 | 429 | 0.16 | 5.1 | 0.32 | 55% | 0.004 |
| 17–24 | 0.483 | 33% | 606 | 0.31 | 0.333 | 314 | 0.13 | 6.0 | 0.34 | 64% | 0.005 |
| 25–32 | 0.525 | 33% | 590 | 0.22 | 0.306 | 324 | 0.03 | 6.1 | 0.33 | 65% | 0.006 |

- **Coder reward:** pass rate on the tester's validated tests.
- **"Valid code":** the program ran. Most invalid code is a truncated answer with no closing code fence.
- **Tester:** it learned to write more valid asserts and much shorter answers. Its attack rate stayed flat at about 0.33 while the coder got stronger.
- **Mistake Book:** it never replayed a test, because with 1 epoch every question is seen only once (see below).

## Cost

| part | time |
|---|---:|
| setup (2 policies + 2 references + vLLM) | 111 s |
| 32 steps | 558 s total, mean 17.4 s / step (median 16.2 s) |
| generation, both models | 7.9 s / step |
| sandbox reward | 2.2 s / step (10–20 s on the 4 steps with a sandbox timeout) |
| update, both models | 7.4 s / step |
| final checkpoint | 53 s |
| peak GPU memory | 77.0 GiB of 141 GiB |

### Full KodCode estimate (1 epoch)

KodCode has 9,688 problems; 9,684 remain after the ≤1024-token prompt filter.

| problems / step | steps | step time (est.) | training time |
|---:|---:|---:|---:|
| 4 (as in this run) | 2,421 | 17.4 s (measured) | ≈ 11.7 h |
| 6 | 1,614 | ≈ 23 s (generation grows less than linearly, update grows linearly) | ≈ 10.3 h |

The 6-problem step time is an estimate and has to be measured. An eval every 100 steps adds about 1 min each in-process (about 17–25 evals, so about 20–30 min). The coe-gpu partition allows jobs of up to 16 h.

## About the Mistake Book

It is part of the Code-A1 paper (Sec. 3.3, App. A.3). It is a per-question replay buffer of tests that candidate code failed. A test's frequency goes up each time it causes a failure and down each time it is passed, and the test is removed when the frequency reaches 0.

The paper's ablation (Table 3) shows that removing it lowers HumanEval+ from 72.69 to 69.66. The authors say its main job is to stop the coder forgetting bugs it had already fixed.

**The Mistake Book only acts from the second epoch onward.** The paper trains with 256 questions per step for 3 epochs over 9,688 questions. Its Fig. 4 shows historical tests first being used at step 38, which is after the first epoch. Any 1-epoch run, including a 1-epoch full-KodCode run, fills the buffer but never uses it.

## Takeaways

1. With a black-box tester whose tests are shared across codes, dual training with `acg_llm` matches the GT-reward coder on held-out data (84.4% greedy in both). It costs 12.5 min on 1 H200 and uses no KodCode GT tests.
2. Both runs gain mainly from finishing inside 1024 tokens, so this setup cannot yet show that one reward is better than the other. A larger budget (2048 / 4096 tokens), more data, or ≥2 epochs (so the Mistake Book acts) is needed to tell them apart.
3. The tester is learning (more valid tests, much shorter outputs, fewer groups with no signal) while its attack rate holds at about 0.33.
