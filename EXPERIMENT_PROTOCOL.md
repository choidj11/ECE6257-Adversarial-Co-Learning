# Fixed-Budget Comparison Protocol

Frozen on 2026-11-08, before model evaluation results were available. The machine-readable protocol is `artifacts/study/protocol.json`; complete per-arm configurations are in `artifacts/study/E*/config.yaml`.

## Research scope

Reproduce the official Code-A1 coder–tester rewards, Mistake Book, GRPO, and dual-model updates, then compare the effect of execution feedback for the tester. This is a resource-limited research study, not a reproduction of the paper's full training budget. Performance gains are a hypothesis to test.

| Setting | Fixed value |
| --- | --- |
| Initialization | Local Qwen2.5-Coder-1.5B-Instruct; coder and tester initialized separately |
| Official code | ZJU-REAL/Code-A1, commit `be86d9315bab10a97d006aa25eb84204b25aa129` |
| Data | Official processed KodCode hard data; 9,684 of 9,688 rows satisfy the 1,024-token prompt limit; randomly select 128 with seed 42 |
| Training | 32 steps, batch size 4, one pass through the selected data; 2 coder outputs per problem and 2 tester outputs per code candidate |
| Optimization | Official FSDP2/GRPO; learning rate 1e-6, KL coefficient 0.001; one training seed, 42 |
| Generation budget | At most 1,024 output tokens per coder/tester response, identical across all four arms |
| Checkpointing | Final Hugging Face inference weights for both models only; no optimizer state or duplicate sharded checkpoint |
| Resources | 2 H100/H200 GPUs, 8 CPUs, 100 GiB memory, and a one-hour job cap per arm; record actual hardware/memory and monitor free Scratch space |

## Comparison arms and feedback

| Arm | Additional tester input during training |
| --- | --- |
| E0 | None; run the official entrypoint directly |
| E1 | Total and covered branch counts, plus selected covered/missing `(source line, destination line)` arcs |
| E2 | Selected locals and return values at function return, with function names and source line numbers |
| E3 | E1 and E2 combined |

For each coder output, use the first two existing training tests for that problem as fixed execution probes, without additional model generation calls. Execute candidate code in a separate sandbox. `coverage.py` collects actual branch arcs; an independent profiler collects locals when candidate functions return. Test frames, reference implementations, and expected assertion answers are excluded from feedback prompts. Training tests remain supervised oracles: execution paths and return values indirectly reveal behavior, which is the feedback signal under study.

All feedback arms use the same probe collection procedure and mask fields by arm. Additional input is capped at 384 tokens per code candidate. Collect at most 16 runtime snapshots, select at most 4 for the prompt, and shorten further as needed. Record the bounded trace, actual feedback text, input tokens, execution time, and collection failures. Generation-call budgets match; feedback input tokens and execution are additional costs, so total compute is not claimed to be equal.

Feedback is appended before the tester generates in the current round. `feedback_trainer.py` extends only tester input; official upstream files, rewards, and update logic remain intact.

## Evaluation

Evaluate the untrained `base` model and the final coder/tester weights from E0–E3. All variants receive identical evaluation inputs and no arm-specific execution feedback at test time, to compare capabilities after training.

- **HumanEval:** all 164 tasks, one greedy output per task; run the locally pinned original tests and record pass@1.
- **BigCodeBench:** all 1,140 v0.1.4 Instruct tasks, one greedy output per task, with the complete unittest suite. First calibrate every canonical solution in the same sandbox. Report the raw full-denominator pass rate, the pass rate on the subset whose canonical solutions pass, and the tasks affected by environment limitations. These resource-limited execution settings differ from the official leaderboard and do not support direct score comparisons.
- **ULT Lite:** the fixed 200 functions, K=2 rounds, at most 8 independent assertions per round. Round two sees the previous output. Each round allows at most 1,024 output tokens and uses greedy decoding for every variant. The interface follows Code-A1's assertion format and is an adaptation of ULT.
- **Tester metrics:** valid assertions/submissions, valid tests under a fixed query budget, mean line/branch coverage, and kill rate for at most 4 deterministic AST comparison/arithmetic/Boolean mutants per function. Only tests passing on the reference function contribute to coverage and mutant kills. This bounded mutation metric is not the official full Cosmic-Ray mutation score and may include equivalent mutants.

Save raw outputs, token counts, truncation status, per-task execution results, timeouts, and dependency errors for every model. The generation cap is 1,024 tokens and the context cap is 8,192 tokens. Oversized inputs are recorded as failures rather than silently truncated. Each sandbox request has a 120-second timeout and a 16 GiB per-process address-space limit; concurrency is 4, with no network or project-file access. CPU evaluation reuses third-party packages from `.venv-rl` rather than installing another environment.

Denominator conventions in the measurement implementation: mean coverage weights all 200 functions equally and assigns zero to measurement failures. Functions with no branches also receive zero branch coverage; line coverage is reported separately. The overall mutation score aggregates mutants from successfully measured functions. Paired bootstrap analysis instead weights per-function kill rates equally and assigns zero when no mutants exist or measurement fails; these two summaries need not have the same difference. Canonical calibration is a single run and cannot rule out all test randomness or flakiness.

## Analysis and limitations

Report paired task differences between each feedback arm and E0. Bootstrap tasks to quantify task-sampling uncertainty; this cannot replace multiple training seeds. Record training curves, gradients, sampling/feedback tokens, allocated GPU time, and execution time. Retain negative results, invalid outputs, and failure cases. Do not describe an improvement as verified without supporting execution results.

## Verified resource limits

`coe-gpu` has a 16-hour maximum walltime per job. The `coe-ice` setting `MaxTRESRunMinsPerUser=gres/gpu=1920` limits the sum of remaining reserved GPU time across running jobs to 32 GPU-hours; completed jobs release this reservation. It is not a lifetime cumulative GPU-hour limit. See the [official Slurm sacctmgr documentation](https://slurm.schedmd.com/sacctmgr.html). This study caps training jobs at one hour and generation jobs at two hours, and submits CPU evaluation separately. Actual usage comes from `sacct`. Training reservations were initially two hours and shortened to one hour while queued to improve backfill, without changing the 32-step budget.

CPU scoring runs four single-threaded sandboxes concurrently. The base evaluation requested 8 CPUs/48 GiB and took 16 minutes 31 seconds. The four trained variants were changed while queued to 4 CPUs/48 GiB and a one-hour cap to fit free cores; all ran on `ice-cpu`. Every candidate retains the same 120-second timeout. Interpret hardware and CPU-request differences using the actual resource records; job walltime alone does not establish algorithmic efficiency.
