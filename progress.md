# Adversarial Co-Training — Progress Tracker

> Last updated: 2026-11-09 (America/New_York)<br>
> Project: Adversarial Co-Training for Agentic Coding<br>
> Team: Daniel Choi, Jack Li, Muyu Pan, Tianyou Zhao<br>
> Plan source: [Adversarial Co-Training .pdf](<Adversarial Co-Training .pdf>), 3 pages.

## 1. Goals and Research Hypothesis

**Project goal:** reproduce Code-A1's adversarial coder–tester co-training and test whether richer execution feedback for the Test LLM improves test quality and Code LLM performance. (PDF pages 1–3.)

**Core hypothesis:** under a fixed generation budget, branch coverage and selected runtime values help the Test LLM produce more valid, bug-revealing unit tests, improving coder performance.

**Motivation:** code-generation RL relies on unit tests as feedback. Valid, diverse tests are needed for a reliable learning signal and to reduce reward hacking; the existing workflow makes limited use of runtime state.

**Training loop:**

```text
Problem specification + function signature
  → Code LLM generates code
  → Test LLM generates adversarial test cases
  → Verifier executes tests and computes rewards
  → RL updates both models; repeat

Extension: extract branch coverage / selected runtime values during execution
           and provide bounded feedback before the tester generates in that round.
           Implementation details: EXPERIMENT_PROTOCOL.md.
```

**Completion criteria:** a reproducible baseline, an execution-feedback training workflow, fixed-budget comparisons and ablations, and a final report. Performance improvement is a hypothesis, not a completion requirement; negative results and their interpretation must also be reported.

## 2. Experimental Scope and Configuration

| Item | Plan in the PDF | Verified status |
| --- | --- | --- |
| Base model | Qwen2.5-Coder-Instruct, 1.5B | Local model/tokenizer downloaded; SHA-256 and offline checks passed; actual vLLM generation passed on coe-gpu H200 |
| Training data | KodCode-V1 | Complete main training set of 484,097 rows downloaded; Code-A1's processed 9,688-row training set is ready |
| RL framework | verl | Official Code-A1 customization, 0.4.0.dev0; one coder/tester RL update completed on two H200 GPUs, job 6888089, exit 0 |
| Inference engine | vLLM | 0.8.5.post1 installed; actual local-model generation passed on H200 |
| Execution sandbox | Sandbox Fusion (page 1) | Official API + Bubblewrap Python runner connected; success/failure/timeout and filesystem/network isolation checks passed; full multilingual Docker service not deployed |
| Tester evaluation | UnLeakedTestBench | Five variants completed adapted ULT Lite 200/K=2; differences from full official Cosmic-Ray evaluation are in the report |
| Coder evaluation | HumanEval + BigCodeBench | All five variants attempted 164/1140 tasks; raw and canonical-qualified subset pass@1 recorded |
| Training signal | Branch coverage + selected runtime values | Implemented and used throughout E1–E3 training; traces, costs, and collection failures are auditable |
| Baseline reference | Code-A1 | Pinned official code reproduced at limited scale; budget deviations, methods, and results in CODE_A1_METHOD.md and FINAL_REPORT.md |

The final-stage schedule on PDF page 3 also mentions “HumanEval, MBPP, etc.”, while the main configuration and expected results list HumanEval, BigCodeBench, and UnLeakedTestBench. The frozen study uses these three; MBPP is outside this round's scope.

## 3. Current State

| Field | Current status |
| --- | --- |
| Overall state | Fixed-budget study complete: four 32-step training arms, final HF weights, five variants evaluated under the three-benchmark adaptation, ablation statistics, and report; see FINAL_REPORT.md for scale/environment limitations |
| Current week | Unknown: the PDF provides relative weeks but no actual start/end dates |
| Verified completion | Infrastructure, four 32-step dual-model runs, final weights, uniform five-variant evaluation, paired ablations, figures, and FINAL_REPORT.md |
| Repository evidence | models/, data/, external/, .venv-rl/, requirements/, scripts/; detailed checks in [ENVIRONMENT.md](ENVIRONMENT.md) and artifacts/setup/ |
| Not established | Other team members' work or training in other directories; this study does not cover the paper's full budget, multiple seeds, or full official Cosmic-Ray evaluation |
| Next milestone | No compute tasks remain within this protocol; larger training, multiple seeds, and a complete official evaluation environment are possible extensions |
| Current dependencies | All computation and W&B cloud readback completed; full loss histories, final numeric metrics, and artifact scope verified; team assignments/start date remain unspecified |

“No record” means no evidence is available in this repository, not that the team has done no work. Actual progress is not inferred from the planned week numbers.

### Status conventions

| State | Meaning |
| --- | --- |
| `UNKNOWN` | Planned, but no evidence of actual progress; default for newly listed research tasks |
| `TODO` | Confirmed not started |
| `IN_PROGRESS` | Active work; record current artifacts and next steps |
| `BLOCKED` | A specific blocker exists; record its cause and resolution conditions |
| `DONE` | Acceptance criteria met, supported by code, logs, results, or documentation |
| `DEFERRED` | Explicitly postponed or removed from scope; record the reason |

Unassigned owners are `TBD`. The implementation breakdown, acceptance criteria, and experimental design below supplement the plan; they are not results already reported in the PDF.

## 4. Weekly Plan

The original duration is seven weeks; actual start/end dates and the current calendar week remain unconfirmed. (PDF pages 1 and 3.)

| Week | Original goal | Tasks | Suggested acceptance criteria | Current state |
| --- | --- | --- | --- | --- |
| 1–2 | Prepare data and baselines: KodCode-V1 and benchmarks; integrate verl, vLLM, and Qwen 1.5B | T1, T2 | Data/evaluation entrypoints ready; model generation, sandbox execution, and a minimal RL update run successfully | `DONE`: minimum execution checks passed; this milestone alone excludes formal full evaluation/full-budget training |
| 3 | Replicate Code-A1: implement, test, and validate the baseline | T3 | Complete the co-training loop; save reproducible configurations, checkpoints, and baseline results | `DONE`: accepted within this resource-limited protocol; paper-scale training is outside scope |
| 4–5 | Implement improvements: branch coverage and runtime values | T4 | Collect execution feedback and pass it to the tester; run the enhanced workflow | `DONE`: accepted within this resource-limited protocol; paper-scale training is outside scope |
| 6–7 | Testing and report writing: benchmark evaluation, ablations, and report | T5, T6 | Complete main evaluations/comparisons/ablations and produce analysis and a final report | `DONE`: accepted within this resource-limited protocol; paper-scale training is outside scope |

## 5. Task / Sub-task Tracker

### Task overview

| ID | Task | Week | Depends on | Owner | Current state | Next step |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Maintain goals, status, and experiment logs | Throughout | — | TBD | `IN_PROGRESS` | Confirm calendar week, assignments, and artifact locations |
| T1 | Prepare training data and evaluation sets | 1–2 | — | TBD | `DONE` | Data and minimum entrypoints passed checks; full evaluation follows T5 |
| T2 | Set up model, RL, inference, and sandbox environments | 1–2 | — | TBD | `DONE` | H200 inference, Python sandbox, and a minimal dual-model update passed; check budget/storage before training |
| T3 | Reproduce the Code-A1 baseline | 3 | T1, T2 | TBD | `DONE` | Fixed-budget scope complete; results, deviations, and limitations in FINAL_REPORT.md |
| T4 | Add execution feedback | 4–5 | T3 | TBD | `DONE` | Fixed-budget scope complete; results, deviations, and limitations in FINAL_REPORT.md |
| T5 | Evaluate, compare, and ablate | 6–7 | T3, T4 | TBD | `DONE` | Fixed-budget scope complete; results, deviations, and limitations in FINAL_REPORT.md |
| T6 | Consolidate results and write the final report | 6–7 | T5 | TBD | `DONE` | Fixed-budget scope complete; results, deviations, and limitations in FINAL_REPORT.md |

### Sub-task details

Sub-tasks inherit their parent's planned weeks. Owners remain `TBD` until assignments are provided.

| ID | Sub-task / goal | Current state | Acceptance criteria / evidence | Next step |
| --- | --- | --- | --- | --- |
| T0.1 | Read the PDF and extract goals/seven-week plan | `DONE` | Sections 1–4 of this file, based on the project PDF | Record later scope decisions |
| T0.2 | Confirm actual progress and assignments | `UNKNOWN` | Calendar week, task owners, existing work, and artifact paths | Collect team updates and revise statuses |
| T1.1 | Prepare KodCode-V1 | `DONE` | data/KodCode-V1/; pinned revision, 15 main training shards, 484,097 rows; Parquet metadata/sample reads passed | Connect the official processed 9,688-row hard set to training |
| T1.2 | Prepare three main benchmarks | `DONE` | All three sets load; job 6888122 HumanEval/0 and BigCodeBench/4 generations passed; ULT K=1 output truncated/invalid; benchmark-smoke.json | Complete dependencies, coverage/mutation, and full evaluation under T5 |
| T1.3 | Check data/training interfaces | `DONE` | Four official examples passed RLHFDataset, collate_fn, and DataProto assembly; trainer data/configuration validation passed; trainer-data-check.json | Use the local processed set for expanded training |
| T2.1 | Verify compute resources/software | `DONE` | coe-gpu/account ece/QoS coe-ice submissions succeeded; H200 driver 615.71.09; actual torch/vLLM/FlashAttention generation passed | Allocate within budget and monitor quota |
| T2.2 | Load Qwen 1.5B with vLLM | `DONE` | 6888046: H200 generated add(a,b), exit 0; environment-gpu-check.json and job log | Reuse the local model for the baseline |
| T2.3 | Deploy Sandbox Fusion | `DONE` | Official API + Bubblewrap Python runner; 6888095 passed success/assertion failure/timeout/isolation; not full multilingual Docker; ENVIRONMENT.md | Start a local service in each job and bound concurrency |
| T2.4 | Connect the minimal verl training path | `DONE` | 6888089: two H200 GPUs, one per coder/tester, one step, exit 0; gradient norms 2.010/2.977; rl-smoke-check.json | Expand within the formal budget; retain smoke entrypoint |
| T3.1 | Review Code-A1 mechanisms/implementation | `DONE` | CODE_A1_METHOD.md reviews actual rewards, Mistake Book, GRPO, FSDP2, and configuration deviations | Follow the protocol without claiming the paper's absolute scores |
| T3.2 | Implement coder–tester–verifier loop | `DONE` | 6888089 completed official generation, sandbox rewards, GRPO advantages, and dual updates; 4 coder/8 tester outputs and rollout logs | Expand training and examine reward/stability |
| T3.3 | Train the baseline | `DONE` | 32 steps complete; final coder/tester HF weights, per-step gradients, and raw logs in artifacts/study/E0/; summary in FINAL_REPORT.md | Complete for this round; define a new protocol before scaling |
| T3.4 | Validate baseline reproduction | `DONE` | Limited-scale HumanEval/BigCodeBench/ULT Lite results complete; configuration deviations and canonical failures disclosed in FINAL_REPORT.md; no claim to absolute paper scores | Complete for this round; define a new protocol before scaling |
| T4.1 | Extract branch coverage | `DONE` | profile_probe.py + feedback-check.json: known partial coverage 1/2, full coverage 2/2, source arcs and timing recorded | Analyze actual training traces |
| T4.2 | Extract selected runtime values | `DONE` | Independent profiler captures bounded return-time locals/return; known-value check passed; no test frames/oracle answers in prompts | Analyze collection failures, lengths, and costs |
| T4.3 | Connect execution feedback to Test LLM | `DONE` | 256 real training-feedback records per enhanced arm, at most 384 tokens, linked to code/traces/timing; artifacts/study/E*/feedback/ | Complete for this round; define a new protocol before scaling |
| T4.4 | Train enhanced co-training variants | `DONE` | E1/E2/E3 completed 32 steps and saved both final HF models; complete training curves in analysis.json | Complete for this round; define a new protocol before scaling |
| T5.1 | Fix budgets and metrics | `DONE` | EXPERIMENT_PROTOCOL.md and artifacts/study/protocol.json fix data, seed, generation budget, extra feedback cost, and metrics | Follow the frozen protocol; do not select samples based on results |
| T5.2 | Evaluate Test LLM | `DONE` | All five variants completed adapted ULT Lite 200/K=2; coverage, bounded mutation, invalid tests, and measurement failures disclosed in FINAL_REPORT.md | Complete for this round; define a new protocol before scaling |
| T5.3 | Evaluate Code LLM | `DONE` | All five variants attempted 164 HumanEval + 1140 BigCodeBench tasks; raw/qualified pass@1 and environment failures reported | Complete for this round; define a new protocol before scaling |
| T5.4 | Run ablations and analyze results | `DONE` | Four-arm coverage/runtime/combined ablation complete; paired task bootstrap, costs, and positive/negative examples in analysis.json; single-seed limitation retained | Complete for this round; define a new protocol before scaling |
| T6.1 | Consolidate figures and conclusions | `DONE` | Three result/training figures in report/figures/ and auditable statistics in analysis.json; no-gain/negative findings retained | Complete for this round; define a new protocol before scaling |
| T6.2 | Finish the project report | `DONE` | FINAL_REPORT.md includes method, budget, actual metrics, ablations, failures, resources, limitations, and reproduction entrypoints | Complete for this round; define a new protocol before scaling |

## 6. Next Steps

Status after acceptance on 2026-11-09; complete operating instructions are in [ENVIRONMENT.md](ENVIRONMENT.md).

1. **This study is complete:** see [FINAL_REPORT.md](FINAL_REPORT.md) for results, evidence, and adaptation limits.
2. **Optional extensions:** increase training scale and seeds, verify in the complete official dependency environment, and expand mutation operators and feedback probes.
3. **Team information:** actual project dates and other members' assignments still require team input; do not infer them.

## 7. Experiment Plan & Results

### Frozen experiment matrix

The user authorized baseline training, evaluation, and subsequent work. This round uses four arms with a uniform, resource-limited budget of 32 steps, one seed, and 128 problems. Scope and deviations are in the [protocol](EXPERIMENT_PROTOCOL.md); this is not paper-scale training.

| Plan ID | Variant | Branch coverage | Runtime values | Current state | Result |
| --- | --- | --- | --- | --- | --- |
| E0 | Code-A1 baseline | No added feedback | No added feedback | `DONE` | 32 steps; HumanEval raw 72.56%; details/limitations in FINAL_REPORT.md |
| E1 | Coverage only | Yes | No added feedback | `DONE` | 32 steps; HumanEval raw 71.34%; details/limitations in FINAL_REPORT.md |
| E2 | Runtime values only | No added feedback | Yes | `DONE` | 32 steps; HumanEval raw 72.56%; details/limitations in FINAL_REPORT.md |
| E3 | Coverage + runtime values | Yes | Yes | `DONE` | 32 steps; HumanEval raw 71.95%; details/limitations in FINAL_REPORT.md |

Model initialization, data selection, training settings, evaluation protocol, and generation caps are fixed across arms. Added feedback input and execution costs are recorded separately. The untrained base model is the starting-point control. Job dependencies and IDs are in artifacts/study/jobs.json.

### Fixed and measured metrics

| Dimension | Metrics / records | Status |
| --- | --- | --- |
| Tester quality | Valid-test fraction, valid tests/query, bounded mutant kills, failure counts | Measured; FINAL_REPORT.md |
| Coder performance | HumanEval/BigCodeBench raw and canonical-qualified subset pass@1 | Measured; FINAL_REPORT.md |
| Feedback effect | Actual branch/line coverage, feedback lengths, and failure cases | Measured; analysis.json and report |
| Resources/stability | Slurm GPU-hours, actual input/output tokens, feedback time, timeouts/failures, training curves | Recorded; resources.json and analysis.json |

### Experiment record template

Append one entry per actual run; a plan ID alone is not evidence that a run happened. The E0–E3 fixed-budget comparison is complete; original runs, per-task results, and analysis are in artifacts/study/ and FINAL_REPORT.md.

```text
Run ID / Plan ID:
Date / Owner:
Task / Sub-task:
Objective:
State: TODO / IN_PROGRESS / BLOCKED / DONE
Code commit / Working-tree changes:
Model / Checkpoint:
Dataset / Split / Version:
Config path / Command / Seed:
Training budget / Generation budget:
Feedback format / Input cost:
Metrics / Results:
Artifacts: paths to logs, checkpoints, result files, or figures
Issues / Interpretation:
Next step:
```

## 8. Open Questions & Dependencies

GPU allocation and the Python sandbox work inside Slurm jobs. This round's budget and evaluation scope are fixed; remaining official-environment limitations and unresolved team metadata are listed below.

| ID | Question / resolved decision | Impact | When to address |
| --- | --- | --- | --- |
| Q1 | Start date, calendar week, and actual team progress | Align relative schedule with actual work | Next team update |
| Q2 | Member responsibilities and existing artifact locations | Avoid duplicate work and support status evidence | Next team update |
| Q3 | Fixed: 2 GPUs/arm, 32 steps, one-hour training cap; 16 hours is the partition's per-job maximum, with separate QoS limits | Details and official semantics in EXPERIMENT_PROTOCOL.md | Resolved for this round |
| Q8 | Python isolation passed; reuse full RL dependencies; BCB canonical 1087/1140 passed and 53 failures are listed separately | Report both raw/qualified results; full multilingual service remains undeployed | Revisit for official-environment expansion |
| Q4 | Official commit fixed; rewards, Mistake Book, and dual-model updates reviewed | CODE_A1_METHOD.md | Resolved for this round |
| Q5 | First two training-test probes, candidate return-time locals, and 384-token cap fixed | EXPERIMENT_PROTOCOL.md and actual feedback records | Resolved for this round |
| Q6 | Fixed: 32 steps, one seed, matching output/query budgets; additional input/execution costs separate | Task bootstrap does not estimate seed variance | Resolved for this round |
| Q7 | Frozen protocol includes HumanEval, BigCodeBench, and ULT Lite; excludes MBPP | A later addition requires a new protocol | Scope fixed for this round |

## 9. Progress Log

Append dated facts, evidence, and next steps. Do not present planned experiments as completed results.

| Date | Task / Sub-task | Event / Evidence | State / Outcome | Next step |
| --- | --- | --- | --- | --- |
| 2026-10-30 | T0.1 | Read all 3 pages of Adversarial Co-Training .pdf and extracted goals, settings, and the seven-week plan | `DONE`: research goals and original schedule documented | Confirm actual progress |
| 2026-10-30 | T0 | Inspected repository; created progress.md with tasks, acceptance criteria, suggested matrix, and log template | `IN_PROGRESS`: tracker established; research tasks remained `UNKNOWN` | Confirm week, owners, and artifacts |
| 2026-10-30 | T1, T2 | Created scripts/env.sh, project-local uv/.venv-tools, and dependency locks; downloaded model/official code; full data download interrupted | `IN_PROGRESS`: incomplete installation was not marked ready | Resume data and install dependencies |
| 2026-10-31 | T1.1, T1.2, T1.3 | Completed planned data and project-local data/models links; model SHA-256, 15 main KodCode shards, and benchmark reads passed | T1.1 `DONE`; counts/revisions/paths in artifacts/setup/assets.json and asset-check.json | Connect training/evaluation entrypoints |
| 2026-10-31 | T2.1, T2.2 | Installed training/evaluation dependencies, customized verl, and FlashAttention; 366-package check, 12 key imports, and offline config/tokenizer checks passed | `IN_PROGRESS`: Python environment verified; GPU inference/RL update not yet tested | Verify inference in a GPU allocation |
| 2026-10-31 | T2.2 | Immediate single-GPU A100/H100 requests 6703728/6703769 exited because resources were busy; no inference/training ran | `IN_PROGRESS`: not counted as successful GPU validation | Retry a short check when resources are available |
| 2026-10-31 | T2.3 | SDK imported, but endpoint was unset and no container runtime was found | `BLOCKED`: execution service not ready | Choose a permitted deployment or existing service |
| 2026-10-31 | T0, T2.1 | Audited Scratch inodes; removed 33 Triton compilation-cache locations at the user's request, preserving installation metadata | Files fell from 972,018 to 892,204 (89.2%); artifacts/setup/triton-cleanup.json | Continue GPU/environment checks when allocated |
| 2026-11-08 | T2.1, T2.2 | sbatch 6888046 used coe-gpu/account ece/QoS coe-ice and H200; local Qwen/vLLM generated add(a,b) | `DONE`: exit 0; environment-gpu-check.json, gpu-smoke-6888046.log | Verify generation in training |
| 2026-11-08 | T2.3 | Pinned SandboxFusion source and Python API/Bubblewrap runner; 6888080 failed a symlink mount; corrected physical software path and 6888095 passed four checks | `DONE`: success/failure/timeout/isolation passed; failed report retained; no full multilingual container | Start the endpoint automatically in each job |
| 2026-11-08 | T1.2 | 6888098 BigCodeBench/0 reference exceeded ten seconds; fixed smoke task /4 used in 6888122, whose HumanEval/0 and BCB/4 generations passed; ULT 10 tests truncated/invalid | `DONE`: minimum interface checks, exit 0, not formal scores; benchmark-smoke.json | Run formal evaluation under the complete protocol |
| 2026-11-08 | T1.2, T2.3 | 6888172 passed four sandbox checks and an independent pytest probe on a real ULT function | `DONE`: exit 0; probe checks pytest/FUT loading, not model test accuracy | Complete ULT coverage/mutation measurement |
| 2026-11-08 | T1.3, T2.4, T3.2 | 6888089 used 4 official rows, batch 2, n=2, two H200 GPUs; generated 4 coder/8 tester outputs and completed rewards, advantages, and both updates | `DONE`: one step, exit 0, 4:36, gradient norms 2.010/2.977; rl-smoke-check.json, trainer-data-check.json, rollouts/raw logs; no checkpoint | Prepare formal baseline training |
| 2026-11-08 | T0, T2.1 | Final pace-quota check: Scratch 210.7/300 GiB (70.2%), 895,352/1,000,000 files (89.5%); added lfs quota startup guard | About 89.3 GiB/104,648 files free; reserve 20 GiB/20,000 files; no other user directories deleted | Estimate checkpoint peak before training; pace-quota-20261108-final.txt |
| 2026-11-08 | T3.1, T5.1 | User authorized baseline/evaluation/subsequent work; froze four-arm, 32-step/128-problem/single-seed resource-limited study, distinct from paper-scale training | `DONE`: protocol/method docs, protocol.json, and per-arm configurations recorded | Execute the frozen budget |
| 2026-11-08 | T2.1 | Verified coe-gpu 16-hour per-job maximum; coe-ice MaxTRESRunMinsPerUser=1920 GPU-minutes is remaining running-job reservation, not lifetime usage | Reduced training reservation from two hours to one for backfill; 32-step budget retained | Record actual GPU-hours |
| 2026-11-08 | T3.3, T4.4 | Submitted E0 6888252, E1 6888295, E2 6888296, E3 6888297; all waiting for two H200 GPUs at this checkpoint | `IN_PROGRESS`: queued only, not completed training; final HF-only saving planned | Wait for resources and inspect actual outputs |
| 2026-11-08 | T4.1, T4.2, T5.2 | CPU jobs 6888280/6888341 checked actual branch arcs, return-time locals, masked fields/lengths, rejection of a known bad assertion, two known mutant kills, and full branch coverage | `DONE`: feedback-check.json and study-evaluation-check.json; upstream unchanged | Verify enhanced training integration |
| 2026-11-08 | T5, T6 | Base generation 6888290 and canonical calibration 6888310 running; later generation, CPU scoring, and report dependencies submitted | `IN_PROGRESS`: all 164 HumanEval canonicals passed; some BCB network/NLTK/version failures recorded; no completed four-arm comparison yet | Score/generate report after dependencies succeed; jobs.json |
| 2026-11-08 | T3, T4 | Broadened GPU type to coe-gpu H100/H200 for scheduling; count/budget retained; E0–E3 started | `IN_PROGRESS`: E0 H100, E1 H200; first updates passed without restarting training for monitoring; actual hardware in sacct | Complete 32 steps and final checkpoints |
| 2026-11-08 | T5 | Base generated all 1304 coder tasks and ULT Lite 200/K=2 outputs; canonical calibration finished | HumanEval 164/164, BCB 1087/1140 canonical passed; 53 failures retained; base CPU scoring running | Use the same calibration for all variants |
| 2026-11-08 | T0, T6 | User supplied .env credentials, authorized a W&B project, and limited uploads to numeric metrics/config/resources/evaluation summaries; mushroom_yeren/6257-adversarial-co-training created and verified PRIVATE | E0–E3 cloud loss/reward readback passed; online monitor 6888688 backfilled/streamed; final numeric upload 6888769 depends on report/monitor; no prompts, outputs, per-example logs, data, or weights | Verify final cloud eval scores; keep credentials unprinted and Git-ignored |
| 2026-11-09 | T3.3, T4.3, T4.4 | All four 32-step training jobs COMPLETED/0:0; each role has 32 finite-gradient records; eight models/16 HF shards total 52.96 GiB; 256 feedback records per enhanced arm, maximum 143/365/382 tokens | Training accepted; Scratch 263.8/300 GiB, about 36.2 GiB free; four 1–32-step dual-loss histories verified in cloud, base scores match local values | Finish uniform evaluation and final cloud verification |
| 2026-11-09 | T5 | Verified 1304 coder outputs/completion markers per arm; released redundant dependencies so CPU scoring could overlap tester generation; base scoring took 16:31 and later CPU caps shortened to one hour for backfill | Scheduling change only; frozen models/tasks/metrics/sampling budget retained; ULT scoring waits for tester completion marker | Check CPU jobs and scoring outputs |
| 2026-11-09 | T3–T6 | Four 32-step runs, final HF weights, five-variant 1304-task coder evaluation and ULT Lite 200/K=2 adaptation completed; analysis/report passed evidence checks | `DONE` within frozen scope; FINAL_REPORT.md, artifacts/study/analysis.json, report/figures/; single-seed/environment/bounded-mutation limits retained | Define a new budget/protocol for extensions |
| 2026-11-09 | T0, T6 | Report 6888608, monitor 6888688, and cloud verification 6888769 COMPLETED/0:0; all four 32-step dual-loss histories and five final numeric summaries match cloud readback; artifacts only config.yaml/metrics.json/protocol.json | `DONE`: wandb-final-verified.json; 3.866 GPU-hours; Scratch 263.9/300 GiB, 36.1 GiB free; main paired intervals include zero, with no stable improvement established | No remaining Slurm tasks within this round |
| 2026-11-09 | T0, T6 | Translated repository Markdown and automatic report/progress output into English for the team; retained original Chinese documents locally under Git-ignored local-docs/zh-CN/ | Documentation language updated; experimental values and evidence preserved | Maintain repository documentation in English |
| 2026-11-09 | T6 | Fixed empty W&B charts caused by summary-only final metrics. Final publishing now logs aligned train_step/history rows, checks live chart queries, preserves original losses, and validates numeric internal history archives; CPU repair job 6906571 | `DONE`: all five runs have chart-readable final metrics, each trained variant retains 32 loss steps per model, and no public numeric summary-only metrics remain; wandb-final-verified.json | Refresh the workspace; use E0–E3 for training curves and final observations for score comparisons |

| 2026-11-09 | T0, T6 | Copied metrics to private team project 6257-final-proj/6257-adversarial-co-training; original personal project retained. Upload job 6911386 completed the copy but failed its initial check on summary propagation; verification now waits for both chart history and summary. Final CPU job 6911407 passed | `DONE`: wandb-team-final-verified.json verifies five runs, 32 dual-loss steps per trained arm, matching final scores, plot-ready numeric metrics, and approved artifact scope; no GPU training repeated | Use the team project link in README; inspect partial destinations before attempting another backfill |

### Future update rules

- After completing work, update the sub-task's state, evidence, and next step, and its parent's status.
- Mark `DONE` only with checkable artifacts; mark `BLOCKED` with a cause and resolution conditions.
- Append a record for each training/evaluation run. Preserve failed and interrupted attempts rather than replacing their history.
- Record dated changes to scope, goals, budget, or schedule, including the reason and impact, and update this file's date.
