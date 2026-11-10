# Experiment Environment and Local Resources

Check date: 2026-11-08. **The weeks 1–2 minimum execution checks passed:** H200 inference on `coe-gpu`, the actual verl data interface, an isolated Python sandbox, minimal generation/judgment paths for three benchmarks, and one step of dual-model RL on two H200 GPUs. At that checkpoint, formal baseline training, saved checkpoints, and full benchmark scores had not yet been produced. See the November 9 section and [final report](FINAL_REPORT.md) for the completed study.

## Activate the environment

Run from the project root:

```bash
source scripts/activate.sh rl
```

This entrypoint sets project-local caches before activating `.venv-rl`. Training, evaluation, downloads, and Slurm scripts should use it first. The user's global shell configuration was not changed; Hugging Face may still use default cache locations in environments that do not load this configuration.

| Component | Project-local path |
| --- | --- |
| Training/evaluation virtual environment | `.venv-rl/` |
| Download/checking tools environment | `.venv-tools/`; activate with `source scripts/activate.sh tools` |
| uv | `.tools/bin/uv`, version 0.13.6 |
| Hugging Face cache root | `.cache/huggingface/` |
| Hub / datasets / Xet caches | `.cache/huggingface/hub/`, `datasets/`, `xet/` |
| uv / pip / temporary files | `.cache/uv/`, `.cache/pip/`, `.cache/tmp/` |
| Torch / CUDA / Triton / vLLM caches | Corresponding subdirectories of `.cache/`; see `scripts/env.sh` |
| W&B output | `artifacts/wandb/`, offline by default |
| Check reports and installation logs | `artifacts/setup/` |

All paths are derived from the project's physical path. `models/` and the Hugging Face dataset directories use relative symlinks to project-local caches to avoid duplicate large files. UnLeakedTestBench points into this project's `external/` directory. Preserve these directories together when moving the project; do not remove the cache independently.

## Software versions and sources

The project uses the customized verl from the [official Code-A1 repository](https://github.com/ZJU-REAL/Code-A1), retaining its training PyTorch/vLLM combination. Training and evaluation share the environment to avoid downloading another CUDA/PyTorch stack. Instead of adopting the separate vLLM 0.11.0 used by the official evaluation scripts, this project uses training-environment vLLM 0.8.5.post1. Minimal generation/execution paths have been verified; compatibility with the complete official evaluation workflow is not established by those checks.

| Component | Installed version / location |
| --- | --- |
| Python | 3.10.10, PACE `python/3.10.10` module |
| PyTorch | 2.6.0+cu124, CUDA runtime 12.4 |
| vLLM | 0.8.5.post1 |
| Transformers | 4.51.3 |
| verl | 0.4.0.dev0, editable installation from `external/Code-A1/Code-A1/verl/` |
| FlashAttention | 2.7.4.post1, official cu12/torch2.6/cp310/ABI FALSE wheel |
| Ray | 2.47.1 |
| Sandbox Fusion SDK | 0.3.7; the SDK does not include a deployed execution service |
| Datasets / Hugging Face Hub | 4.4.1 / 0.36.0 |
| BigCodeBench / EvalPlus / Cosmic Ray | 0.2.5 / 0.3.1 / 8.7.0 |

The combined dependency lock is `requirements/experiment.lock`; original requirements are in `requirements/rl.in` and `requirements/eval.in`. Installation also includes editable verl and a local FlashAttention wheel; see `scripts/setup_environment.sh` and `artifacts/setup/installed-packages.txt`.

Pinned Code-A1 commit: `be86d9315bab10a97d006aa25eb84204b25aa129`. Pinned UnLeakedTestBench commit: `d992b2d95ff0bececa90bfccd1ea683a3a2bea18`. Official checkouts are in `external/`; upstream source files were not modified.

For a fresh team checkout, the PACE workflow expects these pinned source repositories under the Git-ignored `external/` directory, independently of the retained `Code-A1/` tree:

```bash
mkdir -p external
git clone https://github.com/ZJU-REAL/Code-A1.git external/Code-A1
git -C external/Code-A1 checkout be86d9315bab10a97d006aa25eb84204b25aa129
git clone https://github.com/huangd1999/UnLeakedTestBench.git external/UnLeakedTestBench
git -C external/UnLeakedTestBench checkout d992b2d95ff0bececa90bfccd1ea683a3a2bea18
```

To reinstall using the existing local code and uv:

```bash
bash scripts/setup_environment.sh
```

## Model and data

| Resource | Path | Verified contents |
| --- | --- | --- |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | `models/Qwen2.5-Coder-1.5B-Instruct/` | 10 files, about 3.10 GB; weight SHA-256 and offline config/tokenizer checks passed |
| Complete KodCode-V1 | `data/KodCode-V1/` | 15 main training shards, 484,097 rows; upstream `use_with_caution` shards retained separately |
| Code-A1 training data | `data/Code-A1/train_data/` | Official processed hard subset, 9,688 rows |
| Code-A1 validation data | `data/Code-A1/val_data/` | 542 rows; includes upstream mixed validation content, without adding MBPP to this study's evaluation scope |
| HumanEval | `data/HumanEval/` | 164 tasks |
| BigCodeBench | `data/BigCodeBench/` | All repository files downloaded; v0.1.4 has 1,140 tasks |
| UnLeakedTestBench | `data/UnLeakedTestBench/` | ULT: 3,909 rows; ULT_Lite: 200 rows; upstream PLT files also retained |

UnLeakedTestBench's `.jsonl` files are actually JSON arrays and must be read with `json.load()`; line counts are not sample counts. The public repository's `tests` fields were not executed during the asset check or treated as authoritative ground-truth tests.

Pinned revisions, file counts, and cache locations are recorded in `artifacts/setup/assets.json`. Only the planned 1.5B model was downloaded, not 3B/7B weights. Downloads reuse cached content and can resume:

```bash
source scripts/activate.sh tools
python scripts/download_assets.py
```

## Completed validation

```bash
source scripts/activate.sh tools
python scripts/check_assets.py

source scripts/activate.sh rl
uv pip check --python .venv-rl/bin/python
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_environment.py
```

- Weight SHA-256: `c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8`, matching the Hugging Face cache blob identifier.
- Model/data files, project-local paths, Parquet metadata, sample reads, and ULT JSON parsing passed checks.
- Dependency consistency passed for 366 installed packages.
- Actual imports passed for Torch, Transformers, Datasets, Ray, vLLM, verl, FlashAttention, the Sandbox Fusion SDK, BigCodeBench, EvalPlus, Cosmic Ray, and coverage.
- Code-A1's `trainer_a1`, including FSDP worker dependencies, and its Sandbox Fusion reward module imported successfully; see `artifacts/setup/baseline-import-check.log`.
- Local model config/tokenizer loading and chat-template application succeeded in offline mode.

Detailed results are in `artifacts/setup/asset-check.json`, `environment-check.json`, and the adjacent `.log` files. The `passed` field in `environment-check.json` covers only its import and offline-loading checks; it does not establish a successful full training run.

## 2026-10-31 execution checklist (historical snapshot)

1. **GPU inference:** the login node had no GPU. Immediate A100/H100 allocation attempts exited because resources were busy; inference had not run. Once a compatible GPU was available, the proposed check was:

   ```bash
   srun -p ice-gpu --gres=gpu:a100:1 -c 4 --mem=24G -t 00:08:00 bash scripts/gpu_smoke.sh
   ```

   This script only loads the local model, generates a small amount of text, and saves `artifacts/setup/environment-gpu-check.json`. It neither executes generated code nor trains. This older CUDA/PyTorch combination had not been verified on Blackwell/AMD GPUs.

2. **Sandbox Fusion service:** the SDK was installed, but `SANDBOX_FUSION_ENDPOINT` was unset and no Docker, Apptainer, or Singularity runtime was found. A PACE-compatible isolated service or existing endpoint was needed to validate success, failure, and timeout handling. Generated code was not run directly on the login node.
3. **Minimal RL and end-to-end evaluation:** official configuration still needed local model/data/output paths and verification after GPU/sandbox setup. The official default uses 4 GPUs per coder/tester; a single-GPU smoke check did not establish training capacity.
4. **UnLeakedTestBench entrypoint:** upstream recommended Python 3.12; this project used Python 3.10 to match Code-A1. Data had been checked, but generation/evaluation compatibility and missing helper modules still needed integration. Import success was not counted as end-to-end evaluation success.

## Quota and version control

Caches and runtime resources are kept in this Scratch project. The post-installation snapshot showed about 230.1/300 GB and 971,920/1,000,000 files (97.2%) across the user's entire Scratch space; Home was not the download target. File count was closer to its limit than storage capacity, so subsequent `.pyc` writes were disabled, and uv hardlinks and Hugging Face caches were reused. `uv cache prune` found no stale entries to remove. Further large installations require attention to file-count quota.

On 2026-10-31, 33 Triton compilation-cache locations were removed at the user's request. File usage fell from 972,018 to 892,204 (89.2%), releasing about 79,814 inodes and leaving about 107,796. Capacity usage was about 228/300 GB. Triton packages and their 10 installation-metadata directories were preserved. Earlier directory audits are historical snapshots; the cleanup record is `artifacts/setup/triton-cleanup.json`.

`.gitignore` excludes `models/`, `data/`, `external/`, virtual environments, caches, and runtime logs. Scripts, lock files, and documentation can be committed. At the original environment-setup checkpoint, these changes had not yet been committed or pushed.

Cache variables follow the [official Hugging Face guidance](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) and are set before library imports.

## 2026-11-08: weeks 1–2 execution checks

The validated submission commands are below; run them from the project root. `coe-gpu` uses `account=ece` and `qos=coe-ice`. Generic `--gres=gpu:N` requests can receive H100 or H200 GPUs.

The acceptance summary is [week12-summary.json](artifacts/setup/week12-summary.json); final exit statuses are in the [Slurm accounting record](artifacts/setup/week12-slurm-accounting.txt). Failed and interrupted attempts remain in the original logs.

| Entrypoint | Resources | Scope | Evidence |
| --- | --- | --- | --- |
| `sbatch scripts/gpu_smoke.sbatch` | coe-gpu, 1 GPU, 4 CPUs, 32 GiB, 15 minutes | Actual local Qwen 1.5B/vLLM generation | 6888046: H200, exit 0; `environment-gpu-check.json` |
| `sbatch scripts/sandbox_smoke.sbatch` | ice-cpu, 2 CPUs, 8 GiB, 10 minutes | Success, assertion failure, one-second timeout, no project mount/outbound network; ULT pytest probe | 6888172: exit 0, 5 checks passed; `sandbox-check.json` |
| `sbatch scripts/benchmark_smoke.sbatch` | coe-gpu, 1 GPU, 4 CPUs, 32 GiB, 20 minutes | One output each for HumanEval/0, BigCodeBench/4, and the first ULT Lite row; valid code sandboxed and invalid output marked failed | 6888122: exit 0; first two code candidates passed, ULT tests were truncated/invalid; `benchmark-smoke.json` |
| `sbatch scripts/week12_smoke.sbatch` | coe-gpu, 2 GPUs, 8 CPUs, 80 GiB, 45 minutes | Small official-data slice, 1 GPU per coder/tester, one collaborative update | 6888089: exit 0; `trainer-data-check.json`, `week12-smoke-6888089.log`, `rl-smoke-check.json` |

The inference check generated a correct `add(a, b)` function. The host used driver 615.71.09, PyTorch CUDA runtime 12.4, and NVIDIA H200 GPUs. `USE_TF=0` and `USE_FLAX=0` were added to avoid importing unnecessary frameworks in the PyTorch workflow.

### Python sandbox deployment

Bubblewrap (`bwrap`) is available; no Apptainer/Singularity module was found. The project uses [official SandboxFusion source](https://github.com/bytedance/SandboxFusion), pinned to commit `469de0845cb7d665bc68044d541aa3aa28c5b30a`. `scripts/sandbox_service.py` reuses its `/run_code` route, request/response types, and status handling, registering a local Bubblewrap Python runner without modifying upstream source.

This is a Python-only adaptation, not the complete official multilingual Docker service. It provides no file transfer, Jupyter, GPU code execution, or database services and does not use the privileged upstream `lite` isolation mode. Each request creates separate PID, network, and mount namespaces, clears environment variables, and mounts system software and a dedicated runtime read-only. Project and Home directories are not mounted. The working directory and `/tmp` use namespace-local temporary filesystems. Output is bounded; the default execution cap is 30 seconds, per-process address space is limited to 2 GiB, and concurrency is at most 4. The service listens only on the compute node's `127.0.0.1` and exits with the Slurm job.

Rebuild this environment with:

```bash
bash scripts/setup_sandbox.sh
```

`.venv-sandbox` uses PACE Python 3.11.9 and installs only API dependencies, NumPy, and pytest; see `requirements/sandbox.lock`. It adds no CUDA/PyTorch installation. At the weeks 1–2 checkpoint, the remaining BigCodeBench third-party dependencies and full ULT coverage/mutation evaluation were still pending; the fixed standard-library examples alone had been verified.

### Data and minimal training configuration

`scripts/prepare_week12.py` merges verl defaults, the official 1.5B_A1 configuration, and local paths into `artifacts/week12/baseline-local.yaml`. This is a starting point for baseline training, not evidence of a completed full-budget run.

The same script creates `smoke.yaml` and four short official training examples, validating `RLHFDataset → collate_fn → DataProto`. It only reads data and does not execute generated code on the login node. The training smoke uses batch size 2, 2 samples per coder/tester, at most 512 generated tokens, and one training step; full pre-training validation and checkpoint saving are disabled. Its scores validate the environment, not the formal baseline.

Job 6888089 generated 4 coder outputs and 8 tester outputs, then completed sandbox rewards, GRPO advantages, old/reference-policy log probabilities, and both optimizer updates in 4 minutes 36 seconds. Gradient norms were **2.010 / 2.977**, finite and nonzero. Peak allocated GPU memory was about 62.7 GiB per model. Raw rollouts, the Mistake Book, and parsed acceptance reports were saved. The console learning rate `0.000` is rounding to three decimals; the configured rate was `1e-6`.

After a job, check the dual-model completion marker and finite gradients:

```bash
source scripts/activate.sh tools
python scripts/summarize_rl_smoke.py artifacts/setup/week12-smoke-<jobid>.log
```

### Storage checks

On November 8, `pace-quota` showed about **210.6/300 GiB (70.2%)** and **892,609/1,000,000 files (89.3%)**. This is shared across the user's entire Scratch space, so other directories affect available quota. The initial `du` audit showed about 16 GiB in this project's `.cache`. Avoid double-counting model/data symlinks and uv hardlinks when interpreting directory sizes.

`scripts/check_quota.py` reads `lfs quota` directly. Validation jobs require at least **20 GiB free and 20,000 remaining files** before starting and exit otherwise. After all checks, `pace-quota` showed **210.7/300 GiB (70.2%)**, **895,352/1,000,000 files (89.5%)**, and about **89.3 GiB / 104,648 files** remaining. The latest snapshot is `artifacts/setup/quota-check.json`; the human-readable historical record is `pace-quota-20261108-final.txt`. No other user directories were cleaned, no additional models were downloaded, and smoke checkpoints were not saved. Ray temporary sockets/logs use compute-node `/tmp/a1-ray-<user>-<jobid>` to reduce persistent storage and path-length problems.

### Retained failure records

- 6888080: Bubblewrap failed to mount the symlink target at `/usr/local/pace-apps`. Mounting the actual `/storage/pace-apps` path fixed it; job 6888095 passed four sandbox checks. The failed report is `sandbox-check-6888080-failed.json`.
- 6888098: BigCodeBench/0 reference tests enumerate permutations and exceeded the ten-second smoke budget. The fixed BigCodeBench/4 example was used to validate the interface instead; this substitution was not used in formal scores. Original logs remain available.
- 6888122: ULT 10 test generation hit the 512-token cap and was marked invalid, with the full output retained rather than replaced by a successful example. Job 6888172 used an independent pytest probe of that FUT to check loading/execution; the probe does not count as model test accuracy. Full ULT metrics were still pending at this historical checkpoint.

## 2026-11-09: fixed-budget study completed

E0–E3 each completed 32 steps of dual-model training and saved final Hugging Face weights. All five variants completed evaluation on 164 HumanEval tasks, 1,140 BigCodeBench tasks, and ULT Lite 200/K=2 under the adapted protocol. See [FINAL_REPORT.md](FINAL_REPORT.md) for results, figures, and limitations; job mappings are in `artifacts/study/jobs.json` and complete statistics in `artifacts/study/analysis.json`.

CPU evaluation reused the full `.venv-rl` dependencies inside Bubblewrap, with a 120-second timeout, 16 GiB address-space limit, and concurrency 4. The base evaluation requested 8 CPUs; the four trained variants used 4 CPUs, all with 48 GiB scheduling memory. BigCodeBench reference implementations passed 1087/1140 tasks; 53 reference failures are reported separately. ULT uses the bounded assertion/AST-mutation adaptation.

Actual allocation was 3.866 GPU-hours. The eight final HF models occupy 52.96 GiB. Final Scratch usage was 263.9/300 GiB, leaving 36.1 GiB, with 896,514/1,000,000 files. This is the quota for the user's entire Scratch space; 210.7 GiB was the pre-training snapshot.

## W&B training monitoring

Cloud readback passed; see `artifacts/study/wandb-final-verified.json`. All four arms have 32 steps of dual-model loss, final evaluation values match for all five variants, and artifacts contain only configuration and numeric metric records. The monitor and final upload jobs exited successfully.

The user authorized only numeric training metrics, configurations, resource monitoring, and aggregate evaluation summaries. The team project is [6257-final-proj/6257-adversarial-co-training](https://wandb.ai/6257-final-proj/6257-adversarial-co-training/workspace). The [original personal project](https://wandb.ai/mushroom_yeren/6257-adversarial-co-training/workspace) is retained. Each training arm has a separate run; base is retained as the evaluation control.

`scripts/wandb_monitor.py` reads persistent records every 30 seconds in a separate CPU job, backfills earlier steps, and streams metrics to W&B without restarting GPU training for authentication or network delays. Metrics include `actor_a/pg_loss`, `actor_b2/pg_loss`, KL, entropy, rewards, gradient norms, token lengths, execution time, and memory. It adds pass@1, coverage, and bounded mutation summaries as evaluations finish. This study's original console-only metrics have three-decimal precision; earlier lost precision cannot be recovered. Future structured streams through `metrics_tracking.py` can preserve full precision. A learning rate rounded to zero on the console does not mean the actual rate was zero; the initial configuration is 1e-6.

Credentials are read from the user-provided `.env`, never printed or written to run configurations/artifacts. `.env` is Git-ignored. The cloud target and approved scope are in `artifacts/study/wandb-config.json`, currently `scope=metrics`. Prompts, model outputs, per-example logs, datasets, and weights are excluded. Local W&B files also stay in ignored `artifacts/wandb/`.

After the user explicitly specifies/authorizes the destination, configure and verify the project with:

```bash
source scripts/activate.sh rl
python scripts/configure_wandb.py
sbatch scripts/wandb_monitor.sbatch online
```

For local W&B-format records only, use `sbatch scripts/wandb_monitor.sbatch offline`; it does not upload to the cloud. Do not run multiple online monitors against the same runs. The initial cloud check is `artifacts/study/wandb-verified.json`; ongoing state is `artifacts/study/wandb-status.json`.

### Copying completed metrics to another team

The requested Forge URL `https://forge.coreweave.com/wandb/6257-final-proj` identifies the `6257-final-proj` team. It initially contained no projects, so a private project with the original study's name was created there. Its configuration, upload state, and verification evidence are separate from those of the personal project: `wandb-team-config.json`, `wandb-team-status.json`, and `wandb-team-final-verified.json` under `artifacts/study/`.

All five team runs passed cloud verification on November 9, 2026. E0–E3 each retain 32 loss steps for both roles, and final evaluation fields match the local analysis and are readable through live chart queries. Copy job 6911386 uploaded the data but its initial check encountered a lagging summary; final publication and verification job 6911407 completed successfully. Both jobs used CPU resources only.

For an additional authorized, empty destination, use separate output paths:

```bash
source scripts/activate.sh rl
python scripts/configure_wandb.py --entity TEAM --project PROJECT \
  --config-file artifacts/study/wandb-copy-config.json
sbatch scripts/copy_wandb_study.sbatch artifacts/study/wandb-copy-config.json \
  artifacts/study/wandb-copy-status.json artifacts/study/wandb-copy-verified.json
```

The CPU-only copy job refuses a destination containing runs to prevent replaying existing histories. It backfills the recorded training curves, publishes final metrics with `train_step`, and verifies live chart queries and artifact scope. A partial copy needs inspection before resuming; do not rerun the monitor blindly. The final publisher can safely be rerun with `--target-config` and `--verification-file` for the selected destination. No GPU training is repeated.

### Reading charts and diagnosing missing data

Training curves are `actor_a/pg_loss` (coder) and `actor_b2/pg_loss` (tester), plotted against `train_step`. Select E0–E3 to see their 32 recorded steps. The untrained `base` run has evaluation results but no training-loss curve.

`training/completed_steps` is a progress counter. New monitoring logs it with each training step. For the original study, its repaired historical record is a final observation at step 32; `base` records zero completed training steps. Final `eval/*`, `loss/*_policy_final`, compute, storage, and feedback summaries are also final observations, not measurements taken at every training step. Use the Runs table or bar charts to compare final scores across variants.

A numeric value in `run.summary` alone does not populate a history line chart. A custom-axis chart requires the metric and its `train_step` in the same history row. The final publisher therefore logs both through `run.log()` in addition to updating summary. `scripts/publish_wandb_summary.py` resumes the existing runs and appends a missing final row only once; it does not replay the original loss histories. Re-run it with `sbatch scripts/publish_wandb_summary.sbatch` when final metrics need publishing. It uses CPU resources only and verifies aligned history points, matching summary values, preserved original losses, and the approved artifact scope. Results are recorded in `artifacts/study/wandb-final-verified.json`.

Verification queries the live sampled-history interface used by charts because full archived snapshots can lag resumed runs. Original loss comparisons bypass the SDK history cache and ignore null fields in sparse evaluation rows. W&B-generated `wandb-history` Parquet artifacts are accepted only for the matching run and after verifying numeric column types; these are metric archives, not uploaded benchmark data or model weights.
