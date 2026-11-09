# 实验环境与本地资源

检查日期：2026-11-08。**第 1–2 周最小运行验收通过**：coe-gpu H200 推理、真实 verl 数据接口、隔离 Python 沙箱、三个 benchmark 的最小生成/判定入口，以及两张 H200 上的 1 step 双模型 RL 更新均已验证。正式 baseline 训练、checkpoint 与全量 benchmark 分数尚未产生。

## 激活环境

在项目根目录运行：

```bash
source scripts/activate.sh rl
```

该入口先设置项目内缓存，再激活 `.venv-rl`。后续训练、评测、下载和 Slurm 脚本也应先使用该入口。没有修改用户全局 shell 配置；在未加载此配置的其他环境中，Hugging Face 仍可能使用其默认缓存位置。

| 内容 | 项目内路径 |
| --- | --- |
| 训练与评测虚拟环境 | `.venv-rl/` |
| 下载与检查工具环境 | `.venv-tools/`，通过 `source scripts/activate.sh tools` 激活 |
| uv | `.tools/bin/uv`，0.13.6 |
| Hugging Face 根缓存 | `.cache/huggingface/` |
| Hub / datasets / Xet 缓存 | `.cache/huggingface/hub/`、`datasets/`、`xet/` |
| uv / pip / 临时文件 | `.cache/uv/`、`.cache/pip/`、`.cache/tmp/` |
| Torch / CUDA / Triton / vLLM 缓存 | `.cache/` 下对应子目录，见 `scripts/env.sh` |
| W&B 输出 | `artifacts/wandb/`，默认 offline |
| 检查报告和安装日志 | `artifacts/setup/` |

所有上述路径均从项目的物理路径计算。`models/` 和 Hugging Face 数据目录使用相对软链接指向本项目缓存，避免额外保存一份大文件；UnLeakedTestBench 指向本项目 `external/`。移动整个项目时应一起保留这些目录，不能单独删除缓存。

## 软件版本与来源

使用 [Code-A1 官方代码](https://github.com/ZJU-REAL/Code-A1) 的定制 verl，保留其训练用 PyTorch/vLLM 组合。训练与评测共用该环境，避免下载第二套 CUDA/PyTorch。没有照搬官方评测脚本中的独立 vLLM 0.11.0；本项目使用训练环境的 vLLM 0.8.5.post1，最小生成/执行入口已验证，完整官方评测流程的兼容性仍需确认。

| 组件 | 已安装版本 / 位置 |
| --- | --- |
| Python | 3.10.10，PACE `python/3.10.10` module |
| PyTorch | 2.6.0+cu124，CUDA runtime 12.4 |
| vLLM | 0.8.5.post1 |
| Transformers | 4.51.3 |
| verl | 0.4.0.dev0，editable 安装自 `external/Code-A1/Code-A1/verl/` |
| FlashAttention | 2.7.4.post1，官方 cu12/torch2.6/cp310/ABI FALSE wheel |
| Ray | 2.47.1 |
| Sandbox Fusion SDK | 0.3.7；SDK 不包含已部署的执行服务 |
| Datasets / Hugging Face Hub | 4.4.1 / 0.36.0 |
| BigCodeBench / EvalPlus / Cosmic Ray | 0.2.5 / 0.3.1 / 8.7.0 |

完整联合依赖在 `requirements/experiment.lock`；原始需求分别在 `requirements/rl.in` 和 `requirements/eval.in`。实际安装还包括 editable verl 和本地 FlashAttention wheel，见 `scripts/setup_environment.sh` 及 `artifacts/setup/installed-packages.txt`。

Code-A1 固定 commit：`be86d9315bab10a97d006aa25eb84204b25aa129`。UnLeakedTestBench 固定 commit：`d992b2d95ff0bececa90bfccd1ea683a3a2bea18`。官方代码下载在 `external/`，没有修改上游源码。

在现有本地代码和 uv 的基础上重新安装：

```bash
bash scripts/setup_environment.sh
```

## 模型和数据

| 资源 | 路径 | 已验证内容 |
| --- | --- | --- |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | `models/Qwen2.5-Coder-1.5B-Instruct/` | 10 个文件，约 3.10 GB；权重 SHA-256、离线 config/tokenizer 通过 |
| 完整 KodCode-V1 | `data/KodCode-V1/` | 15 个主 train 分片，共 484,097 条；另保留上游 `use_with_caution` 分片，与主 train 分开 |
| Code-A1 训练数据 | `data/Code-A1/train_data/` | 官方处理好的 hard 子集，9,688 条 |
| Code-A1 验证数据 | `data/Code-A1/val_data/` | 542 条；包含上游混合验证内容，未另行决定新增 MBPP 评测范围 |
| HumanEval | `data/HumanEval/` | 164 条 |
| BigCodeBench | `data/BigCodeBench/` | 下载仓库所有文件；v0.1.4 共 1,140 条 |
| UnLeakedTestBench | `data/UnLeakedTestBench/` | ULT 3,909 条，ULT_Lite 200 条；同时保留上游附带的 PLT 文件 |

UnLeakedTestBench 的 `.jsonl` 文件实际是 JSON 数组，应使用 `json.load()`；不要按行数当作样本数量。公开仓库中的 `tests` 字段不在本次检查中执行，也不视为权威 ground-truth 测试。

各资源的固定 revision、文件数和缓存位置记录在 `artifacts/setup/assets.json`。仅下载了计划中的 1.5B 模型，没有下载 3B/7B 权重。模型与数据按缓存内容复用，支持续传：

```bash
source scripts/activate.sh tools
python scripts/download_assets.py
```

## 已执行的验证

```bash
source scripts/activate.sh tools
python scripts/check_assets.py

source scripts/activate.sh rl
uv pip check --python .venv-rl/bin/python
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_environment.py
```

- 模型权重 SHA-256：`c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8`，与 Hugging Face 缓存 blob 标识一致。
- 模型和数据文件、项目内路径、Parquet 元数据和少量样本读取、ULT JSON 解析检查通过。
- 366 个已安装包的依赖一致性检查通过。
- Torch、Transformers、Datasets、Ray、vLLM、verl、FlashAttention、Sandbox Fusion SDK、BigCodeBench、EvalPlus、Cosmic Ray、coverage 的实际导入通过。
- Code-A1 的 `trainer_a1`（包含 FSDP worker 依赖）和 Sandbox Fusion reward 模块导入通过，见 `artifacts/setup/baseline-import-check.log`。
- 在离线模式下读取本地模型 config/tokenizer，并成功应用 chat template。

详细结果：`artifacts/setup/asset-check.json`、`environment-check.json`，以及同目录 `.log` 文件。`environment-check.json` 的 `passed` 仅代表脚本中导入和离线读取检查通过，不代表完整训练流程通过。

## 2026-10-31 运行验证待办（历史快照）

1. **GPU 推理**：当前为无 GPU 的登录节点。A100/H100 的即时申请均因资源繁忙退出，未运行推理。获得兼容 GPU 后运行：

   ```bash
   srun -p ice-gpu --gres=gpu:a100:1 -c 4 --mem=24G -t 00:08:00 bash scripts/gpu_smoke.sh
   ```

   该脚本仅加载本地模型、生成少量文本，保存 `artifacts/setup/environment-gpu-check.json`。不执行生成的代码，不训练。此旧版 CUDA/PyTorch 组合尚未验证 Blackwell/AMD GPU。

2. **Sandbox Fusion 服务**：SDK 已安装，但 `SANDBOX_FUSION_ENDPOINT` 未设置；当前节点未发现 Docker、Apptainer 或 Singularity。需要部署 PACE 允许的隔离执行服务或接入已有服务，再验证成功、失败和超时场景。没有将生成代码直接放在登录节点执行。
3. **最小 RL 和端到端评测**：还需把官方配置接到本地模型/数据/输出路径，并在 GPU 和沙箱就绪后验证。官方默认 coder/tester 各使用 4 张 GPU；单 GPU smoke test 不等于训练所需资源已满足。
4. **UnLeakedTestBench 入口**：上游 README 推荐 Python 3.12，本项目为匹配 Code-A1 使用 Python 3.10；数据已验证，具体生成/评测脚本的兼容性与缺失辅助模块仍需对接，不能将导入成功记为端到端评测通过。

## 配额与版本管理

缓存和运行资源均在当前 Scratch 项目内。安装后查询显示 Scratch 约 230.1/300 GB、971,920/1,000,000 文件（97.2%，是整个用户 Scratch 的用量，且会变化）；Home 并非这些下载的目标。文件数比容量更接近上限，因此关闭后续 `.pyc` 写入，并复用 uv hardlink 和 Hugging Face 缓存。已运行 `uv cache prune`，未发现可清理的失效条目；后续大规模安装需留意文件数量配额。

2026-10-31 按用户要求清理了 Scratch 内 33 处 Triton 编译缓存。清理前后文件配额从 972,018 降至 892,204（89.2%），释放约 79,814 个 inode，剩余约 107,796 个；容量约为 228/300 GB。保留了 Triton 软件包及其 10 处安装元数据目录。此前目录审计是清理前的历史快照，最新清理记录见 `artifacts/setup/triton-cleanup.json`。

`models/`、`data/`、`external/`、虚拟环境、缓存和运行日志均由 `.gitignore` 排除。脚本、锁文件和本说明可以提交到 Git；本次环境配置尚未自动提交或推送。

缓存变量依据 [Hugging Face 官方说明](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)，在库导入之前设置。

## 2026-11-08：第 1–2 周运行验证

当前有效的提交方式如下，均从项目根目录运行。`coe-gpu` 提交使用 `account=ece`、`qos=coe-ice`；泛型 `--gres=gpu:N` 可分配 H100 或 H200。

本阶段验收汇总在 [week12-summary.json](artifacts/setup/week12-summary.json)，各作业的最终退出状态在 [Slurm 记录](artifacts/setup/week12-slurm-accounting.txt)。失败与中断记录也保留在原始日志中。

| 入口 | 资源 | 范围 | 证据 |
| --- | --- | --- | --- |
| `sbatch scripts/gpu_smoke.sbatch` | coe-gpu，1 GPU，4 CPU，32 GiB，15 分钟 | 本地 Qwen 1.5B / vLLM 实际生成 | 6888046：H200，退出 0；`environment-gpu-check.json` |
| `sbatch scripts/sandbox_smoke.sbatch` | ice-cpu，2 CPU，8 GiB，10 分钟 | 成功、断言失败、1 秒超时、无项目挂载/外网访问，另验证 ULT pytest probe | 6888172：退出 0，5 项通过；`sandbox-check.json` |
| `sbatch scripts/benchmark_smoke.sbatch` | coe-gpu，1 GPU，4 CPU，32 GiB，20 分钟 | HumanEval/0、BigCodeBench/4、ULT Lite 首条各生成一个输出；有效代码送沙箱，无效输出判失败 | 6888122：退出 0；前两条代码通过，ULT 测试截断无效；`benchmark-smoke.json` |
| `sbatch scripts/week12_smoke.sbatch` | coe-gpu，2 GPU，8 CPU，80 GiB，45 分钟 | 官方训练数据小切片、coder/tester 各 1 GPU，1 次协同更新 | 6888089：退出 0；`trainer-data-check.json`、`week12-smoke-6888089.log`、`rl-smoke-check.json` |

推理验证生成了正确的 `add(a, b)` 函数。运行主机的驱动为 615.71.09，PyTorch CUDA runtime 为 12.4，GPU 为 NVIDIA H200。`USE_TF=0`、`USE_FLAX=0` 已加入项目环境，避免纯 PyTorch 流程导入不需要的框架。

### Python 沙箱部署

当前系统可用 Bubblewrap (`bwrap`)，没有找到 Apptainer/Singularity module。使用 [官方 SandboxFusion 源码](https://github.com/bytedance/SandboxFusion)，固定 commit `469de0845cb7d665bc68044d541aa3aa28c5b30a`；`scripts/sandbox_service.py` 复用其 `/run_code` 路由、请求/响应类型及状态判定，注册本地 Bubblewrap Python runner，上游源码保持不变。

这是一套 Python 专用适配部署，区别于官方完整 Docker 多语言服务：不提供文件传输、Jupyter、GPU 代码执行或数据库服务；也没有使用需要特权的上游 `lite` 隔离方式。每个请求分别创建 PID、网络与挂载 namespace，清空环境变量，只读挂载系统软件和专用运行环境；项目与 Home 不挂载。工作目录和 `/tmp` 使用 namespace 内临时文件系统，输出有长度限制，单次运行上限 30 秒，每个执行进程的地址空间上限 2 GiB，并发最多 4 个。服务只监听当前计算节点的 `127.0.0.1`，随 Slurm 作业退出。

重建这部分环境：

```bash
bash scripts/setup_sandbox.sh
```

`.venv-sandbox` 使用 PACE Python 3.11.9，只安装 API 依赖、NumPy 和 pytest，完整锁定列表见 `requirements/sandbox.lock`，没有新增 CUDA/PyTorch。其余 BigCodeBench 第三方依赖尚未完整安装；当前只验证了固定的 stdlib 样例。完整 benchmark 与 ULT 覆盖率/变异评测仍需在评测阶段补齐。

### 数据与最小训练配置

`scripts/prepare_week12.py` 合并 verl 默认配置、官方 1.5B_A1 配置与本地路径，输出 `artifacts/week12/baseline-local.yaml`。这是正式 baseline 的配置起点，尚未代表完整预算训练完成。

同一脚本另生成 `smoke.yaml` 和官方训练集中的 4 个短样例，实际验证 `RLHFDataset → collate_fn → DataProto`；仅读取数据，不在登录节点执行代码。训练 smoke 使用 batch size 2、coder/tester 各采样 2 个、最多生成 512 tokens、1 个训练 step，关闭训练前全量验证与 checkpoint 保存。其结果仅用于环境验收，不作正式 baseline 分数。

6888089 实际生成 4 个 coder 输出和 8 个 tester 输出，完成沙箱奖励、GRPO advantage、旧策略/参考策略 log probability 以及两个模型的优化器更新，作业总耗时 4 分 36 秒。梯度范数分别为 **2.010 / 2.977**（有限且非零），GPU 峰值已分配内存约 62.7 GiB/模型。原始 rollout、mistake book 和解析后的验收报告均已保存。控制台学习率显示 `0.000` 是保留三位小数的舍入；实际配置学习率为 `1e-6`。

作业结束后检查双模型完成标记与有限梯度指标：

```bash
source scripts/activate.sh tools
python scripts/summarize_rl_smoke.py artifacts/setup/week12-smoke-<jobid>.log
```

### 存储核对

11 月 8 日 `pace-quota` 显示 Scratch 约 **210.6/300 GiB（70.2%）**，**892,609/1,000,000 文件（89.3%）**。这是整个用户 Scratch 的共享额度，其他目录变化也会影响本项目。最初 `du` 审计显示本项目 `.cache` 约 16 GiB；模型与数据软链接、uv 硬链接需要避免重复计数，不能直接相加目录大小。

新增 `scripts/check_quota.py` 直接读取 `lfs quota`，所有验证作业在开始前要求至少 **20 GiB 空间和 20,000 个文件额度**；不足会退出。全部验证结束后 `pace-quota` 显示 **210.7/300 GiB（70.2%）**、**895,352/1,000,000 文件（89.5%）**，约剩 **89.3 GiB / 104,648 文件**。最新快照在 `artifacts/setup/quota-check.json`，人工可读原始记录在 `pace-quota-20261108-final.txt`。本次不清理用户其他目录，不下载其他模型，不保存 smoke checkpoint；Ray 临时 socket 与日志使用计算节点 `/tmp/a1-ray-<user>-<jobid>`，减少持久存储与长路径问题。

### 已发现并保留的失败记录

- 6888080：Bubblewrap 挂载 `/usr/local/pace-apps` 的软链接目标失败；改为挂载真实的 `/storage/pace-apps` 后，6888095 的四项沙箱验证通过。原报告保存在 `sandbox-check-6888080-failed.json`。
- 6888098：BigCodeBench/0 的参考测试涉及排列枚举，超过 10 秒 smoke 预算。改用固定 BigCodeBench/4 验证接口，不把该样例替换用于任何正式成绩计算；原日志保留。
- 6888122：ULT 10 的模型测试生成达到 512 token 上限后截断，判为无效输出并保留全文；未替换成成功样例。6888172 用该 FUT 的独立 pytest probe 验证了加载与执行路径，这个 probe 不算模型测试正确率。完整 ULT 指标尚未测量。

## 2026-11-09：固定预算实验完成

E0–E3 各完成 32-step 双模型训练并保存最终 HF 权重。五模型各完成 HumanEval 164、BigCodeBench 1140 和 ULT Lite 200/K=2 的适配评测。结果、图表与限制见 [FINAL_REPORT.md](FINAL_REPORT.md)，任务映射为 `artifacts/study/jobs.json`，完整统计为 `analysis.json`。

CPU 评测复用 `.venv-rl` 的完整依赖，在 Bubblewrap 内执行；120 秒超时、16 GiB 地址空间、并发 4。base 评分申请 8 CPU，后四组为 4 CPU，均保留 48 GiB 调度内存。BigCodeBench 参考实现 1087/1140 通过，53 个参考失败单列；ULT 为受限 assert/AST mutation 适配指标。

本轮实际分配 3.866 GPU-hours。八个最终 HF 模型共 52.96 GiB；最终 Scratch 用量 263.9/300 GiB，剩余 36.1 GiB，896,514/1,000,000 文件。该额度由用户整个 Scratch 共享；此前 210.7 GiB 是训练前快照。

## W&B 训练监控

已完成云端读回验收，见 `artifacts/study/wandb-final-verified.json`：四组各 32 步双模型 loss、五模型最终 eval 数值一致，附件仅配置与数值汇总；监控及最终上传作业均已成功退出。

用户指定只上传数值训练指标、配置、资源监控与评测汇总。项目为 [6257-adversarial-co-training](https://wandb.ai/mushroom_yeren/6257-adversarial-co-training/workspace)。四个训练组使用独立 run，另保留 base 评测对照。

`scripts/wandb_monitor.py` 在独立 CPU 作业内每 30 秒读取持久化训练记录，补记已有步骤并实时写入 W&B，GPU 训练无需因登录或网络等待而重启。指标包括 `actor_a/pg_loss`、`actor_b2/pg_loss`、KL、entropy、reward、梯度范数、token 长度、执行时间和内存统计；各模型评测完成后追加 pass@1、coverage 和有界 mutation 等汇总分数。当前轮原有 console-only 指标保留三位小数，无法恢复更早步骤丢失的精度；后续通过 `metrics_tracking.py` 的结构化 stream 可记录完整精度。控制台中舍入为 0 的学习率不应解释为真实学习率为 0，实际初始配置是 1e-6。

密钥从用户提供的 `.env` 读取，不打印、不写入 run config 或 artifacts；`.env` 已被 Git 忽略。云端项目和允许范围记录在 `artifacts/study/wandb-config.json`，当前为 `scope=metrics`，不上传题目、模型输出、逐条日志、数据集或权重。W&B 本地文件同样在忽略的 `artifacts/wandb/` 中。

在用户明确指定/授权目标后配置并验证项目：

```bash
source scripts/activate.sh rl
python scripts/configure_wandb.py
sbatch scripts/wandb_monitor.sbatch online
```

若只需本地 W&B 格式记录，使用 `sbatch scripts/wandb_monitor.sbatch offline`，不会向云端上传。不要同时启动多个在线 monitor 写入同一组 run。云端验收见 `artifacts/study/wandb-verified.json`，持续状态见 `wandb-status.json`。
