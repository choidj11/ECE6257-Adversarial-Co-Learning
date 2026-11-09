# Adversarial Co-Training — Progress Tracker

> 最后更新：2026-11-09（America/New_York）
> 项目：Adversarial Co-Training for Agentic Coding  
> 成员：Daniel Choi、Jack Li、Muyu Pan、Tianyou Zhao  
> 计划来源：[Adversarial Co-Training .pdf](<Adversarial Co-Training .pdf>)，共 3 页。

## 1. 目标与研究假设

**项目目标**：复现 Code-A1 的 coder–tester 对抗协同训练流程，并验证向 Test LLM 提供更丰富的执行反馈，是否能改善测试质量和 Code LLM 的表现。（PDF 第 1–3 页）

**核心假设**：在固定生成预算下，向 Test LLM 提供 branch coverage 和 selected runtime values，能够生成更多有效、能发现 bug 的单元测试，进而改善 coder performance。

**动机**：代码生成 RL 依赖单元测试作为反馈。测试需要有效且多样，才能提供可靠的学习信号并减少 reward hacking；现有流程对代码运行时状态的利用有限。

**训练闭环**：

```text
Problem specification + function signature
  → Code LLM 生成代码
  → Test LLM 生成对抗测试场景
  → Verifier 执行测试并计算奖励
  → RL 更新模型，进入下一轮

改进点：从执行过程中提取 branch coverage / selected runtime values，
        在当前轮 tester 生成前提供受限反馈。已实现细节见 EXPERIMENT_PROTOCOL.md。
```

**完成标准**：形成可复现的 baseline、带执行反馈的训练流程、固定预算下的对比与消融结果，以及最终报告。性能提升属于待验证假设；如未提升，也应报告实验结果和原因分析。

## 2. 实验范围与配置

| 项目 | PDF 中的计划 | 当前确认状态 |
| --- | --- | --- |
| Base model | Qwen2.5-Coder-Instruct，1.5B | 本地模型与 tokenizer 已下载；SHA-256、离线检查通过；coe-gpu H200 实际 vLLM 生成通过 |
| Training data | KodCode-V1 | 完整主训练集 484,097 条已下载；Code-A1 处理好的 9,688 条训练数据也已就绪 |
| RL framework | verl | 官方 Code-A1 定制版 0.4.0.dev0；两张 H200 上完成 coder/tester 各一次 RL 更新，作业 6888089 退出 0 |
| Inference engine | vLLM | 0.8.5.post1 已安装；H200 本地模型实际生成通过 |
| Execution sandbox | Sandbox Fusion（第 1 页） | 官方 API + Bubblewrap Python runner 已接通；正常/失败/超时及文件系统/网络隔离验证通过；完整多语言 Docker 服务未部署 |
| Tester evaluation | UnLeakedTestBench | 五个模型的 ULT Lite 200/K=2 assert 适配评测完成；指标与官方完整 Cosmic-Ray 范围的区别见报告 |
| Coder evaluation | HumanEval + BigCodeBench | 五个模型各完成 164/1140 题全量尝试，记录 raw 与 canonical 合格子集 pass@1 |
| Training signal | Branch coverage + selected runtime values | 已实现并用于 E1–E3 的完整 32-step 训练；trace、成本和采集失败可追溯 |
| Baseline reference | Code-A1 | 官方固定 commit 的受限规模复现完成；预算差异、方法与实际结果见 CODE_A1_METHOD.md 和 FINAL_REPORT.md |

PDF 第 3 页的末阶段计划还提到 “HumanEval, MBPP, etc.”，但主要实验配置与预期结果均列出 HumanEval、BigCodeBench、UnLeakedTestBench。本轮冻结协议采用这三个评测范围，MBPP 未纳入。

## 3. Current State

| 字段 | 当前状态 |
| --- | --- |
| Overall state | 本轮固定预算研究实验完成：四组各 32-step 训练、最终 HF 权重、五模型三 benchmark 适配评测、消融统计与报告；受限规模及环境限制见 FINAL_REPORT.md |
| Current week | 未知：PDF 给出相对周次，没有开始日期或完成日期 |
| 已确认完成 | 基础设施、四组各 32-step 双模型训练、最终权重、五模型统一评测、配对消融统计、图表及 FINAL_REPORT.md |
| 仓库可见证据 | models/、data/、external/、.venv-rl/、requirements/、scripts/；详细检查见 [环境说明](ENVIRONMENT.md) 和 artifacts/setup/ |
| 无法据此确认 | 团队其他成员或其他目录中的训练工作；本轮不涵盖论文完整训练预算、多 seed 或官方完整 Cosmic-Ray 评测 |
| Next milestone | 当前协议范围无剩余计算任务；扩展训练规模、多 seed 和完整官方评测环境属于后续扩展 |
| 当前依赖 | 本轮全部计算及 W&B 云端读回核验已完成；四组完整 loss 历史、五模型最终数值和附件范围均通过核验；团队分工/起始日期仍未提供 |

这里的“未见记录”仅表示当前仓库没有证据，不代表团队尚未开展该项工作。不根据周计划推断实际进度。

### 状态约定

| State | 含义 |
| --- | --- |
| `UNKNOWN` | 已纳入计划，但实际进度尚无证据；初始化的研发任务默认使用此状态 |
| `TODO` | 已确认尚未开始 |
| `IN_PROGRESS` | 正在执行，需记录当前产物和下一步 |
| `BLOCKED` | 有明确阻塞，需记录原因与解除条件 |
| `DONE` | 已达到验收条件，并有代码、日志、结果或文档作为证据 |
| `DEFERRED` | 明确延后或移出当前范围，需记录原因 |

负责人未分配时记为 `TBD`。下表中的实施拆分、验收条件和实验设计建议是为执行计划补充的内容，不是 PDF 已报告的结果。

## 4. Weekly Plan

原定周期为 7 周；实际起止日期及当前周次待确认。（PDF 第 1、3 页）

| Week | 原定目标 | 对应 Task | 阶段验收条件（建议） | Current state |
| --- | --- | --- | --- | --- |
| 1–2 | Prepare data & baselines：准备 KodCode-V1 和 benchmarks；集成 verl、vLLM 与 Qwen 1.5B | T1、T2 | 数据与评测入口可用；模型生成、沙箱执行及最小 RL 更新可运行 | `DONE`：最小运行验收通过；不包含正式全量评测或完整预算训练 |
| 3 | Replicate Code-A1：实现、测试并验证 baseline | T3 | 完成协同训练闭环，保存可复现实验配置、checkpoint 与 baseline 评测结果 | `DONE`：本轮受限规模协议验收完成；论文完整预算不在本轮范围 |
| 4–5 | Implement improvements：加入 branch coverage 和 runtime values | T4 | 执行反馈可提取并传给 tester；增强版流程可运行 | `DONE`：本轮受限规模协议验收完成；论文完整预算不在本轮范围 |
| 6–7 | Testing & report writing：benchmark evaluation、消融和报告 | T5、T6 | 完成主要评测、对比和消融，形成结果分析与最终报告 | `DONE`：本轮受限规模协议验收完成；论文完整预算不在本轮范围 |

## 5. Task / Sub-task Tracker

### Task 总览

| ID | Task | Week | Depends on | Owner | Current state | Next step |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | 维护目标、状态与实验日志 | 全程 | — | TBD | `IN_PROGRESS` | 补充实际周次、分工和已有产物位置 |
| T1 | 准备训练数据与评测集 | 1–2 | — | TBD | `DONE` | 数据和最小训练/评测入口已通过；正式评测按 T5 的协议开展 |
| T2 | 搭建模型、RL、推理与沙箱环境 | 1–2 | — | TBD | `DONE` | H200 推理、Python 沙箱和最小双模型更新通过；正式训练前评估预算与存储 |
| T3 | 复现 Code-A1 baseline | 3 | T1、T2 | TBD | `DONE` | 本轮固定预算范围完成；结果、配置差异与限制见 FINAL_REPORT.md |
| T4 | 加入执行反馈 | 4–5 | T3 | TBD | `DONE` | 本轮固定预算范围完成；结果、配置差异与限制见 FINAL_REPORT.md |
| T5 | 评测、对比与消融 | 6–7 | T3、T4 | TBD | `DONE` | 本轮固定预算范围完成；结果、配置差异与限制见 FINAL_REPORT.md |
| T6 | 整理结果与最终报告 | 6–7 | T5 | TBD | `DONE` | 本轮固定预算范围完成；结果、配置差异与限制见 FINAL_REPORT.md |

### Sub-task 明细

每个子任务继承父任务的计划周次；负责人均为 `TBD`，待实际分配后更新。

| ID | Sub-task / 目标 | Current state | 验收条件 / 证据 | Next step |
| --- | --- | --- | --- | --- |
| T0.1 | 阅读 PDF，整理目标和 7 周计划 | `DONE` | 本文件第 1–4 节，来源为目标 PDF | 后续需求变化时记录决策 |
| T0.2 | 核实真实进度与分工 | `UNKNOWN` | 明确当前周次、各任务负责人、已有工作与产物路径 | 汇总团队当前进展，逐项更新状态 |
| T1.1 | 准备 KodCode-V1 | `DONE` | data/KodCode-V1/；固定 revision、15 个主训练分片共 484,097 条；Parquet 元数据与样本读取通过 | 使用 Code-A1 处理好的 9,688 条 hard 数据接训练入口 |
| T1.2 | 准备三个主要 benchmark | `DONE` | 三个主集读取通过；6888122 的 HumanEval/0、BigCodeBench/4 生成代码通过；ULT K=1 生成截断并判无效；见 benchmark-smoke.json | 正式协议下补齐完整依赖、覆盖率/变异与全量评测 |
| T1.3 | 检查数据与训练接口 | `DONE` | 官方数据的 4 条切片经实际 RLHFDataset、collate_fn、DataProto 拼装通过；trainer 的数据/配置校验通过；见 trainer-data-check.json | 在完整预算训练中使用本地全量处理数据 |
| T2.1 | 确认计算资源与软件环境 | `DONE` | coe-gpu/account ece/QoS coe-ice 提交成功；H200 驱动 615.71.09；现有 torch/vLLM/FlashAttention 实际生成通过 | 按训练预算分配资源并持续检查配额 |
| T2.2 | 加载 Qwen 1.5B 并接入 vLLM | `DONE` | 6888046：H200 实际生成 add(a,b)，退出 0；见 environment-gpu-check.json 与作业日志 | 在 baseline 中复用本地模型 |
| T2.3 | 部署 Sandbox Fusion | `DONE` | 官方 API + Bubblewrap Python runner，6888095 正常/断言失败/超时/隔离均通过；非完整 Docker 多语言部署；见 ENVIRONMENT.md | 在训练作业内启动本地服务并限制并发 |
| T2.4 | 接通 verl 的最小训练路径 | `DONE` | 6888089：两张 H200、coder/tester 各 1 GPU、1 step 完成，退出 0；梯度范数 2.010/2.977；rl-smoke-check.json | 以正式预算扩展训练，保留 smoke 作为复现入口 |
| T3.1 | 核对 Code-A1 方法与实现 | `DONE` | CODE_A1_METHOD.md 核对了实际 coder/tester 奖励、Mistake Book、GRPO、FSDP2 和本轮配置偏差 | 按协议开展受限规模复现，不声称论文绝对分数复现 |
| T3.2 | 实现 coder–tester–verifier 闭环 | `DONE` | 6888089 完成官方流程的生成、沙箱奖励、GRPO advantage 与双模型更新；4 个 coder 输出、8 个 tester 输出，保留 rollout 日志 | 正式训练时扩大规模并检查奖励与稳定性 |
| T3.3 | 运行 baseline 训练 | `DONE` | 32 steps 完成；最终 coder/tester HF 权重、逐步梯度与原始日志保存在 artifacts/study/E0/；汇总见 FINAL_REPORT.md | 本轮已完成；需要扩大规模时另立实验协议 |
| T3.4 | 验证 baseline 复现 | `DONE` | 受限规模 baseline 的 HumanEval/BigCodeBench/ULT Lite 结果齐全；配置偏差和 canonical 校准失败均在 FINAL_REPORT.md 明示；未声称论文绝对分数复现 | 本轮已完成；需要扩大规模时另立实验协议 |
| T4.1 | 提取 branch coverage | `DONE` | profile_probe.py + feedback-check.json：已知函数部分覆盖 1/2、完整覆盖 2/2，关联源码行弧并记录耗时 | 从实际训练 trace 分析反馈有效性 |
| T4.2 | 提取 selected runtime values | `DONE` | 独立 profiler 收集候选函数返回时的有限 locals/return；已知返回值检查通过；测试帧与 oracle 答案不进入提示词 | 分析实际采集失败、长度与开销 |
| T4.3 | 将执行反馈接入 Test LLM | `DONE` | 三组各 256 条实际训练反馈记录，均不超过 384 tokens，关联源码、trace 和执行耗时；artifacts/study/E*/feedback/ | 本轮已完成；需要扩大规模时另立实验协议 |
| T4.4 | 运行增强版 co-training | `DONE` | E1/E2/E3 各完成 32 steps 并保存两个模型的最终 HF 权重；analysis.json 含完整训练曲线 | 本轮已完成；需要扩大规模时另立实验协议 |
| T5.1 | 明确预算与指标 | `DONE` | EXPERIMENT_PROTOCOL.md、artifacts/study/protocol.json：固定数据、seed、生成预算、反馈额外成本及指标 | 按冻结协议执行，不依据模型结果更换样本 |
| T5.2 | 评估 Test LLM | `DONE` | 五个模型在 ULT Lite 200 条、K=2 的适配评测完成；coverage、有界 mutation、无效测试和测量失败均在 FINAL_REPORT.md 单列 | 本轮已完成；需要扩大规模时另立实验协议 |
| T5.3 | 评估 Code LLM | `DONE` | 五个模型各 164 HumanEval + 1140 BigCodeBench 全量尝试；报告 raw/参考合格子集 pass@1 和环境失败列表 | 本轮已完成；需要扩大规模时另立实验协议 |
| T5.4 | 开展消融与结果分析 | `DONE` | coverage/runtime/结合的四组消融完成；analysis.json 保存配对 task bootstrap、成本与正负案例；单 seed 限制保留 | 本轮已完成；需要扩大规模时另立实验协议 |
| T6.1 | 汇总图表与结论 | `DONE` | report/figures/ 中三幅结果/训练图，analysis.json 中可追溯统计；结论包含无提升/负向情况 | 本轮已完成；需要扩大规模时另立实验协议 |
| T6.2 | 完成最终项目报告 | `DONE` | FINAL_REPORT.md 已生成，包含方法、预算、实际指标、消融、失败案例、资源与限制及复现入口 | 本轮已完成；需要扩大规模时另立实验协议 |

## 6. Next Steps

以下为 2026-11-09 完成本轮验收后的状态；完整环境说明见 [ENVIRONMENT.md](ENVIRONMENT.md)。

1. **本轮已完成**：结果、证据与适配评测限制见 [FINAL_REPORT.md](FINAL_REPORT.md)。
2. **可选扩展**：增加训练规模与 seed，在完整官方依赖环境复核，扩大 mutation 算子与反馈探针。
3. **团队信息**：项目原始起止日期和其他成员分工仍待团队提供，不据此虚构状态。

## 7. Experiment Plan & Results

### 本轮固定实验矩阵

用户已授权执行 baseline、评测及后续任务。本轮以统一 32-step、单 seed、128 个问题的受限规模预算执行四组；正式范围与偏差见 [实验协议](EXPERIMENT_PROTOCOL.md)。不将该预算称为论文完整规模复现。

| Plan ID | Variant | Branch coverage | Runtime values | Current state | Result |
| --- | --- | --- | --- | --- | --- |
| E0 | Code-A1 baseline | 无新增反馈 | 无新增反馈 | `DONE` | 32 steps；HumanEval raw 72.56%；详细结果与限制见 FINAL_REPORT.md |
| E1 | Coverage only | 有 | 无新增反馈 | `DONE` | 32 steps；HumanEval raw 71.34%；详细结果与限制见 FINAL_REPORT.md |
| E2 | Runtime values only | 无新增反馈 | 有 | `DONE` | 32 steps；HumanEval raw 72.56%；详细结果与限制见 FINAL_REPORT.md |
| E3 | Coverage + runtime values | 有 | 有 | `DONE` | 32 steps；HumanEval raw 71.95%；详细结果与限制见 FINAL_REPORT.md |

各组的模型、数据划分、训练设置、评测协议与生成上限固定；新增反馈输入和执行成本另记。保留 base 未训练模型作为起点对照。作业依赖与 ID 见 artifacts/study/jobs.json。

### 本轮已固定并测量的指标

| 维度 | 指标 / 记录内容 | 当前状态 |
| --- | --- | --- |
| Tester 质量 | 有效测试比例、有效测试/query、有界 mutation kill、失败数量 | 已测量；见 FINAL_REPORT.md |
| Coder 表现 | HumanEval / BigCodeBench raw 与 canonical 合格子集 pass@1 | 已测量；见 FINAL_REPORT.md |
| 反馈效果 | 实际 branch/line coverage、反馈长度和失败案例 | 已测量；见 analysis.json 与报告 |
| 资源与稳定性 | Slurm GPU-hours、真实输入/输出 tokens、反馈耗时、超时/失败、训练曲线 | 已记录；见 resources.json、analysis.json |

### 实验记录模板

每次实际运行单独追加一条；计划 ID 不等于已经发生的 run。本轮 E0–E3 固定预算对比实验已完成；原始运行、逐题结果和分析见 artifacts/study/ 与 FINAL_REPORT.md。

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
Artifacts: 日志、checkpoint、结果文件或图表路径
Issues / Interpretation:
Next step:
```

## 8. Open Questions & Dependencies

GPU allocation 和 Python 沙箱已在 Slurm 作业内接通；正式训练预算、评测完整依赖等仍需在对应阶段确定。

| ID | 待确定事项 | 影响 | 处理时机 |
| --- | --- | --- | --- |
| Q1 | 起始日期、当前周次和实际完成情况 | 周计划与真实进度对齐 | 下一次进度更新 |
| Q2 | 各成员负责范围与已有产物位置 | 避免重复工作，补齐状态证据 | 下一次进度更新 |
| Q3 | 已明确：每组 2 GPU、32 steps、单作业 1 小时；16 小时是 partition 单作业上限，QoS 另有限制 | 细节与官方语义见 EXPERIMENT_PROTOCOL.md | 本轮已解决 |
| Q8 | Python 隔离沙箱已通过；复用完整 RL 依赖，BCB canonical 1087/1140 通过，53 个环境失败明确单列 | 原始结果与校准子集并列报告；完整多语言服务未部署 | 扩展官方环境时复核 |
| Q4 | 已固定官方 commit，核对奖励函数、Mistake Book 与双模型更新 | 见 CODE_A1_METHOD.md | 本轮已解决 |
| Q5 | 已固定前两个训练测试探针、candidate 返回时 locals、384-token 上限 | 见 EXPERIMENT_PROTOCOL.md 和实际 feedback 记录 | 本轮已解决 |
| Q6 | 已固定 32 steps、单 seed、相同输出/query 预算；额外输入及执行成本另计 | 题目 bootstrap 不代替多 seed 方差 | 本轮已解决 |
| Q7 | 冻结协议采用 HumanEval、BigCodeBench 与 ULT Lite；MBPP 不在本轮范围 | 未来若加入需另定协议 | 本轮范围已明确 |

## 9. Progress Log

按时间追加，记录事实、证据和下一步；计划中的实验不写成已完成结果。

| Date | Task / Sub-task | Event / Evidence | State / Outcome | Next step |
| --- | --- | --- | --- | --- |
| 2026-10-30 | T0.1 | 阅读 Adversarial Co-Training .pdf 全部 3 页，提取目标、配置与 7 周计划 | `DONE`：研究目标和原定计划已整理 | 核实实际进度 |
| 2026-10-30 | T0 | 检查当前仓库，创建 progress.md，补充任务拆分、验收条件、建议实验矩阵和日志模板 | `IN_PROGRESS`：tracker 已建立；研发任务保留 `UNKNOWN` | 补充当前周次、负责人及实际产物 |
| 2026-10-30 | T1、T2 | 建立 scripts/env.sh、项目内 uv/.venv-tools、依赖锁文件；下载模型和官方代码；完整数据下载中断 | `IN_PROGRESS`：未将未完成安装标为已就绪 | 续传数据并安装依赖 |
| 2026-10-31 | T1.1、T1.2、T1.3 | 补齐全部计划数据，建立 data/ 和 models/ 项目内链接；模型 SHA-256、15 个 KodCode 主分片和各 benchmark 读取检查通过 | T1.1 `DONE`；详细数量、revision 和路径见 artifacts/setup/assets.json、asset-check.json | 接通训练/评测入口 |
| 2026-10-31 | T2.1、T2.2 | 安装训练与评测依赖、Code-A1 定制 verl 和 FlashAttention；366 包依赖检查、12 项关键包导入、离线 config/tokenizer 检查通过 | `IN_PROGRESS`：Python 环境验证通过；GPU 推理与 RL 更新尚未验证 | GPU allocation 中验证推理 |
| 2026-10-31 | T2.2 | 单 GPU A100/H100 即时申请（6703728、6703769），均因资源繁忙退出；未执行推理或训练 | `IN_PROGRESS`：不计为 GPU 验证通过 | 有空闲资源时重试短推理检查 |
| 2026-10-31 | T2.3 | 确认 SDK 可导入，但 endpoint 未配置、当前节点未发现容器运行时 | `BLOCKED`：执行服务未就绪 | 确定可用的服务部署/接入方式 |
| 2026-10-31 | T0、T2.1 | 完成 Scratch inode 审计；按用户要求删除 33 处 Triton 编译缓存，保留软件安装元数据 | 文件配额从 972,018 降至 892,204（89.2%）；清理日志见 artifacts/setup/triton-cleanup.json | 有 GPU 资源时继续推理与训练环境验证 |
| 2026-11-08 | T2.1、T2.2 | sbatch 6888046：coe-gpu/account ece/QoS coe-ice，分配 H200；本地 Qwen/vLLM 生成 add(a,b) 成功 | `DONE`：退出 0；environment-gpu-check.json、gpu-smoke-6888046.log | 在训练路径中验证生成 |
| 2026-11-08 | T2.3 | 获取固定 SandboxFusion 源码，配置 Python API + Bubblewrap runner；6888080 因软链接挂载失败，修正真实软件路径后 6888095 通过四项验证 | `DONE`：Python 正常/失败/超时/隔离通过；保留失败报告；无完整多语言容器 | 在各作业内自动启动 endpoint |
| 2026-11-08 | T1.2 | 6888098：BigCodeBench/0 参考测试超过 10 秒预算；固定改用 /4 后，6888122 的 HumanEval/0、BigCodeBench/4 生成代码通过；ULT 10 测试生成截断并判无效 | `DONE`：最小入口验证完成，退出 0；不是正式成绩；见 benchmark-smoke.json | 按完整协议开展正式评测 |
| 2026-11-08 | T1.2、T2.3 | 6888172：沙箱四项检查及真实 ULT 函数的独立 pytest probe 均通过 | `DONE`：退出 0；probe 仅检查 pytest/FUT 加载，不算模型测试正确率 | 补齐 ULT 正式 coverage/mutation 流程 |
| 2026-11-08 | T1.3、T2.4、T3.2 | 6888089：官方数据 4 条切片，batch 2、n=2、两张 H200；生成 4 个 coder / 8 个 tester 输出，完成奖励、advantage 和两模型更新 | `DONE`：1 step，退出 0，耗时 4:36；梯度范数 2.010/2.977；rl-smoke-check.json、trainer-data-check.json、rollout 和原始日志；未保存 checkpoint | 进入正式 baseline 训练准备 |
| 2026-11-08 | T0、T2.1 | pace-quota 最终核对 Scratch 210.7/300 GiB（70.2%），895,352/1,000,000 文件（89.5%）；新增直接 lfs quota 的启动检查 | 约剩 89.3 GiB / 104,648 文件；运行前保留至少 20 GiB / 20,000 文件；未删除其他用户目录内容 | 正式训练前估计 checkpoint 峰值；见 pace-quota-20261108-final.txt |
| 2026-11-08 | T3.1、T5.1 | 用户授权 baseline、评测及后续任务；冻结 32-step/128 问题/单 seed 的四组资源受限实验，明确与论文训练规模不同 | `DONE`：协议与方法核对文档、protocol.json、各组配置可追溯 | 按冻结预算运行 |
| 2026-11-08 | T2.1 | 核实 coe-gpu 单作业上限 16 小时；coe-ice MaxTRESRunMinsPerUser=1920 GPU 分钟，是运行中剩余预留额度，非生命周期累计上限 | 训练任务预留由 2 小时缩为 1 小时以改善 backfill；32-step 预算不变 | 记录实际 GPU-hours |
| 2026-11-08 | T3.3、T4.4 | E0 6888252、E1 6888295、E2 6888296、E3 6888297 已提交；当前均等待双 H200 allocation | `IN_PROGRESS`：仅排队，不计为训练完成；预期只保存最后 HF 模型 | 等待资源并检查实际训练输出 |
| 2026-11-08 | T4.1、T4.2、T5.2 | CPU 作业 6888280/6888341：验证实际 branch arcs、返回时 locals、反馈字段屏蔽和长度；剔除已知错误断言、检出两个人工已知变异并覆盖全部分支 | `DONE`：feedback-check.json、study-evaluation-check.json；上游代码未修改 | 验证实际增强训练集成 |
| 2026-11-08 | T5、T6 | base 生成 6888290、全量 canonical 校准 6888310 运行中；各组后续生成、CPU 评分与报告依赖已提交 | `IN_PROGRESS`：HumanEval 164 canonical 全通过；BigCodeBench 发现少量网络/NLTK/版本失败，单列记录；尚无四组正式结果 | 依赖成功后评分和生成报告，完整映射见 jobs.json |
| 2026-11-08 | T3、T4 | 为改善排队，将 GPU 类型约束扩展为 coe-gpu 的 H100/H200，数量与预算不变；E0–E3 已启动 | `IN_PROGRESS`：E0 H100、E1 H200；各组已通过首轮实际更新，无需为监控重启训练；实际硬件由 sacct 记录 | 完成 32 steps 与最终 checkpoint |
| 2026-11-08 | T5 | base 的 1304 题 coder 与 ULT Lite 200/K=2 生成完成；canonical 校准完成 | HumanEval 164/164、BigCodeBench 1087/1140 canonical 通过；53 个失败逐题保留，base CPU 评分进行中 | 所有模型使用同一校准口径 |
| 2026-11-08 | T0、T6 | 用户要求 W&B 监控，提供 .env 密钥并授权创建项目，明确仅数值指标/配置/资源/评测汇总；创建并读回确认 mushroom_yeren/6257-adversarial-co-training 为 PRIVATE | 云端 E0–E3 loss/reward 历史已读回验证；在线监控 6888688 补记并持续同步，最终数值汇总 6888769 依赖报告/监控成功；不上传题目、输出、逐条日志、数据集或权重 | 对最终 eval score 再作云端读回验证；密钥始终不打印且被 Git 忽略 |
| 2026-11-09 | T3.3、T4.3、T4.4 | E0–E3 各完成 32 steps，四作业均 COMPLETED/0:0；两角色各 32 步有限梯度，八模型 16 个 HF shards 合计 52.96 GiB；三增强组各 256 条反馈，最长 143/365/382 tokens | 训练验收通过；Scratch 263.8/300 GiB，约剩 36.2 GiB；W&B 四组 1–32 步双模型 loss 已云端读回验证，base eval score 与本地一致 | 完成训练后统一评测与最终云端分数核验 |
| 2026-11-09 | T5 | 四组 coder 各 1304 条输出和完成标记已核验；解除冗余评测依赖，使 CPU coder 评分可与 tester 生成重叠；base 完整评分耗时 16:31，后续 CPU 上限缩短为 1 小时以改善 backfill | 只调整执行调度；模型、样本、指标和采样预算沿用冻结协议；ULT 评分等待完整 tester 标记 | 继续核对 CPU 任务及评分输出 |
| 2026-11-09 | T3–T6 | 四组 32-step 训练、最终 HF 权重、五模型 1304 coder 题和 ULT Lite 200/K=2 适配评测全部产生结果；分析与报告通过证据检查 | `DONE`：本轮固定预算范围；FINAL_REPORT.md、artifacts/study/analysis.json、report/figures/；保留单 seed、环境失败与有界 mutation 的限制 | 扩展实验需另定预算与协议 |
| 2026-11-09 | T0、T6 | 6888608 报告、6888688 监控、6888769 云端核验均 COMPLETED/0:0；云端四组各 32 步双模型 loss 和五模型最终数值逐项读回一致，附件仅 config.yaml/metrics.json/protocol.json | `DONE`：wandb-final-verified.json；本轮 3.866 GPU-hours；Scratch 263.9/300 GiB，剩余 36.1 GiB；主要配对区间均包含 0，尚无稳定提升证据 | 本轮范围已完成，无剩余 Slurm 任务 |

### 后续更新规则

- 每次完成工作后，更新对应 sub-task 的 `Current state`、证据和 `Next step`，同步父 task 的状态。
- 标为 `DONE` 时附上可检查的产物；标为 `BLOCKED` 时写明原因和解除条件。
- 每次训练或评测追加实验记录，保留失败和中断记录，不覆盖历史结果。
- 修改目标、范围、预算或周计划时，在 log 中注明日期、原因与影响，并更新本文件顶部日期。
