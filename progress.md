# Adversarial Co-Training — Progress Tracker

> 最后更新：2026-10-30（America/New_York）  
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
        将反馈提供给 Test LLM。具体反馈时机与格式待设计。
```

**完成标准**：形成可复现的 baseline、带执行反馈的训练流程、固定预算下的对比与消融结果，以及最终报告。性能提升属于待验证假设；如未提升，也应报告实验结果和原因分析。

## 2. 实验范围与配置

| 项目 | PDF 中的计划 | 当前确认状态 |
| --- | --- | --- |
| Base model | Qwen2.5-Coder-Instruct，1.5B | 尚无模型下载或运行记录 |
| Training data | KodCode-V1 | 尚无数据准备记录 |
| RL framework | verl | 尚无环境或集成记录 |
| Inference engine | vLLM | 尚无环境或集成记录 |
| Execution sandbox | Sandbox Fusion（第 1 页） | 尚无部署或执行记录 |
| Tester evaluation | UnLeakedTestBench | 尚无评测记录 |
| Coder evaluation | HumanEval + BigCodeBench | 尚无评测记录 |
| Training signal | Branch coverage + selected runtime values | 反馈格式与接入方式待设计 |
| Baseline reference | PDF 使用 Coder-A1 / Code-A1 两种写法；仓库有 [code-A1.pdf](code-A1.pdf) | 参考论文及实现细节待核对；本表暂沿用 Code-A1 |

PDF 第 3 页的末阶段计划还提到 “HumanEval, MBPP, etc.”，但主要实验配置与预期结果均列出 HumanEval、BigCodeBench、UnLeakedTestBench。当前以这三个为主要评测范围，MBPP 是否加入待确认。

## 3. Current State

| 字段 | 当前状态 |
| --- | --- |
| Overall state | 计划已整理；实际研发进度待核实 |
| Current week | 未知：PDF 给出相对周次，没有开始日期或完成日期 |
| 已确认完成 | 已阅读目标 PDF 的 3 页内容并建立本 tracker |
| 仓库可见证据 | 根目录包含 README.md、两份 PDF；本次检查未见实现代码、环境配置、训练日志或评测结果 |
| 无法据此确认 | 数据是否已在其他目录准备、模型是否已下载、是否已在集群运行训练、成员是否已有独立实现 |
| Next milestone | 建议先核实已有工作，再按依赖推进数据、环境和 baseline |
| 当前依赖 | 项目起始日期、已有产物位置、任务分工、可用计算资源，以及 baseline 的实现来源 |

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
| 1–2 | Prepare data & baselines：准备 KodCode-V1 和 benchmarks；集成 verl、vLLM 与 Qwen 1.5B | T1、T2 | 数据与评测入口可用；模型生成、沙箱执行及最小 RL 更新可运行 | `UNKNOWN` |
| 3 | Replicate Code-A1：实现、测试并验证 baseline | T3 | 完成协同训练闭环，保存可复现实验配置、checkpoint 与 baseline 评测结果 | `UNKNOWN` |
| 4–5 | Implement improvements：加入 branch coverage 和 runtime values | T4 | 执行反馈可提取并传给 tester；增强版流程可运行 | `UNKNOWN` |
| 6–7 | Testing & report writing：benchmark evaluation、消融和报告 | T5、T6 | 完成主要评测、对比和消融，形成结果分析与最终报告 | `UNKNOWN` |

## 5. Task / Sub-task Tracker

### Task 总览

| ID | Task | Week | Depends on | Owner | Current state | Next step |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | 维护目标、状态与实验日志 | 全程 | — | TBD | `IN_PROGRESS` | 补充实际周次、分工和已有产物位置 |
| T1 | 准备训练数据与评测集 | 1–2 | — | TBD | `UNKNOWN` | 检查 KodCode-V1 与三个 benchmark 的可用性 |
| T2 | 搭建模型、RL、推理与沙箱环境 | 1–2 | — | TBD | `UNKNOWN` | 确认计算资源与依赖，运行最小生成及执行验证 |
| T3 | 复现 Code-A1 baseline | 3 | T1、T2 | TBD | `UNKNOWN` | 核对参考论文、实现、奖励与模型更新流程 |
| T4 | 加入执行反馈 | 4–5 | T3 | TBD | `UNKNOWN` | 明确覆盖率和运行时值的采集范围、反馈时机与格式 |
| T5 | 评测、对比与消融 | 6–7 | T3、T4 | TBD | `UNKNOWN` | 先确定预算口径、指标与实验矩阵，再开展正式对比 |
| T6 | 整理结果与最终报告 | 6–7 | T5 | TBD | `UNKNOWN` | 随实验保存配置、日志、图表与结论依据 |

### Sub-task 明细

每个子任务继承父任务的计划周次；负责人均为 `TBD`，待实际分配后更新。

| ID | Sub-task / 目标 | Current state | 验收条件 / 证据 | Next step |
| --- | --- | --- | --- | --- |
| T0.1 | 阅读 PDF，整理目标和 7 周计划 | `DONE` | 本文件第 1–4 节，来源为目标 PDF | 后续需求变化时记录决策 |
| T0.2 | 核实真实进度与分工 | `UNKNOWN` | 明确当前周次、各任务负责人、已有工作与产物路径 | 汇总团队当前进展，逐项更新状态 |
| T1.1 | 准备 KodCode-V1 | `UNKNOWN` | 数据版本、位置、划分、样本格式及加载结果有记录 | 确认获取入口并检查少量样本 |
| T1.2 | 准备三个主要 benchmark | `UNKNOWN` | HumanEval、BigCodeBench、UnLeakedTestBench 的版本与运行方式有记录 | 检查评测依赖并各运行最小样例 |
| T1.3 | 检查数据与训练接口 | `UNKNOWN` | 题目描述、函数签名、参考解/测试等实际字段与训练输入对应清楚；训练与评测划分明确 | 对照 baseline 确认转换与校验需求 |
| T2.1 | 确认计算资源与软件环境 | `UNKNOWN` | GPU、显存、CUDA、Python 和依赖版本，以及可执行环境配置有记录 | 盘点资源并确定兼容的依赖组合 |
| T2.2 | 加载 Qwen 1.5B 并接入 vLLM | `UNKNOWN` | 模型/tokenizer 版本明确，完成一次代码或测试生成 | 核实模型位置，运行最小推理 |
| T2.3 | 部署 Sandbox Fusion | `UNKNOWN` | 能执行代码和测试，返回结果，处理超时及异常 | 验证一组成功、失败与超时样例 |
| T2.4 | 接通 verl 的最小训练路径 | `UNKNOWN` | generation → reward → RL update 能完成一次更新，配置和日志可追溯 | 连接推理、奖励接口和训练器 |
| T3.1 | 核对 Code-A1 方法与实现 | `UNKNOWN` | 明确参考版本、coder/tester 输入、奖励定义、更新顺序及必要超参数 | 阅读 code-A1.pdf 并定位对应实现 |
| T3.2 | 实现 coder–tester–verifier 闭环 | `UNKNOWN` | 一条训练样例能完整流转，测试有效性与奖励计算可检查 | 先运行小规模端到端流程 |
| T3.3 | 运行 baseline 训练 | `UNKNOWN` | 保存配置、seed、预算、日志与 checkpoint，记录训练稳定性 | 在小规模验证通过后运行计划预算 |
| T3.4 | 验证 baseline 复现 | `UNKNOWN` | 保存 coder/tester 评测结果；与参考设置的偏差有解释 | 评测 baseline 并记录复现差异 |
| T4.1 | 提取 branch coverage | `UNKNOWN` | 分支覆盖信息能关联到具体样例/执行；采集开销有记录 | 确定采集方式和反馈粒度 |
| T4.2 | 提取 selected runtime values | `UNKNOWN` | 变量选择、采集位置、截断与序列化规则明确；示例可检查 | 确定要暴露的运行时值及长度限制 |
| T4.3 | 将执行反馈接入 Test LLM | `UNKNOWN` | tester 输入中包含可追溯反馈；反馈时机和 prompt 有版本记录 | 设计输入格式，并验证一个完整交互样例 |
| T4.4 | 运行增强版 co-training | `UNKNOWN` | 相对 baseline 的改动、配置、预算、日志与 checkpoint 可追溯 | 在相同对比口径下运行增强版 |
| T5.1 | 明确预算与指标 | `UNKNOWN` | 生成预算、输入反馈成本、评测设置和指标定义可复现 | 在正式对比前固定实验协议 |
| T5.2 | 评估 Test LLM | `UNKNOWN` | UnLeakedTestBench 结果及有效测试/发现 bug 的分析有记录 | 对比 baseline 与增强版 tester |
| T5.3 | 评估 Code LLM | `UNKNOWN` | HumanEval、BigCodeBench 的 baseline 与增强版结果完整 | 使用一致的评测协议运行两个版本 |
| T5.4 | 开展消融与结果分析 | `UNKNOWN` | 分离 coverage、runtime values 的贡献，记录失败案例与运行开销 | 依据下方建议矩阵确定可执行范围 |
| T6.1 | 汇总图表与结论 | `UNKNOWN` | 每个结论能定位到实验记录，包含未提升或不稳定的情况 | 随实验完成更新结果表 |
| T6.2 | 完成最终项目报告 | `UNKNOWN` | 方法、复现设置、主要结果、消融、局限与复现步骤完整 | 建立报告提纲，再填入验证后的结果 |

## 6. Next Steps

以下为基于当前可见证据的建议顺序，不表示这些工作在其他位置尚未开展。

1. **核实进度（T0.2）**：补充项目起始日期、当前周次、成员分工，以及已有代码、数据、checkpoint 和运行日志的位置。
2. **核对 baseline（T3.1）**：阅读 code-A1.pdf，明确对应实现和复现设置，避免后续实现偏离 baseline。
3. **验证数据与环境（T1、T2）**：根据已有产物补齐数据加载、模型生成、沙箱执行和最小 RL 更新。
4. **建立可比较的 baseline（T3）**：完成协同训练和初始评测，再接入执行反馈。
5. **在正式实验前固定对比协议（T5.1）**：特别明确“固定生成预算”的口径，以及反馈输入增加的 token 和运行开销如何记录。

## 7. Experiment Plan & Results

### 建议实验矩阵（待确认）

PDF 明确要求 baseline、执行反馈改进与消融；以下四组是据此提出的具体拆分，尚未确认执行。

| Plan ID | Variant | Branch coverage | Runtime values | Current state | Result |
| --- | --- | --- | --- | --- | --- |
| E0 | Code-A1 baseline | 无新增反馈 | 无新增反馈 | `UNKNOWN` | 尚无结果记录 |
| E1 | Coverage only | 有 | 无新增反馈 | `UNKNOWN` | 尚无结果记录 |
| E2 | Runtime values only | 无新增反馈 | 有 | `UNKNOWN` | 尚无结果记录 |
| E3 | Coverage + runtime values | 有 | 有 | `UNKNOWN` | 尚无结果记录 |

各组应保持可比较的模型、数据划分、训练设置、评测协议与生成预算，并记录反馈带来的额外成本。具体 seed 数量、预算数值、训练步数和采样参数均待确定。

### 需要固定的指标（建议）

| 维度 | 指标 / 记录内容 | 当前状态 |
| --- | --- | --- |
| Tester 质量 | 有效测试比例、发现 bug 的表现、固定预算下的有效发现数量；具体定义依 benchmark 确认 | 待定义，无测量结果 |
| Coder 表现 | HumanEval、BigCodeBench 的官方评测指标；是否使用 pass@k 及具体 k 待确认 | 待定义，无测量结果 |
| 反馈效果 | 分支覆盖情况、反馈长度、典型有效/无效反馈案例 | 待定义，无测量结果 |
| 资源与稳定性 | GPU 时间、生成 token、反馈输入 token、执行开销、超时/失败情况和训练曲线 | 待定义，无测量结果 |

### 实验记录模板

每次实际运行单独追加一条；计划 ID 不等于已经发生的 run。当前没有已确认的实验运行记录。

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

尚无已确认的运行阻塞；下列事项需要核实或在对应阶段确定。

| ID | 待确定事项 | 影响 | 处理时机 |
| --- | --- | --- | --- |
| Q1 | 起始日期、当前周次和实际完成情况 | 周计划与真实进度对齐 | 下一次进度更新 |
| Q2 | 各成员负责范围与已有产物位置 | 避免重复工作，补齐状态证据 | 下一次进度更新 |
| Q3 | GPU 资源、依赖兼容性与训练预算 | T2 环境和 T3/T4 训练可执行性 | 环境与训练准备时 |
| Q4 | Code-A1 参考版本、奖励函数和更新机制 | baseline 复现有效性 | 实现 baseline 前 |
| Q5 | 执行反馈来源、时机、变量选择及长度限制 | T4 实现和实验可解释性 | 接入反馈前 |
| Q6 | 固定预算的计量方式、指标及 seed 数量 | 对比公平性与结论可信度 | 正式对比前 |
| Q7 | MBPP 是否纳入最终评测 | 评测范围与工作量 | 固定评测协议时 |

## 9. Progress Log

按时间追加，记录事实、证据和下一步；计划中的实验不写成已完成结果。

| Date | Task / Sub-task | Event / Evidence | State / Outcome | Next step |
| --- | --- | --- | --- | --- |
| 2026-10-30 | T0.1 | 阅读 Adversarial Co-Training .pdf 全部 3 页，提取目标、配置与 7 周计划 | `DONE`：研究目标和原定计划已整理 | 核实实际进度 |
| 2026-10-30 | T0 | 检查当前仓库，创建 progress.md，补充任务拆分、验收条件、建议实验矩阵和日志模板 | `IN_PROGRESS`：tracker 已建立；研发任务保留 `UNKNOWN` | 补充当前周次、负责人及实际产物 |

### 后续更新规则

- 每次完成工作后，更新对应 sub-task 的 `Current state`、证据和 `Next step`，同步父 task 的状态。
- 标为 `DONE` 时附上可检查的产物；标为 `BLOCKED` 时写明原因和解除条件。
- 每次训练或评测追加实验记录，保留失败和中断记录，不覆盖历史结果。
- 修改目标、范围、预算或周计划时，在 log 中注明日期、原因与影响，并更新本文件顶部日期。
