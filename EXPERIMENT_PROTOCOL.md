# 固定预算对比协议

冻结时间：2026-11-08，正式模型评测结果产生前。机器可读配置为 `artifacts/study/protocol.json`，各组完整配置为 `artifacts/study/E*/config.yaml`。

## 研究范围

复现官方 Code-A1 coder–tester 的奖励、Mistake Book、GRPO 和双模型更新机制，比较向 tester 提供执行反馈的作用。本轮是资源受限规模的研究实验，不等于原论文完整训练预算复现，也不预设性能会提高。

| 设置 | 固定值 |
| --- | --- |
| 起点 | 本地 Qwen2.5-Coder-1.5B-Instruct，coder/tester 分别初始化 |
| 官方代码 | ZJU-REAL/Code-A1，commit `be86d9315bab10a97d006aa25eb84204b25aa129` |
| 数据 | 官方处理后的 KodCode hard 数据；9,688 条中 9,684 条满足 1,024 token prompt 上限；seed 42 随机选 128 条 |
| 训练 | 32 steps，batch size 4，一遍所选数据；每个问题 2 个 coder 输出，每份代码 2 个 tester 输出 |
| 优化 | 继承官方 FSDP2/GRPO；学习率 1e-6，KL 系数 0.001；单训练 seed 42 |
| 生成预算 | coder/tester 各最多 1,024 输出 tokens；四组完全相同 |
| 保存 | 只保存最后一步两个模型的 Hugging Face 推理权重；不另存优化器或重复分片 |
| 资源 | 每组 2 张 H100/H200、8 CPU、100 GiB 内存、单作业上限 1 小时；实际硬件与显存另记，持续检查 Scratch 余量 |

## 对比组与反馈

| 组 | 训练期间 tester 的新增输入 |
| --- | --- |
| E0 | 无；直接运行官方 entrypoint |
| E1 | 分支总数、已覆盖分支数、部分已覆盖/未覆盖的 `(source line, destination line)` |
| E2 | 函数返回时部分 locals 和 return value，关联函数名和源码行号 |
| E3 | E1 与 E2 的结合 |

对每份 coder 输出，使用同一样例前两个既有训练测试作为固定执行探针，不增加模型生成调用。通过独立沙箱执行候选代码；`coverage.py` 收集实际分支弧，独立 profiler 收集候选函数返回时的 locals。测试代码所在帧、参考实现和断言预期答案不进入反馈提示词。训练测试仍是有监督的 oracle，执行路径和返回值会间接揭示行为，这属于本研究要检验的反馈信号。

所有增强组执行相同的探针采集，按组屏蔽输入字段；每份代码新增输入最多 384 tokens，最多 16 个运行时快照，提示词最多选其中 4 个并进一步缩短。记录完整受限 trace、实际反馈文本、输入 token 数、执行耗时和采集失败。总体生成调用预算一致；反馈输入 tokens 和执行开销是额外成本，不能声称总计算成本相同。

采集后的反馈在当前轮 tester 生成前加入提示词。`feedback_trainer.py` 仅扩展 tester 输入，官方上游源文件、奖励及更新逻辑不修改。

## 评测

同时测未训练起点 base 和 E0–E3 最后一步的 coder/tester。所有版本使用完全相同的评测输入，不在测试阶段追加不同组的执行反馈，观察训练后能力的差异。

- **HumanEval**：全部 164 条，每题一个 greedy 输出；运行本地固定版本的原始测试，记录 pass@1。
- **BigCodeBench**：v0.1.4 全部 1,140 条 Instruct，每题一个 greedy 输出，执行完整 unittest。先对所有 canonical solution 校准同一沙箱环境；分别报告完整分母下原始通过率、canonical 通过子集上的通过率，以及被环境限制影响的任务列表。资源受限执行设置与官方 leaderboard 不完全一致，不能直接横向比较。
- **ULT Lite**：固定 200 条，K=2 轮，每轮最多生成 8 条独立 assert，第二轮看到上一轮输出。输出预算每轮最多 1,024 tokens，greedy；所有版本相同。当前接口沿用 Code-A1 的 assert 格式，属于 ULT 的适配评测。
- **测试指标**：有效 assert 数/提交数、固定 query 预算的有效测试数、平均行/分支覆盖率，以及每个函数最多 4 个固定 AST 比较/算术/布尔算子变异体的 kill rate。仅在参考函数上通过的测试参与覆盖率与 kill 判定。该有界变异指标不是官方完整 Cosmic-Ray mutation score，可能存在等价变异体。

每个模型保存原始输出、token 数、截断状态、逐题执行结果、超时/依赖错误。生成上限为 1,024 tokens，context 上限 8,192 tokens，超长输入不默默截断，作为失败单列。沙箱每个请求上限 120 秒、每进程地址空间 16 GiB；并发 4 个，禁止网络和项目文件访问。CPU 评测复用已有 `.venv-rl` 的第三方库，避免重复安装一套环境。

测量实现的分母约定：平均覆盖率按 200 个函数等权，测量失败记为 0；没有分支的函数在平均分支覆盖率中取 0，行覆盖率另报。mutation 总体分数按成功测量函数的变异体汇总；配对 bootstrap 则按每函数 kill rate 等权，没有变异体或测量失败时取 0，两者的差异不必相同。canonical 校准为单次运行，不能排除所有测试随机性或不稳定性。

## 分析与限制

报告各增强组相对 E0 的配对题目差异；对题目重采样给出 bootstrap 不确定性，不能用它替代多训练 seed。记录训练曲线、梯度、采样/反馈 token、GPU allocation 时间和执行耗时，保留负结果、无效输出与失败案例。未经有效运行支持的结论不写成已验证提升。

## 已核实的资源限制

`coe-gpu` 的单作业最长运行时间为 16 小时。`coe-ice` 的 `MaxTRESRunMinsPerUser=gres/gpu=1920` 表示运行中作业合计剩余预留 GPU 时间最多 32 GPU 小时，作业结束会释放；不是生命周期总时长额度。语义依据 [Slurm 官方 sacctmgr 文档](https://slurm.schedmd.com/sacctmgr.html)。本轮训练作业主动限制为 1 小时、生成作业 2 小时，CPU 评测单独提交，实际用量以 `sacct` 记录为准。训练预留最初为 2 小时，排队期间缩短为 1 小时以改善 backfill，不改变 32-step 预算。

CPU 评分固定并发 4 个单线程沙箱。base 评分申请 8 CPU/48 GiB，实测 16 分 31 秒；后四组在排队期间改为 4 CPU/48 GiB、1 小时上限以匹配空闲核心，四组均在 ice-cpu 运行。各候选的 120 秒超时一致，硬件及 CPU 申请差异需结合实际资源记录解释，不能将作业 walltime 直接作为算法效率结论。
