# Code-A1 实现核对

核对版本：ZJU-REAL/Code-A1 commit `be86d9315bab10a97d006aa25eb84204b25aa129`。代码入口为 `external/Code-A1/Code-A1/code/rl/main_a1.py`，本地对照配置见 `artifacts/study/E0/config.yaml`。来源为[官方仓库](https://github.com/ZJU-REAL/Code-A1)。

## 实际训练路径

1. `main_a1.py` 创建 coder/tester 两个资源池，分别初始化 actor/rollout 与 reference policy；继承官方定制 verl 的 FSDP2 worker。
2. `trainer_a1.py::_dual_model_rollout` 先为每题生成多份代码，再调用 `_prepare_model_b_batch_phase2`，将题意与每份候选代码交给 tester；tester 生成断言。
3. `reward.py` 在 Sandbox Fusion 执行接口中用参考实现检查测试有效性，再检查候选代码在有效测试及 Mistake Book 历史测试上的表现。
4. 计算旧策略和参考策略的 log probability，构造 GRPO advantage，以官方 PPO clipped loss 和 KL loss 更新两个 actor。
5. 保存 rollout 与 Mistake Book；本轮只在最终 step 保存两个模型的 HF 推理权重，控制磁盘占用。

## 奖励

当前训练数据 `data_source=kodcode_light`，使用注册为 `acg_llm` 的奖励类，`alpha=0.5`。在有效候选代码上：

- coder 奖励是新测试通过率；如果同时存在历史攻击测试，则与历史测试通过率取平均。
- tester 的有效性分量为 `max(0, min(1, 有效测试数/8, 1-无效测试数/8))`。
- 无历史攻击测试时，攻击分量为候选代码在新测试上的失败数/8；存在历史攻击测试时按官方实现与历史攻击失败率组合。
- tester 奖励为有效性与攻击分量按 `alpha` 加权；非可打印输出另有惩罚。无效代码不会获得正常有效性/攻击奖励。

trainer 对每份代码对应的多个 tester 奖励作聚合，并保留方差较高的 tester 组用于更新。Mistake Book 记录曾使候选代码失败的有效测试，供后续轮次重放。不能将该实现简单描述为“对官方测试的通过率做 RL”。

## 本轮偏差与保留项

保留奖励、Mistake Book、FSDP2、GRPO、双模型更新、KL 系数、学习率与基础模型。E0 直接运行上游 entrypoint；E1–E3 仅通过本地 subclass 扩展 tester prompt，不修改上游源码。

受资源约束，训练由上游配置的 3 epochs、batch 256、每方 n=8、每方 4 GPU，改为 128 个固定问题、32 steps、batch 4、每方 n=2、每方 1 GPU；输出上限由 4,096 改为 1,024 tokens。正式结论限于本轮固定预算，不声称复现论文绝对分数。

## 验收依据

第 1–2 周的作业 6888089 已跑通完整更新，两个 actor 的梯度范数分别为 2.010、2.977。新增执行反馈在 `artifacts/setup/feedback-check.json` 中验证了部分/完整分支覆盖与返回时局部变量；`study-evaluation-check.json` 验证了错误断言剔除、已知变异体检出以及完整分支覆盖。后续 32-step 训练与评测结果以对应作业、checkpoint、逐题 JSONL 和最终报告为准。
