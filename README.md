# Adversarial Co-Training for Agentic Coding

复现 Code-A1 的 coder–tester 对抗协同训练，并检验 branch coverage 和 selected runtime values 对测试质量与代码生成表现的影响。

- [进度与 7 周安排](progress.md)
- [本轮实验结果与报告](FINAL_REPORT.md)（32 steps、单 seed 的受限规模实验）
- [环境、数据、Slurm 与运行说明](ENVIRONMENT.md)
- [W&B 训练与评测指标](https://wandb.ai/mushroom_yeren/6257-adversarial-co-training/workspace)（私有项目）

在项目根目录激活训练环境：

```bash
source scripts/activate.sh rl
python scripts/check_quota.py
```

第 1–2 周的运行验证通过 Slurm 提交：

```bash
sbatch scripts/gpu_smoke.sbatch       # coe-gpu：单 GPU 推理
sbatch scripts/sandbox_smoke.sbatch   # ice-cpu：Python 沙箱与隔离检查
sbatch scripts/benchmark_smoke.sbatch # coe-gpu：三个 benchmark 各一个样例
sbatch scripts/week12_smoke.sbatch    # coe-gpu：两个 GPU，最小协同训练更新
```

脚本默认使用 `account=ece`、`qos=coe-ice`，从项目根目录提交。日志与检查报告在 `artifacts/setup/`，本地训练配置由 `scripts/prepare_week12.py` 生成到 `artifacts/week12/`。运行沙箱前需已有 `.venv-sandbox` 和固定版本的上游代码，重建入口为 `bash scripts/setup_sandbox.sh`。

这些是小规模运行验证，不代表正式 benchmark 分数或完成 baseline 复现。Scratch 的 300 GiB 和 100 万文件配额由用户的所有 Scratch 目录共享；新增任务先检查配额，训练验证不保存 checkpoint。

本轮固定预算实验已完成：E0–E3 各 32-step 双模型训练、五模型统一评测、消融统计、图表与 W&B 数值核验。增强组的主要配对区间均包含 0，尚无稳定提升证据。范围与偏差见 [实验协议](EXPERIMENT_PROTOCOL.md)、[Code-A1 方法核对](CODE_A1_METHOD.md) 和 [结果报告](FINAL_REPORT.md)；实际任务映射在 `artifacts/study/jobs.json`。

在已准备环境与数据的全新实验目录中，可用 `python scripts/submit_study.py` 一次性提交依赖图。已有 `jobs.json` 时脚本会拒绝重复提交；重跑须先归档本轮产物。W&B 指标监控见 [环境说明](ENVIRONMENT.md)。
