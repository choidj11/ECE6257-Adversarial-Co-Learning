#!/usr/bin/env python3
"""Build evidence tables, paired task intervals, figures, and the final report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/study'
variants=['base','E0','E1','E2','E3']
summaries={variant:json.loads((STUDY/variant/'metrics.json').read_text()) for variant in variants}
jobs=json.loads((STUDY/'jobs.json').read_text())
protocol=json.loads((STUDY/'protocol.json').read_text())
resources=json.loads((STUDY/'resources.json').read_text())
subprocess.run([sys.executable,str(ROOT/'scripts/audit_study_overlap.py')],check=True,cwd=ROOT)
overlap=json.loads((STUDY/'overlap-audit.json').read_text())
curves={}
for variant in variants[1:]:
    path=ROOT/f"artifacts/setup/study-train-{jobs['training'][variant]}.log"
    log=re.sub(r'\x1b\[[0-9;]*m','',path.read_text(errors='replace'))
    assert 'Dual model collaborative training completed!' in log,(variant,'training did not complete')
    records=[]
    for line in log.splitlines():
        match=re.search(r'step:(\d+) - (.*)',line)
        if not match:
            continue
        metrics={}
        for part in match.group(2).split(' - '):
            if ':' in part:
                key,value=part.rsplit(':',1)
                try: metrics[key]=float(value)
                except ValueError: pass
        metrics['step']=int(match.group(1))
        records.append(metrics)
    for role in ('a','b2'):
        role_records=[r for r in records if f'actor_{role}/grad_norm' in r]
        assert {r['step'] for r in role_records}==set(range(1,33)),(variant,role,'missing steps')
        assert all(np.isfinite(r[f'actor_{role}/grad_norm']) for r in role_records)
    curves[variant]=records


def load_results(variant,role):
    return {r['task_id']:r for r in map(json.loads,(STUDY/variant/f'{role}-results.jsonl').read_text().splitlines())}


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(42)
    means=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(2000)])
    return {'delta':float(values.mean()),'lower':float(np.quantile(means,.025)),
            'upper':float(np.quantile(means,.975)),'tasks':len(values)}


baseline_coder=load_results('E0','coder')
baseline_tester=load_results('E0','tester')
comparisons={}
examples={}
for variant in variants[2:]:
    coder=load_results(variant,'coder')
    tester=load_results(variant,'tester')
    comparisons[variant]={}
    examples[variant]={}
    for benchmark in ('HumanEval','BigCodeBench'):
        keys=sorted(key for key,row in baseline_coder.items() if row['benchmark']==benchmark and row['reference_passed'])
        differences=[int(coder[key]['execution']['passed'])-int(baseline_coder[key]['execution']['passed']) for key in keys]
        comparisons[variant][benchmark]=interval(differences)
        examples[variant][benchmark]={'improved':[key for key,d in zip(keys,differences) if d>0][:3],
                                       'regressed':[key for key,d in zip(keys,differences) if d<0][:3]}
    def tester_value(row,metric):
        m=row.get('metrics')
        if not m: return 0.0
        if metric=='valid_tests_per_query': return m['valid_tests']/2
        if metric=='bounded_mutation_score': return m['killed_mutants']/max(1,m['mutants'])
        c=m['coverage']
        return (c.get('branch_covered') or 0)/max(1,c.get('branch_total') or 0)
    for metric in ('valid_tests_per_query','bounded_mutation_score','branch_coverage'):
        comparisons[variant][metric]=interval([tester_value(tester[key],metric)-tester_value(baseline_tester[key],metric)
                                               for key in sorted(baseline_tester)])

feedback={}
for variant in variants[2:]:
    records=[]
    for path in sorted((STUDY/variant/'feedback').glob('*.jsonl')):
        records.extend(map(json.loads,path.read_text().splitlines()))
    assert len(records)==32*4*2,(variant,'missing feedback records',len(records))
    feedback[variant]={'records':len(records),'input_tokens':sum(r['input_tokens'] for r in records),
                       'execution_seconds_sum':sum(r['trace']['execution_seconds'] for r in records),
                       'collection_errors':sum(bool(r['trace'].get('error')) for r in records)}
generation_summary={}
for variant in variants:
    generation_summary[variant]={}
    for role,expected in (('coder',1304),('tester',400)):
        rows=[json.loads(line) for line in (STUDY/variant/f'{role}-generations.jsonl').read_text().splitlines()]
        assert len(rows)==expected,(variant,role,'incomplete generations')
        generation_summary[variant][role]={
            'outputs':len(rows),'length_limited_outputs':sum(r.get('finish_reason')=='length' for r in rows),
            'context_failures':sum(r.get('error')=='prompt_exceeds_context' for r in rows),
            'prompt_tokens':sum(r.get('prompt_tokens',0) for r in rows),
            'generated_tokens':sum(r.get('generated_tokens',0) for r in rows)}
analysis={'completed_at':datetime.now(timezone.utc).isoformat(),'summaries':summaries,
          'paired_task_bootstrap_95ci':comparisons,'examples':examples,'feedback_cost':feedback,
          'training_curves':curves,'training_seed_count':1,'bootstrap_replicates':2000,
          'resources':resources,'generation_summary':generation_summary,'overlap_audit':overlap}
(STUDY/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=ROOT/'report/figures'
FIG.mkdir(parents=True,exist_ok=True)
colors=['#7c8795','#245a81','#469989','#b58a42','#8862a0']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,benchmark in zip(axes,('HumanEval','BigCodeBench')):
    ax.bar(variants,[100*summaries[v]['coder'][benchmark]['qualified_pass_at_1'] for v in variants],color=colors)
    ax.set(title=benchmark+' (reference-qualified tasks)',ylabel='Pass@1 (%)',ylim=(0,100))
fig.savefig(FIG/'coder-results.png',dpi=180); plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,metric,label in zip(axes,('valid_tests_per_query','mean_branch_coverage','bounded_mutation_score'),
                           ('Valid assertions / query','Mean branch coverage','Bounded mutation kill rate')):
    factor=1 if metric=='valid_tests_per_query' else 100
    ax.bar(variants,[factor*summaries[v]['tester'][metric] for v in variants],color=colors)
    ax.set(title=label,ylabel='Count' if factor==1 else 'Percent')
fig.savefig(FIG/'tester-results.png',dpi=180); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for variant,color in zip(variants[1:],colors[1:]):
    for column,role in enumerate(('a','b2')):
        rs=[r for r in curves[variant] if f'reward_{role}/mean' in r]
        axes[0,column].plot([r['step'] for r in rs],[r[f'reward_{role}/mean'] for r in rs],label=variant,color=color)
        ls=[r for r in curves[variant] if f'actor_{role}/pg_loss' in r]
        axes[1,column].plot([r['step'] for r in ls],[r[f'actor_{role}/pg_loss'] for r in ls],label=variant,color=color)
for column,role in enumerate(('Coder','Tester')):
    axes[0,column].set(title=role+' training reward',xlabel='Training step',ylabel='Reward')
    axes[1,column].set(title=role+' policy loss (console precision)',xlabel='Training step',ylabel='Policy loss')
    axes[0,column].legend(); axes[1,column].legend()
fig.savefig(FIG/'training-curves.png',dpi=180); plt.close(fig)

reference=json.loads((STUDY/'reference-harnesses-summary.json').read_text())
lines=['# Adversarial Co-Training：固定预算实验报告','',
       '本报告比较未训练模型、Code-A1 baseline 和三组执行反馈改进。结论仅覆盖本轮 32-step、单 seed 的受限规模实验；不等同于论文完整预算复现或官方 leaderboard 分数。','',
       '## 目标与实现','',
       '检验向 tester 提供候选代码的分支覆盖率与返回时局部变量，是否在固定模型输出预算下改善测试质量和 coder 表现。四组共享基础模型、128 个训练问题、32 steps、batch 4、coder/tester 各 n=2、1,024 输出 token 上限。学习率 1e-6，KL 系数 0.001。',
       '', 'E0 直接运行官方 Code-A1；E1 提供 coverage，E2 提供 runtime values，E3 提供两者。增强组每份候选代码执行前两个固定训练测试作为探针，新增反馈最多 384 tokens；不把测试断言预期答案或参考实现交给 tester。反馈输入与执行是额外成本。',
       '', '完整协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，上游机制核对见 [CODE_A1_METHOD.md](CODE_A1_METHOD.md)。',
       '', f'训练子集 {overlap["training_rows"]} 条与三个评测集共 {overlap["evaluation_tasks"]} 条参考实现进行 AST 精确对照（去除 docstring、注释与格式，保留标识符及 imports），发现 {len(overlap["exact_normalized_source_matches"])} 个相同源码匹配；训练/评测未解析数为 {overlap["unparsed_training_sources"]}/{overlap["unparsed_evaluation_sources"]}。这不能排除改写、语义重复或基础模型预训练污染，详见 `artifacts/study/overlap-audit.json`。','',
       '## 训练结果与可复现性','',
       '四组均需具有 1–32 步两模型的有限梯度记录、最终 HF 权重和逐轮 rollout，报告脚本在这些证据缺失时退出。模型权重、数据和原始日志位于 Git 忽略的 `artifacts/`、`data/` 和 `models/` 目录，通过脚本复现。','',
       '| 组 | 新增反馈 | 训练作业 | 最终 step |','| --- | --- | --- | --- |']
for variant,description in zip(variants[1:],('无','Coverage','Runtime values','两者结合')):
    lines.append(f'| {variant} | {description} | {jobs["training"][variant]} | 32 |')
lines += ['', '![Training curves](report/figures/training-curves.png)','',
          '训练 reward 是当前 coder/tester 交互下的信号，分布会随两模型变化，不能单独作为独立 benchmark 改善的证据。Loss 为 GRPO policy loss，可为负数，其符号或绝对大小也不能直接判定模型优劣。本轮 console-only 曲线保留三位小数。','',
          '## Coder 评测','',
          '所有模型采用相同 greedy prompt、每题一个输出、最大 1,024 tokens。HumanEval 共 164 条，BigCodeBench v0.1.4 Instruct 共 1,140 条；每个候选执行同一版本的完整测试。先校准参考实现，分别列出全量原始通过率与参考实现通过子集上的通过率。环境不合格任务不被默认为模型失败证据。','',
          '| 模型 | HumanEval 原始 pass@1 | HumanEval 校准子集 | BigCodeBench 原始 pass@1 | BigCodeBench 校准子集 |',
          '| --- | --- | --- | --- | --- |']
for v in variants:
    he=summaries[v]['coder']['HumanEval']; bcb=summaries[v]['coder']['BigCodeBench']
    lines.append(f'| {v} | {he["raw_pass_at_1"]:.2%} | {he["qualified_pass_at_1"]:.2%} | {bcb["raw_pass_at_1"]:.2%} | {bcb["qualified_pass_at_1"]:.2%} |')
lines += ['',f'参考校准：HumanEval {reference["benchmarks"]["HumanEval"]["reference_passed"]}/164；BigCodeBench {reference["benchmarks"]["BigCodeBench"]["reference_passed"]}/1140。失败任务与原始 stderr 保存在 `artifacts/study/reference-harnesses.jsonl`。',
          '', '![Coder results](report/figures/coder-results.png)','',
          '## Tester 评测与消融','',
          '采用 ULT Lite 固定 200 个函数，K=2 轮，每轮最多 8 条 assert。第二轮看到第一轮输出；所有模型的测试阶段都不额外提供执行反馈，以比较训练后能力。只在参考函数上通过的测试参与 coverage 与 mutant kill。每函数最多 4 个固定 AST 比较/算术/布尔变异体，属于有界 mutation 指标，不是官方完整 Cosmic-Ray 分数。','',
          '| 模型 | 有效测试/提交数 | 有效测试/query | 平均行覆盖率 | 平均分支覆盖率 | 有界 mutation score | 执行测量失败函数 |',
          '| --- | --- | --- | --- | --- | --- | --- |']
for v in variants:
    t=summaries[v]['tester']
    lines.append(f'| {v} | {t["valid_tests"]}/{t["submitted_tests"]} | {t["valid_tests_per_query"]:.3f} | {t["mean_line_coverage"]:.2%} | {t["mean_branch_coverage"]:.2%} | {t["bounded_mutation_score"]:.2%} | {t["measurement_failures"]} |')
lines += ['', '每轮最多计 8 个顶层 assert；语法错误、重定义函数或没有顶层 assert 的整轮输出计为一次无效提交。有效测试/query 始终以固定 400 次 query 为分母。平均覆盖率按 200 个函数等权；测量失败函数取 0，没有分支的函数在分支覆盖率中也取 0，行覆盖率另报。测试有效率与 mutation 总体比率基于成功返回测量数据的函数，须结合失败函数数量阅读。逐函数配对分析对每个函数等权，测量失败或没有变异体的函数取 0；其 mutation 差异与上表按变异体汇总的比率之差不必相同。','',
          '![Tester results](report/figures/tester-results.png)','',
          '## 相对 baseline 的配对差异','',
          '下表对同一题目作配对差，使用 2,000 次按题 bootstrap，列出 95% 区间。它刻画题目抽样不确定性，不包含训练 seed 方差；四组共享同一训练 seed，且未作多重比较校正。','',
          '| 组 | 指标 | 差异 | 95% 区间 |','| --- | --- | --- | --- |']
for v in variants[2:]:
    for metric,d in comparisons[v].items():
        scale=1 if metric=='valid_tests_per_query' else 100
        unit='' if scale==1 else ' pp'
        lines.append(f'| {v} | {metric} | {d["delta"]*scale:+.3f}{unit} | [{d["lower"]*scale:+.3f}, {d["upper"]*scale:+.3f}]{unit} |')
lines += ['', '## 反馈成本与失败案例','',
          '| 组 | Trace 数 | 额外输入 tokens | 探针执行耗时合计（秒） | 采集错误 |',
          '| --- | --- | --- | --- | --- |']
for v,c in feedback.items():
    lines.append(f'| {v} | {c["records"]} | {c["input_tokens"]} | {c["execution_seconds_sum"]:.1f} | {c["collection_errors"]} |')
lines += ['', '评测生成预算实际用量（coder 1,304 次输出，tester 400 次输出；长度上限是生成的停止原因，可能产生不完整代码）：','',
          '| 模型 | Coder 输入/输出 tokens | Coder 达到长度上限 | Tester 输入/输出 tokens | Tester 达到长度上限 | 超长输入失败 |',
          '| --- | --- | --- | --- | --- | --- |']
for v in variants:
    c=generation_summary[v]['coder'];t=generation_summary[v]['tester']
    lines.append(f'| {v} | {c["prompt_tokens"]}/{c["generated_tokens"]} | {c["length_limited_outputs"]}/1304 | {t["prompt_tokens"]}/{t["generated_tokens"]} | {t["length_limited_outputs"]}/400 | {c["context_failures"]+t["context_failures"]} |')
quota=resources['scratch_quota']
lines += ['',f'实验后 Scratch 配额：{quota["used_gib"]:.1f}/{quota["limit_gib"]:.0f} GiB，剩余 {quota["free_gib"]:.1f} GiB；文件 {quota["used_files"]:,}/{quota["limit_files"]:,}。该配额由用户整个 Scratch 共享，运行前保护阈值为 20 GiB / 20,000 文件。']
lines += ['', '三个增强组的探针数量和采集设置一致，实际执行成本与反馈输入长度不同；执行耗时合计是请求耗时之和，不等于并发后的 walltime。逐条反馈和源码关联见 `artifacts/study/E*/feedback/`。原始 generation 文件记录真实 prompt/output token 数及截断状态。','',
          f'本轮 Slurm 已分配 GPU 用量合计 **{resources["gpu_hours"]:.3f} GPU-hours**，逐作业实际 GPU 类型、时长、退出状态见 `artifacts/study/resources.json`。E1 训练使用 H200，其余三组使用 H100；不同硬件的数值与生成差异未被单 seed 控制，不能将组间差异完全归因于反馈。base CPU 评分申请 8 核，后四组申请 4 核，均为 4 个沙箱并发；也不能把 walltime 差直接解释为算法收益。','',
          '代表性配对例子（先按 task id 排序选取，未只挑成功案例）：','']
for v,benchmarks in examples.items():
    for benchmark,example in benchmarks.items():
        lines.append(f'- {v} / {benchmark}：新增通过 {", ".join(example["improved"]) or "无"}；相对退步 {", ".join(example["regressed"]) or "无"}。')
lines += ['', '## 结论与局限','']
for v in variants[2:]:
    d=comparisons[v]['bounded_mutation_score']
    statement='区间跨越 0，当前证据不足以确认稳定改善。' if d['lower']<=0<=d['upper'] else ('区间高于 0，本轮题目抽样下呈正向差异。' if d['lower']>0 else '区间低于 0，本轮题目抽样下呈负向差异。')
    lines.append(f'{v} 的逐函数有界 mutation score 相对 E0 平均差异为 {d["delta"]*100:+.3f} 个百分点；{statement}单 seed、小训练集及受限生成长度限制了该结论的外推。\n')
lines += ['受限训练与论文配置有明显差异；完整 BigCodeBench 的部分依赖、网络或参考测试可能不适配本沙箱；ULT 采用 assert 接口与少量变异算子，存在等价变异体及分母选择限制。没有把未提升、截断、超时或测量失败隐藏为成功。后续应增加训练规模和 seed，扩大反馈探针与 mutation 算子，并在完整官方评测环境复核。','',
          '## 复现入口','',
          '```bash','source scripts/activate.sh rl','python scripts/check_quota.py',
          '# 仅限已准备环境与数据、尚无 jobs.json 的全新实验目录：',
          'python scripts/submit_study.py --wandb',
          '# --wandb 要求已有明确授权的 metrics 配置；不上传时省略。',
          '# 重新分析本轮已有完整结果：',
          'python scripts/record_study_resources.py','python scripts/analyze_study.py','```','',
          '完整作业映射：`artifacts/study/jobs.json`；完整统计与训练曲线：`artifacts/study/analysis.json`。重新运行请使用独立输出目录或先归档本轮产物，生成脚本会拒绝覆盖已有原始输出。','']
wandb_config_path=STUDY/'wandb-config.json'
if wandb_config_path.exists():
    monitoring=json.loads(wandb_config_path.read_text())
    lines += ['## W&B 监控','',f'项目：[6257-adversarial-co-training]({monitoring["url"]})。按用户指定范围，仅同步数值训练指标、配置、资源统计和评测汇总；不上传原始题目、模型输出、逐条日志、数据集或权重。',
              '', 'Loss 历史位于 `actor_a/pg_loss`、`actor_b2/pg_loss`；最终评测汇总在 `eval/` 和 `evaluation/`。本轮较早的 console-only 数据只有三位小数，精度限制保留在 run config 中；后续结构化 metrics stream 可记录完整精度。最终云端读回验收记录为 `artifacts/study/wandb-final-verified.json`。','']
(ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines))
print('PASS: completed evidence-backed report, figures and paired statistics')
