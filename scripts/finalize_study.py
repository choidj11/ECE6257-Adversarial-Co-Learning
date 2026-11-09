#!/usr/bin/env python3
"""Update progress only after every required study artifact actually exists."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1]
study=root/'artifacts/study'
analysis=json.loads((study/'analysis.json').read_text())
assert (root/'FINAL_REPORT.md').is_file()
assert all((root/'report/figures'/name).is_file() for name in
           ('coder-results.png','tester-results.png','training-curves.png'))
for variant in ('base','E0','E1','E2','E3'):
    m=analysis['summaries'][variant]
    assert m['coder']['HumanEval']['tasks']==164
    assert m['coder']['BigCodeBench']['tasks']==1140
    assert m['tester']['tasks']==200
for variant in ('E0','E1','E2','E3'):
    for role in ('coder','tester'):
        checkpoints=list((study/variant/role).glob('*/global_step_32/actor_rollout_ref/huggingface'))
        assert len(checkpoints)==1,(variant,role,checkpoints)
        assert list(checkpoints[0].glob('*.safetensors'))
for variant in ('E1','E2','E3'):
    for file in (study/variant/'feedback').glob('*.jsonl'):
        assert all(json.loads(line)['input_tokens']<=384 for line in file.read_text().splitlines())
today=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
path=root/'progress.md'
lines=path.read_text().splitlines()
evidence={
 'T3.3':'32 steps 完成；最终 coder/tester HF 权重、逐步梯度与原始日志保存在 artifacts/study/E0/；汇总见 FINAL_REPORT.md',
 'T3.4':'受限规模 baseline 的 HumanEval/BigCodeBench/ULT Lite 结果齐全；配置偏差和 canonical 校准失败均在 FINAL_REPORT.md 明示；未声称论文绝对分数复现',
 'T4.3':'三组各 256 条实际训练反馈记录，均不超过 384 tokens，关联源码、trace 和执行耗时；artifacts/study/E*/feedback/',
 'T4.4':'E1/E2/E3 各完成 32 steps 并保存两个模型的最终 HF 权重；analysis.json 含完整训练曲线',
 'T5.2':'五个模型在 ULT Lite 200 条、K=2 的适配评测完成；coverage、有界 mutation、无效测试和测量失败均在 FINAL_REPORT.md 单列',
 'T5.3':'五个模型各 164 HumanEval + 1140 BigCodeBench 全量尝试；报告 raw/参考合格子集 pass@1 和环境失败列表',
 'T5.4':'coverage/runtime/结合的四组消融完成；analysis.json 保存配对 task bootstrap、成本与正负案例；单 seed 限制保留',
 'T6.1':'report/figures/ 中三幅结果/训练图，analysis.json 中可追溯统计；结论包含无提升/负向情况',
 'T6.2':'FINAL_REPORT.md 已生成，包含方法、预算、实际指标、消融、失败案例、资源与限制及复现入口',
}
parents={'T3','T4','T5','T6'}
out=[]
for line in lines:
    if line.startswith('> 最后更新：'):
        line=f'> 最后更新：{today}（America/New_York）'
    if line.startswith('| Overall state |'):
        line='| Overall state | 本轮固定预算研究实验完成：四组各 32-step 训练、最终 HF 权重、五模型三 benchmark 适配评测、消融统计与报告；受限规模及环境限制见 FINAL_REPORT.md |'
    elif line.startswith('| 已确认完成 |'):
        line='| 已确认完成 | 基础设施、四组各 32-step 双模型训练、最终权重、五模型统一评测、配对消融统计、图表及 FINAL_REPORT.md |'
    elif line.startswith('| Next milestone |'):
        line='| Next milestone | 当前协议范围无剩余计算任务；扩展训练规模、多 seed 和完整官方评测环境属于后续扩展 |'
    elif line.startswith('| 当前依赖 |'):
        line='| 当前依赖 | 训练、生成、评测与分析已完成；W&B 最终云端读回由依赖作业另行核验，证据为 wandb-final-verified.json；团队分工/起始日期仍未提供 |'
    elif line.startswith('| Training signal |'):
        line='| Training signal | Branch coverage + selected runtime values | 已实现并用于 E1–E3 的完整 32-step 训练；trace、成本和采集失败可追溯 |'
    elif line.startswith('| Tester evaluation |'):
        line='| Tester evaluation | UnLeakedTestBench | 五个模型的 ULT Lite 200/K=2 assert 适配评测完成；指标与官方完整 Cosmic-Ray 范围的区别见报告 |'
    elif line.startswith('| Coder evaluation |'):
        line='| Coder evaluation | HumanEval + BigCodeBench | 五个模型各完成 164/1140 题全量尝试，记录 raw 与 canonical 合格子集 pass@1 |'
    elif line.startswith('| Baseline reference |'):
        line='| Baseline reference | Code-A1 | 官方固定 commit 的受限规模复现完成；预算差异、方法与实际结果见 CODE_A1_METHOD.md 和 FINAL_REPORT.md |'
    elif line.startswith('| Tester 质量 |'):
        line='| Tester 质量 | 有效测试比例、有效测试/query、有界 mutation kill、失败数量 | 已测量；见 FINAL_REPORT.md |'
    elif line.startswith('| Coder 表现 |'):
        line='| Coder 表现 | HumanEval / BigCodeBench raw 与 canonical 合格子集 pass@1 | 已测量；见 FINAL_REPORT.md |'
    elif line.startswith('| 反馈效果 |'):
        line='| 反馈效果 | 实际 branch/line coverage、反馈长度和失败案例 | 已测量；见 analysis.json 与报告 |'
    elif line.startswith('| 资源与稳定性 |'):
        line='| 资源与稳定性 | Slurm GPU-hours、真实输入/输出 tokens、反馈耗时、超时/失败、训练曲线 | 已记录；见 resources.json、analysis.json |'
    if line.startswith('| T'):
        cells=line.split('|')
        identifier=cells[1].strip()
        if identifier in parents:
            cells[6]=' `DONE` '
            cells[7]=' 本轮固定预算范围完成；结果、配置差异与限制见 FINAL_REPORT.md '
            line='|'.join(cells)
        elif identifier in evidence:
            cells[3]=' `DONE` '
            cells[4]=' '+evidence[identifier]+' '
            cells[5]=' 本轮已完成；需要扩大规模时另立实验协议 '
            line='|'.join(cells)
    if line.startswith('| 3 | Replicate') or line.startswith('| 4–5 |') or line.startswith('| 6–7 |'):
        cells=line.split('|');cells[-2]=' `DONE`：本轮受限规模协议验收完成；论文完整预算不在本轮范围 ';line='|'.join(cells)
    if any(line.startswith(f'| {v} |') for v in ('E0','E1','E2','E3')):
        cells=line.split('|');variant=cells[1].strip();m=analysis['summaries'][variant]
        cells[5]=' `DONE` '
        cells[6]=f' 32 steps；HumanEval raw {m["coder"]["HumanEval"]["raw_pass_at_1"]:.2%}；详细结果与限制见 FINAL_REPORT.md '
        line='|'.join(cells)
    out.append(line)
log=(f'| {today} | T3–T6 | 四组 32-step 训练、最终 HF 权重、五模型 1304 coder 题和 ULT Lite 200/K=2 适配评测全部产生结果；分析与报告通过证据检查 | '
     '`DONE`：本轮固定预算范围；FINAL_REPORT.md、artifacts/study/analysis.json、report/figures/；保留单 seed、环境失败与有界 mutation 的限制 | 扩展实验需另定预算与协议 |')
index=out.index('### 后续更新规则')
out.insert(index-1,log)
text='\n'.join(out)+'\n'
text=text.replace('当前只有第 1–2 周环境 smoke 记录，没有正式 E0–E3 对比实验结果；运行记录见下方日志及 artifacts/setup/。',
                  '本轮 E0–E3 固定预算对比实验已完成；原始运行、逐题结果和分析见 artifacts/study/ 与 FINAL_REPORT.md。')
text=text.replace('本轮 E0–E3 训练已经完成，完整对比仍需等待统一评测；运行记录见下方日志及 artifacts/study/。',
                  '本轮 E0–E3 固定预算对比实验已完成；原始运行、逐题结果和分析见 artifacts/study/ 与 FINAL_REPORT.md。')
start=text.index('1. **训练（T3/T4）**')
end=text.index('\n## 7.',start)
text=text[:start]+('1. **本轮已完成**：结果、证据与适配评测限制见 [FINAL_REPORT.md](FINAL_REPORT.md)。\n'
                   '2. **可选扩展**：增加训练规模与 seed，在完整官方依赖环境复核，扩大 mutation 算子与反馈探针。\n'
                   '3. **团队信息**：项目原始起止日期和其他成员分工仍待团队提供，不据此虚构状态。\n')+text[end:]
path.write_text(text)
(study/'completion.json').write_text(json.dumps({'completed_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),
                                               'passed':True,'scope':'Frozen 32-step resource-limited study',
                                               'report':'FINAL_REPORT.md'},indent=2)+'\n')
print('PASS: evidence verified; scoped study progress finalized')
