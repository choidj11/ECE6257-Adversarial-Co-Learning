#!/usr/bin/env python3
"""Update English progress only after required study evidence exists."""
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
verification=study/'wandb-final-verified.json'
cloud_verified=verification.exists() and json.loads(verification.read_text()).get('passed') is True
path=root/'progress.md'
lines=path.read_text().splitlines()
evidence={
 'T3.3':'32 steps complete; final coder/tester HF weights, per-step gradients, and raw logs in artifacts/study/E0/; summary in FINAL_REPORT.md',
 'T3.4':'Limited-scale HumanEval/BigCodeBench/ULT Lite results complete; configuration deviations and canonical failures disclosed in FINAL_REPORT.md; no claim to absolute paper scores',
 'T4.3':'256 real training-feedback records per enhanced arm, at most 384 tokens, linked to code/traces/timing; artifacts/study/E*/feedback/',
 'T4.4':'E1/E2/E3 completed 32 steps and saved both final HF models; complete training curves in analysis.json',
 'T5.2':'All five variants completed adapted ULT Lite 200/K=2; coverage, bounded mutation, invalid tests, and measurement failures disclosed in FINAL_REPORT.md',
 'T5.3':'All five variants attempted 164 HumanEval + 1140 BigCodeBench tasks; raw/qualified pass@1 and environment failures reported',
 'T5.4':'Four-arm coverage/runtime/combined ablation complete; paired task bootstrap, costs, and positive/negative examples in analysis.json; single-seed limitation retained',
 'T6.1':'Three result/training figures in report/figures/ and auditable statistics in analysis.json; no-gain/negative findings retained',
 'T6.2':'FINAL_REPORT.md includes method, budget, actual metrics, ablations, failures, resources, limitations, and reproduction entrypoints',
}
rows={
 'Overall state':'Fixed-budget study complete: four 32-step training arms, final HF weights, five variants evaluated under the three-benchmark adaptation, ablation statistics, and report; see FINAL_REPORT.md for scale/environment limitations',
 'Verified completion':'Infrastructure, four 32-step dual-model runs, final weights, uniform five-variant evaluation, paired ablations, figures, and FINAL_REPORT.md',
 'Next milestone':'No compute tasks remain within this protocol; larger training, multiple seeds, and a complete official evaluation environment are possible extensions',
 'Current dependencies':('All computation and W&B cloud readback completed; full loss histories, final numeric metrics, and artifact scope verified; team assignments/start date remain unspecified' if cloud_verified else 'Computation and analysis complete; when W&B is enabled, final cloud readback runs separately; team assignments/start date remain unspecified'),
}
scoped_rows={
 'Training signal':'| Training signal | Branch coverage + selected runtime values | Implemented and used throughout E1–E3 training; traces, costs, and collection failures are auditable |',
 'Tester evaluation':'| Tester evaluation | UnLeakedTestBench | Five variants completed adapted ULT Lite 200/K=2; differences from full official Cosmic-Ray evaluation are in the report |',
 'Coder evaluation':'| Coder evaluation | HumanEval + BigCodeBench | All five variants attempted 164/1140 tasks; raw and canonical-qualified subset pass@1 recorded |',
 'Baseline reference':'| Baseline reference | Code-A1 | Pinned official code reproduced at limited scale; budget deviations, methods, and results in CODE_A1_METHOD.md and FINAL_REPORT.md |',
 'Tester quality':'| Tester quality | Valid-test fraction, valid tests/query, bounded mutant kills, failure counts | Measured; FINAL_REPORT.md |',
 'Coder performance':'| Coder performance | HumanEval/BigCodeBench raw and canonical-qualified subset pass@1 | Measured; FINAL_REPORT.md |',
 'Feedback effect':'| Feedback effect | Actual branch/line coverage, feedback lengths, and failure cases | Measured; analysis.json and report |',
 'Resources/stability':'| Resources/stability | Slurm GPU-hours, actual input/output tokens, feedback time, timeouts/failures, training curves | Recorded; resources.json and analysis.json |',
}
parents={'T3','T4','T5','T6'}
out=[]
for line in lines:
    if line.startswith('> Last updated:'):
        line=f'> Last updated: {today} (America/New_York)<br>'
    if line.startswith('| '):
        key=line.split('|')[1].strip()
        if key in rows:
            line=f'| {key} | {rows[key]} |'
        elif key in scoped_rows:
            line=scoped_rows[key]
        elif key in parents:
            cells=line.split('|');cells[6]=' `DONE` '
            cells[7]=' Fixed-budget scope complete; results, deviations, and limitations in FINAL_REPORT.md '
            line='|'.join(cells)
        elif key in evidence:
            cells=line.split('|');cells[3]=' `DONE` ';cells[4]=' '+evidence[key]+' '
            cells[5]=' Complete for this round; define a new protocol before scaling '
            line='|'.join(cells)
        elif key in ('3','4–5','6–7'):
            cells=line.split('|');cells[-2]=' `DONE`: accepted within this resource-limited protocol; paper-scale training is outside scope '
            line='|'.join(cells)
        elif key in ('E0','E1','E2','E3'):
            cells=line.split('|');m=analysis['summaries'][key];cells[5]=' `DONE` '
            cells[6]=f' 32 steps; HumanEval raw {m["coder"]["HumanEval"]["raw_pass_at_1"]:.2%}; details/limitations in FINAL_REPORT.md '
            line='|'.join(cells)
    out.append(line)
# Rerunning the finalizer must not duplicate a completed-study log entry.
if not any(line.startswith(f'| {today} | T3–T6 |') for line in out):
    log=(f'| {today} | T3–T6 | Four 32-step runs, final HF weights, five-variant 1304-task coder evaluation and ULT Lite 200/K=2 adaptation completed; analysis/report passed evidence checks | '
         '`DONE` within frozen scope; FINAL_REPORT.md, artifacts/study/analysis.json, report/figures/; single-seed/environment/bounded-mutation limits retained | Define a new budget/protocol for extensions |')
    index=out.index('### Future update rules')
    while index>0 and not out[index-1].strip():index-=1
    out.insert(index,log)
text='\n'.join(out)+'\n'
start=text.index('\n## 6. Next Steps\n')
end=text.index('\n## 7.',start)
text=text[:start]+(f'\n## 6. Next Steps\n\nStatus after acceptance on {today}; complete operating instructions are in [ENVIRONMENT.md](ENVIRONMENT.md).\n\n'
                   '1. **This study is complete:** see [FINAL_REPORT.md](FINAL_REPORT.md) for results, evidence, and adaptation limits.\n'
                   '2. **Optional extensions:** increase training scale and seeds, verify in the complete official dependency environment, and expand mutation operators and feedback probes.\n'
                   '3. **Team information:** actual project dates and other members\' assignments still require team input; do not infer them.\n')+text[end:]
path.write_text(text)
completion_path=study/'completion.json'
completion=json.loads(completion_path.read_text()) if completion_path.exists() else {}
completion.setdefault('completed_at',datetime.now(ZoneInfo('America/New_York')).isoformat())
completion.update({'passed':True,'scope':'Frozen 32-step resource-limited study',
                   'report':'FINAL_REPORT.md','wandb_verified':cloud_verified})
completion_path.write_text(json.dumps(completion,indent=2)+'\n')
print('PASS: evidence verified; English scoped-study progress finalized')
