#!/usr/bin/env python3
"""Write an English inference-only report from completed local evidence."""
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/lfm3-evaluation'


def percent(value):
    return str((Decimal(str(value))*100).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))+'%'


protocol=json.loads((OUT/'protocol.json').read_text())
small=json.loads((OUT/'small/metrics.json').read_text())
full=json.loads((OUT/'full/metrics.json').read_text())
gate=json.loads((OUT/'small/gate.json').read_text())
assert gate['passed']
assert {key:value['tasks'] for key,value in full['coder'].items()}=={'HumanEval':164,'BigCodeBench':1140}
assert full['tester']['tasks']==200
jobs=json.loads((OUT/'jobs.json').read_text())
resources=[]
allocations=[(stage,job,True) for stage,job in jobs['generation'].items()]
allocations += [(row.get('stage','inference')+' throughput attempt',row['job_id'],False)
                for row in jobs['attempts'] if row.get('gpu_job')]
for stage,job,must_complete in allocations:
    rows=subprocess.check_output(['sacct','-j',str(job),'--noheader','--parsable2',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],text=True).splitlines()
    row=next(row for row in rows if row.split('|')[0]==str(job)).split('|')
    if must_complete:assert row[1]=='COMPLETED' and row[2]=='0:0',(stage,row)
    else:assert row[1].startswith(('CANCELLED','FAILED','TIMEOUT','COMPLETED')),(stage,row)
    resources.append({'stage':stage,'job_id':job,'state':row[1],'seconds':int(row[3]),
                      'gpu_hours':int(row[3])/3600,'allocation':row[4]})
cleanup=json.loads((ROOT/'artifacts/setup/qwen-weight-cleanup.json').read_text())
diagnostics={}
for stage in ('small','full'):
    diagnostics[stage]={}
    for role in ('coder','tester'):
        rows=[json.loads(line) for line in (OUT/stage/f'{role}-generations.jsonl').read_text().splitlines()]
        diagnostics[stage][role]={'outputs':len(rows),
            'length_limited':sum(r['finish_reason']=='length' for r in rows),
            'empty_outputs':sum(not r['output'].strip() for r in rows)}
result={'passed':True,'completed_at':datetime.now(timezone.utc).isoformat(),
        'model_id':protocol['model_id'],'revision':protocol['revision'],
        'mode':'inference_only','training_steps':0,'wandb_mode':'disabled',
        'small_gate':gate,'small':small,'full':full,'generation_resources':resources,
        'gpu_hours':sum(r['gpu_hours'] for r in resources),
        'qwen_removed_gib':cleanup['removed_gib'],'generation_diagnostics':diagnostics}
(OUT/'report-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
lines=[
    '# LFM3-1.7B Inference Evaluation','',
    f"Model: `{protocol['model_id']}` at revision `{protocol['revision']}`.",
    '',
    'The [aggregate results and protocol](report/lfm3/summary.json) are available as JSON. Reproduction scripts and pinned inference dependencies are included in this repository; raw generations, per-task records, datasets, and weights remain local.','',
    'The original checkpoint serves both roles. There is no fine-tuning, RL, parameter update, candidate reranking, or inference-time code repair. Coder and tester are scored on their respective benchmarks; the result is not a jointly trained system score.','',
    'HumanEval uses the original OpenAI test set, not HumanEval+. BigCodeBench uses the full v0.1.4 Instruct variant. ULT Lite uses the existing 200-task bounded adaptation.','',
    '## Coder results','',
    '| Stage | Benchmark | Passed / total | Raw pass@1 | Reference-qualified pass@1 |',
    '| --- | --- | ---: | ---: | ---: |',
]
for stage,metrics in [('128-task feasibility',small),('Full',full)]:
    for benchmark,row in metrics['coder'].items():
        lines.append(f"| {stage} | {benchmark} | {row['passed_all_tasks']}/{row['tasks']} | {percent(row['raw_pass_at_1'])} | {percent(row['qualified_pass_at_1'])} ({row['reference_qualified_tasks']} tasks) |")
lines+=['','The feasibility subset contains 32 HumanEval, 64 BigCodeBench, and 32 ULT Lite tasks. Full evaluation includes all 164, 1140, and 200 tasks respectively, reusing feasibility outputs/results. Raw scores retain every task in the denominator. Qualified scores exclude tasks whose reference solution failed in the unchanged sandbox calibration.','',
        '## Tester results','',
        '| Stage | Tasks measured / total | Valid assertions / submitted | Valid / query | Mean branch coverage | Bounded mutation score |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
for stage,metrics in [('128-task feasibility',small),('Full',full)]:
    row=metrics['tester']
    lines.append(f"| {stage} | {row['measured_tasks']}/{row['tasks']} | {row['valid_tests']}/{row['submitted_tests']} | {row['valid_tests_per_query']:.4f} | {percent(row['mean_branch_coverage'])} | {percent(row['bounded_mutation_score'])} ({row['killed_mutants']}/{row['mutants']}) |")
lines+=['','Tester evaluation makes two queries per function, requesting up to eight assertions per query. Incorrect assertions are rejected against the reference function. Coverage uses valid assertions; fault detection uses at most four deterministic AST mutants per function. This is the existing bounded ULT Lite adaptation, not the official full Cosmic Ray score. Measurement failures are retained in the report rather than silently dropped.','',
        f"Full evaluation attempted all 200 functions; {full['tester']['measurement_failures']} whole-function measurements reached the 120-second execution limit. Assertion and mutation totals use the {full['tester']['measured_tasks']} measured functions. Coverage means and per-query counts keep all 200 functions in their denominators, with zero contribution from failed measurements. Full assertion validity is {percent(full['tester']['valid_test_fraction'])} ({full['tester']['valid_tests']}/{full['tester']['submitted_tests']}).",'',
        '## Protocol and resources','',
        f"- Seed {protocol['seed']}; BF16; Transformers/SDPA; context cap {protocol['context_tokens']} tokens; output cap {protocol['max_new_tokens']} tokens.",
        f"- Sampling: temperature {protocol['temperature']}, top_k {protocol['top_k']}, top_p {protocol['top_p']}, repetition penalty {protocol['repetition_penalty']}; thinking requested off through the official chat template.",
        f"- Full-run responses reaching the output cap: coder {diagnostics['full']['coder']['length_limited']}/{diagnostics['full']['coder']['outputs']}; tester {diagnostics['full']['tester']['length_limited']}/{diagnostics['full']['tester']['outputs']}. Truncations remain in the score denominators.",
        '- One coder response per task. A single sampled response is reported as pass@1; repeated-sampling uncertainty is not estimated.',
        f"- Generation allocation: {result['gpu_hours']:.4f} GPU-hours; job details in `artifacts/lfm3-evaluation/report-evidence.json`.",
        '- Feasibility started at batch 8 and resumed at batch 32 after resource monitoring. A full-run batch-64 attempt was stopped after slow initial progress, and full inference resumed at the verified batch 32 while retaining outputs. This was not a controlled steady-state throughput comparison. Cancelled GPU allocation time is included. Sampling randomness depends on batching and process restarts; retained outputs and the allocation history are recorded locally.',
        '- W&B reporting is disabled for this evaluation. The report and aggregate metrics are versioned in this repository; raw generations and per-task results stay local under ignored `artifacts/lfm3-evaluation/`.',
        f"- User-authorized Qwen cleanup removed {cleanup['removed_gib']:.2f} GiB by file size, including the original checkpoint cache and eight fine-tuned coder/tester checkpoint trees. Historical metrics, outputs, and reports remain available; trained Qwen weights no longer exist locally.",
        '- The earlier Qwen study used different sampling/output settings. Its results are historical context, not a controlled model comparison.','',
        '## Local evidence','',
        '- `protocol.json`, `model-revision.json`, and `custom-code-sha256.json` under `artifacts/lfm3-evaluation/`.',
        '- `small/metrics.json`, `small/gate.json`, `full/metrics.json`, `report-evidence.json`, and `final-verification.json` in that directory.',
        '- Weight removal manifest: `artifacts/setup/qwen-weight-cleanup.json`.','',
]
(ROOT/'LFM3_EVALUATION.md').write_text('\n'.join(lines))
print(json.dumps({'passed':True,'gpu_hours':result['gpu_hours'],'coder':full['coder'],'tester':full['tester']},indent=2))
