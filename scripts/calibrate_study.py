#!/usr/bin/env python3
"""Validate all reference harnesses before reporting environment-qualified scores."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from study_evaluation import ROOT, STUDY, read_tasks, coder_test, execute

destination=STUDY/'reference-harnesses.jsonl'
done={}
if destination.exists():
    done={r['task_id']:r for r in map(json.loads,destination.read_text().splitlines())}
tasks=[r for r in read_tasks() if r['task_id'] not in done]


def check(row):
    reference=row['prompt']+row['canonical_solution'] if row['benchmark']=='HumanEval' else row['complete_prompt']+row['canonical_solution']
    return {'benchmark':row['benchmark'],'task_id':row['task_id'],
            'execution':execute(coder_test(row,reference))}


with destination.open('a') as output,ThreadPoolExecutor(max_workers=4) as executor:
    for result in executor.map(check,tasks):
        output.write(json.dumps(result)+'\n')
        output.flush()
        done[result['task_id']]=result
        if len(done)%50==0:
            print(f'Reference harnesses: {len(done)}/1304',flush=True)
summary={'completed_at':datetime.now(timezone.utc).isoformat(),'timeout_seconds':120,
         'runtime':'.venv-rl in Bubblewrap','benchmarks':{}}
for benchmark in ('HumanEval','BigCodeBench'):
    group=[r for r in done.values() if r['benchmark']==benchmark]
    summary['benchmarks'][benchmark]={'total':len(group),'reference_passed':sum(r['execution']['passed'] for r in group),
                                      'failures':[r['task_id'] for r in group if not r['execution']['passed']]}
(STUDY/'reference-harnesses-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
