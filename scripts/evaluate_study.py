#!/usr/bin/env python3
"""Evaluate frozen generations using isolated tests, coverage, bounded mutants."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from study_evaluation import ROOT, STUDY, read_tasks, extract_code, coder_test, execute, mutations

parser=argparse.ArgumentParser()
parser.add_argument('variant',choices=['base','E0','E1','E2','E3'],nargs='?')
parser.add_argument('--run-dir',type=Path,help='Evaluate an inference-only directory with selection.json')
parser.add_argument('--label',default='LFM3-1.7B')
parser.add_argument('--stream-coder',action='store_true',help='Score flushed coder outputs while generation continues')
args=parser.parse_args()
assert args.variant or args.run_dir,'Provide a study variant or an inference directory.'
directory=args.run_dir if args.run_dir else STUDY/args.variant
label=args.label if args.run_dir else args.variant
selection=json.loads((directory/'selection.json').read_text()) if args.run_dir else None
coder_tasks=[r for r in read_tasks() if selection is None or r['task_id'] in selection[r['benchmark']]]
if not args.stream_coder:assert (directory/'coder-generation-meta.json').exists()
assert (STUDY/'reference-harnesses-summary.json').exists()
references={r['task_id']:r for r in map(json.loads,(STUDY/'reference-harnesses.jsonl').read_text().splitlines())}
def read_generations():
    path=directory/'coder-generations.jsonl'
    lines=path.read_text().splitlines() if path.exists() else []
    rows=[]
    for index,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The writer may be between a JSON record and its next flush.
            if not args.stream_coder or index!=len(lines)-1:raise
    result={r['task_id']:r for r in rows}
    assert len(result)==len(rows),'Duplicate coder generation records.'
    assert set(result)<={r['task_id'] for r in coder_tasks},'Unexpected coder outputs.'
    return result


generations=read_generations()
if not args.stream_coder:assert len(generations)==len(coder_tasks),'Missing coder outputs.'
destination=directory/'coder-results.jsonl'
done={r['task_id']:r for r in map(json.loads,destination.read_text().splitlines())} if destination.exists() else {}
assert set(done)<={r['task_id'] for r in coder_tasks},'Unexpected cached coder results.'


def evaluate_coder(row):
    generation=generations[row['task_id']]
    code=extract_code(generation['output'])
    return {'benchmark':row['benchmark'],'task_id':row['task_id'],
            'reference_passed':references[row['task_id']]['execution']['passed'],
            'execution':execute(coder_test(row,code)),
            'prompt_tokens':generation.get('prompt_tokens',0),
            'generated_tokens':generation.get('generated_tokens',0)}


with destination.open('a') as output,ThreadPoolExecutor(max_workers=4) as executor:
    generation_deadline=time.monotonic()+3300
    while True:
        for record in executor.map(evaluate_coder,[r for r in coder_tasks if r['task_id'] not in done and r['task_id'] in generations]):
            output.write(json.dumps(record)+'\n'); output.flush()
            done[record['task_id']]=record
            if len(done)%50==0:
                print(f'{label} coder evaluated {len(done)}/{len(coder_tasks)}',flush=True)
        if len(done)==len(coder_tasks) and (directory/'coder-generation-meta.json').exists():break
        assert args.stream_coder,'Missing coder results.'
        if time.monotonic()>generation_deadline:raise TimeoutError('Coder generation/scoring did not finish within 55 minutes')
        time.sleep(10)
        generations=read_generations()

# Completed coder outputs can be scored while the GPU finishes tester generation.
# The completion marker is written only after all tester outputs are flushed.
deadline=time.monotonic()+1800
while not (directory/'tester-generation-meta.json').exists():
    if time.monotonic()>deadline:
        raise TimeoutError('Tester generation did not finish within 30 minutes after coder scoring')
    print(f'{label}: coder scoring complete; waiting for tester generation',flush=True)
    time.sleep(30)
ult=json.loads((ROOT/'data/UnLeakedTestBench/ULT_Lite.jsonl').read_text())
if selection is not None:ult=[r for r in ult if str(r['task_id']) in selection['ULT_Lite']]
test_outputs={}
for record in map(json.loads,(directory/'tester-generations.jsonl').read_text().splitlines()):
    test_outputs.setdefault(record['task_id'],[]).append(record)
assert set(test_outputs)=={str(r['task_id']) for r in ult},'Missing or unexpected tester outputs.'
assert all(sorted(r['round'] for r in rows)==[1,2] for rows in test_outputs.values())
profile=Path(__file__).with_name('profile_probe.py').read_text()
measurement=Path(__file__).with_name('ult_measure.py').read_text()
destination=directory/'tester-results.jsonl'
tester_done={r['task_id']:r for r in map(json.loads,destination.read_text().splitlines())} if destination.exists() else {}
assert set(tester_done)<=set(test_outputs),'Unexpected cached tester results.'


def evaluate_tester(row):
    outputs=sorted(test_outputs[str(row['task_id'])],key=lambda r:r['round'])
    candidates=[extract_code(output['output']) for output in outputs]
    try:
        mutants=mutations(row['code'])
        script=(f"SOURCE={row['code']!r}\nGENERATED={candidates!r}\nMUTANTS={mutants!r}\n"
                f"FUT={row['func_name']!r}\nPROFILE_SOURCE={profile!r}\n"+measurement)
        execution=execute(script)
        lines=execution.get('stdout','').splitlines()
        messages=[line[len('__A1_ULT__'):] for line in lines if line.startswith('__A1_ULT__')]
        metrics=json.loads(messages[-1]) if messages else None
    except Exception as error:
        execution={'passed':False,'error':type(error).__name__+': '+str(error)}
        metrics=None
    return {'task_id':str(row['task_id']),'execution':execution,'metrics':metrics,
            'generated_tokens':sum(o.get('generated_tokens',0) for o in outputs),
            'prompt_tokens':sum(o.get('prompt_tokens',0) for o in outputs)}


with destination.open('a') as output,ThreadPoolExecutor(max_workers=4) as executor:
    for record in executor.map(evaluate_tester,[r for r in ult if str(r['task_id']) not in tester_done]):
        output.write(json.dumps(record)+'\n'); output.flush()
        tester_done[record['task_id']]=record
        if len(tester_done)%20==0:
            print(f'{label} tester evaluated {len(tester_done)}/{len(ult)}',flush=True)

summary={'completed_at':datetime.now(timezone.utc).isoformat(),'variant':label,'coder':{},'tester':{}}
for benchmark in ('HumanEval','BigCodeBench'):
    rows=[r for r in done.values() if r['benchmark']==benchmark]
    eligible=[r for r in rows if r['reference_passed']]
    summary['coder'][benchmark]={'tasks':len(rows),'reference_qualified_tasks':len(eligible),
                                'passed_all_tasks':sum(r['execution']['passed'] for r in rows),
                                'passed_qualified_tasks':sum(r['execution']['passed'] for r in eligible),
                                'raw_pass_at_1':sum(r['execution']['passed'] for r in rows)/len(rows),
                                'qualified_pass_at_1':sum(r['execution']['passed'] for r in eligible)/len(eligible) if eligible else None}
measured=[r['metrics'] for r in tester_done.values() if r['metrics'] is not None]
valid=sum(m['valid_tests'] for m in measured)
submitted=sum(m['submitted_tests'] for m in measured)
mutants=sum(m['mutants'] for m in measured)
killed=sum(m['killed_mutants'] for m in measured)
summary['tester']={'tasks':len(tester_done),'measured_tasks':len(measured),'submitted_tests':submitted,
                   'valid_tests':valid,'valid_test_fraction':valid/submitted if submitted else 0,
                   'measurement_failures':len(tester_done)-len(measured),
                   'valid_tests_per_query':valid/(2*len(tester_done)),
                   'killed_mutants_per_query':killed/(2*len(tester_done)),
                   'mutants':mutants,'killed_mutants':killed,'bounded_mutation_score':killed/mutants if mutants else 0,
                   'mean_branch_coverage':sum((m['coverage'].get('branch_covered') or 0)/max(1,m['coverage'].get('branch_total') or 0) for m in measured)/len(tester_done),
                   'mean_line_coverage':sum((m['coverage'].get('line_covered') or 0)/max(1,m['coverage'].get('line_total') or 0) for m in measured)/len(tester_done),
                   'protocol':f'ULT Lite {len(ult)}, K=2, up to 8 assertions/query; max 4 deterministic AST mutants/FUT; not official Cosmic-Ray score.'}
summary['tokens']={role:{key:sum(r.get(key,0) for r in map(json.loads,(directory/f'{role}-generations.jsonl').read_text().splitlines()))
                         for key in ('prompt_tokens','generated_tokens')} for role in ('coder','tester')}
(directory/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
