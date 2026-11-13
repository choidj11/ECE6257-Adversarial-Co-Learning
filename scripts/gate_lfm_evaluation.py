#!/usr/bin/env python3
"""Check inference feasibility before authorizing the larger evaluation allocation."""
import argparse
import ast
from datetime import datetime,timezone
import json
from pathlib import Path

from study_evaluation import extract_code

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/lfm3-evaluation'
parser=argparse.ArgumentParser()
parser.add_argument('--require-passed',action='store_true')
args=parser.parse_args()
directory=OUT/'small'
if args.require_passed:
    assert json.loads((directory/'gate.json').read_text())['passed']
else:
    metrics=json.loads((directory/'metrics.json').read_text())
    coder=[json.loads(line) for line in (directory/'coder-generations.jsonl').read_text().splitlines()]
    tester=[json.loads(line) for line in (directory/'tester-generations.jsonl').read_text().splitlines()]
    coder_results=[json.loads(line) for line in (directory/'coder-results.jsonl').read_text().splitlines()]
    def contains_node(row,node_type):
        try:return any(isinstance(node,node_type) for node in ast.walk(ast.parse(extract_code(row['output']))))
        except SyntaxError:return False
    checks={
        'selected_tasks_128':sum(metrics['coder'][b]['tasks'] for b in metrics['coder'])+metrics['tester']['tasks']==128,
        'complete_generations':len(coder)==96 and len(tester)==64,
        'generation_requests_succeeded':all('output' in row and not row.get('error') for row in coder+tester),
        'no_coder_transport_errors':all(r['execution'].get('status')!='transport_error' for r in coder_results),
        'coder_has_parseable_functions':any(contains_node(row,ast.FunctionDef) for row in coder),
        'tester_has_parseable_assertions':any(contains_node(row,ast.Assert) for row in tester),
        'tester_measurement_available':metrics['tester']['measured_tasks']>0,
    }
    result={'passed':all(checks.values()),'checks':checks,'checked_at':datetime.now(timezone.utc).isoformat(),
            'purpose':'Infrastructure and usable-output feasibility only; no minimum benchmark score or hyperparameter search.',
            'length_limited_outputs':sum(r['finish_reason']=='length' for r in coder+tester),
            'empty_outputs':sum(not r['output'].strip() for r in coder+tester)}
    (directory/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['passed'],'Feasibility gate failed; inspect records before allocating the full run.'
