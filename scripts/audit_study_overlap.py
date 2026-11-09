#!/usr/bin/env python3
"""Check exact normalized source overlap without executing any dataset code."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

root=Path(__file__).resolve().parents[1]
study=root/'artifacts/study'


def fingerprint(source):
    try:
        tree=ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            body=node.body
            if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
                del body[0]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


training=pq.read_table(study/'train.parquet').to_pylist()
by_hash=defaultdict(list)
unparsed=0
for index,row in enumerate(training):
    key=fingerprint(row['GT_code'])
    if key: by_hash[key].append(index)
    else: unparsed+=1
evaluations=[]
he=pq.read_table(next((root/'data/HumanEval/openai_humaneval').glob('*.parquet'))).to_pylist()
evaluations.extend((r['task_id'],r['prompt']+r['canonical_solution']) for r in he)
bcb=pq.read_table(next((root/'data/BigCodeBench/data').glob('v0.1.4*.parquet'))).to_pylist()
evaluations.extend((r['task_id'],r['complete_prompt']+r['canonical_solution']) for r in bcb)
ult=json.loads((root/'data/UnLeakedTestBench/ULT_Lite.jsonl').read_text())
evaluations.extend(('ULT_Lite/'+str(r['task_id']),r['code']) for r in ult)
matches=[]
unparsed_eval=0
for identifier,source in evaluations:
    key=fingerprint(source)
    if key is None: unparsed_eval+=1
    elif key in by_hash:
        matches.append({'task_id':identifier,'training_subset_indices':by_hash[key]})
report={'training_rows':len(training),'evaluation_tasks':len(evaluations),
        'unparsed_training_sources':unparsed,'unparsed_evaluation_sources':unparsed_eval,
        'exact_normalized_source_matches':matches,
        'method':'AST equality after removing docstrings, comments and formatting; preserves identifiers and imports.',
        'limitation':'Does not detect semantic/paraphrased overlap or pretrained-model contamination.'}
(study/'overlap-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
