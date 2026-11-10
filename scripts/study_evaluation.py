"""Shared, versioned input and isolated execution helpers for the study."""
import ast
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from sandbox_fusion import run_code, RunCodeRequest

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/study'


def read_tasks():
    he=pq.read_table(next((ROOT/'data/HumanEval/openai_humaneval').glob('*.parquet'))).to_pylist()
    bcb=pq.read_table(next((ROOT/'data/BigCodeBench/data').glob('v0.1.4*.parquet'))).to_pylist()
    for row in he:
        row['benchmark']='HumanEval'
    for row in bcb:
        row['benchmark']='BigCodeBench'
    return he+bcb


def extract_code(output):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',output,re.S)
    if blocks:
        return '\n'.join(blocks)
    if output.lstrip().startswith('```'):
        return output.split('\n',1)[-1].removesuffix('```').strip()
    return output.strip()


def coder_test(row,candidate):
    seed='import random\nrandom.seed(42)\n'
    if row['benchmark']=='HumanEval':
        return seed+candidate+'\n'+row['test']+f"\ncheck({row['entry_point']})\n"
    return seed+candidate+'\n'+row['test']+"\nif __name__ == '__main__':\n    unittest.main(exit=True)\n"


def execute(code,timeout=120):
    try:
        response=run_code(RunCodeRequest(code=code,language='python',run_timeout=timeout),
                          max_attempts=1,client_timeout=timeout+30)
        run=response.run_result
        return {'passed':response.status=='Success','status':str(response.status.value),
                'run_status':run.status.value if run else None,
                'seconds':run.execution_time if run else None,
                'stdout':(run.stdout or '')[-16000:] if run else '',
                'stderr':(run.stderr or '')[-16000:] if run else '',
                'return_code':run.return_code if run else None}
    except Exception as error:
        return {'passed':False,'status':'transport_error','error':str(error)}


def mutations(source,limit=4):
    """Deterministic bounded condition/arithmetic mutants, distinct and compilable."""
    tree=ast.parse(source)
    candidates=[]
    replacements={ast.Lt:ast.GtE,ast.LtE:ast.Gt,ast.Gt:ast.LtE,ast.GtE:ast.Lt,
                  ast.Eq:ast.NotEq,ast.NotEq:ast.Eq,ast.Add:ast.Sub,ast.Sub:ast.Add,
                  ast.Mult:ast.Add,ast.And:ast.Or,ast.Or:ast.And}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Compare,ast.BinOp,ast.BoolOp)):
            ops=node.ops if isinstance(node,ast.Compare) else [node.op]
            for index,op in enumerate(ops):
                if type(op) in replacements:
                    newop=replacements[type(op)]()
                    if isinstance(node,ast.Compare):
                        node.ops[index]=newop
                    else:
                        node.op=newop
                    candidate=ast.unparse(tree)
                    if isinstance(node,ast.Compare):
                        node.ops[index]=op
                    else:
                        node.op=op
                    if candidate not in candidates:
                        compile(candidate,'mutant.py','exec')
                        candidates.append(candidate)
                    if len(candidates)==limit:
                        return candidates
    return candidates
