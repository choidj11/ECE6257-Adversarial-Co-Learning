#!/usr/bin/env python3
"""Freeze the inference-only LFM3 protocol and a 128-task feasibility subset."""
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/lfm3-evaluation'
OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'protocol.json').exists(),'An evaluation protocol already exists.'
revision=json.loads((OUT/'model-revision.json').read_text())
sources={
    'HumanEval':next((ROOT/'data/HumanEval/openai_humaneval').glob('*.parquet')),
    'BigCodeBench':next((ROOT/'data/BigCodeBench/data').glob('v0.1.4*.parquet')),
    'ULT_Lite':ROOT/'data/UnLeakedTestBench/ULT_Lite.jsonl',
}
tasks={key:([str(r['task_id']) for r in json.loads(path.read_text())] if key=='ULT_Lite'
             else [r['task_id'] for r in pq.read_table(path).to_pylist()]) for key,path in sources.items()}
small_counts={'HumanEval':32,'BigCodeBench':64,'ULT_Lite':32}
small={key:random.Random(42).sample(ids,small_counts[key]) for key,ids in tasks.items()}
protocol={
    'model_id':revision['model_id'],'revision':revision['revision'],
    'mode':'inference_only','training_steps':0,'parameter_updates':False,'wandb_mode':'disabled',
    'roles':['coder','tester'],'execution_feedback':False,'candidate_reranking':False,
    'seed':42,'context_tokens':16384,'max_new_tokens':2048,'batch_size':32,
    'full_batch_size':32,
    'do_sample':True,'temperature':0.1,'top_k':50,'top_p':1.0,'repetition_penalty':1.1,
    'enable_thinking':False,'dtype':'bfloat16','attention_implementation':'sdpa',
    'coder_samples_per_task':1,'tester_queries':2,'assertions_per_query':8,
    'benchmark_variants':{'HumanEval':'Original OpenAI HumanEval tests, not HumanEval+',
                          'BigCodeBench':'v0.1.4 Instruct, full 1140 tasks',
                          'ULT_Lite':'200-task bounded assertion/AST-mutation adaptation'},
    'small_counts':small_counts,'full_counts':{k:len(v) for k,v in tasks.items()},
    'dataset_sha256':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in sources.items()},
    'small_selection':small,'full_selection':tasks,
    'reference_calibration':'artifacts/study/reference-harnesses.jsonl',
    'reference_calibration_sha256':hashlib.sha256((ROOT/'artifacts/study/reference-harnesses.jsonl').read_bytes()).hexdigest(),
    'limitations':[
        'Fixed untrained checkpoint used separately for coder and tester benchmark tasks; no co-training or inference-time repair.',
        'One sampled coder response per task and one seed; pass@1 is not a repeated-sampling estimate.',
        'ULT Lite uses up to four deterministic AST mutants per function, not the official full Cosmic Ray evaluation.',
        'The prior Qwen study used different generation settings; this is not a controlled model comparison.',
        'Full evaluation reuses feasibility outputs on the same tasks and settings to avoid duplicate GPU work.',
    ],
}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for stage,selection in [('small',small),('full',tasks)]:
    directory=OUT/stage;directory.mkdir(exist_ok=True)
    (directory/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print(json.dumps({k:protocol[k] for k in ('model_id','revision','small_counts','full_counts','training_steps')},indent=2))
