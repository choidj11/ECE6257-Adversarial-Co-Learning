#!/usr/bin/env python3
"""Generate fixed-budget coder or iterative tester outputs for the frozen study."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

parser=argparse.ArgumentParser()
parser.add_argument('variant',choices=['base','E0','E1','E2','E3'])
parser.add_argument('role',choices=['coder','tester'])
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/study'
run=STUDY/args.variant
run.mkdir(exist_ok=True)
if args.variant=='base':
    model=ROOT/'models/Qwen2.5-Coder-1.5B-Instruct'
else:
    paths=sorted((run/args.role).glob('*/global_step_32/actor_rollout_ref/huggingface'))
    assert len(paths)==1,paths
    model=paths[0]
    assert list(model.glob('*.safetensors')),model
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
llm=LLM(model=str(model),tokenizer=str(model),dtype='bfloat16',max_model_len=8192,
        max_num_seqs=8,max_num_batched_tokens=8192,enforce_eager=True,
        gpu_memory_utilization=0.35,disable_log_stats=True,tensor_parallel_size=1,seed=42)
sampling=SamplingParams(temperature=0,max_tokens=1024)
destination=run/f'{args.role}-generations.jsonl'
assert not destination.exists(),f'Refusing to overwrite {destination}'
if args.role=='coder':
    he=pq.read_table(next((ROOT/'data/HumanEval/openai_humaneval').glob('*.parquet'))).to_pylist()
    bcb=pq.read_table(next((ROOT/'data/BigCodeBench/data').glob('v0.1.4*.parquet'))).to_pylist()
    tasks=[{'benchmark':'HumanEval','task_id':r['task_id'],'prompt':
            'Implement this Python function. Return only the complete function with required imports inside a Python code block.\n'+r['prompt']}
           for r in he]
    tasks.extend({'benchmark':'BigCodeBench','task_id':r['task_id'],'prompt':
                  'Implement the requested Python function, including required imports. Return only the complete Python code inside a Python code block.\n'+r['instruct_prompt']}
                 for r in bcb)
else:
    ult=json.loads((ROOT/'data/UnLeakedTestBench/ULT_Lite.jsonl').read_text())
    tasks=[{'benchmark':'ULT_Lite','task_id':str(r['task_id']),'source':r['code'],'function':r['func_name'],
            'prompt':'Generate 8 assertion-based tests for the following Python function. The function is already defined; do not repeat or redefine it. '
                     'Each test must be a standalone assert calling the function and comparing its result to an expected value. '
                     'Return only assert statements inside one Python code block.\nDescription:\n'+r['prompt']+'\nFunction:\n'+r['code']}
           for r in ult]
history={}
with destination.open('x') as file:
    for round_index in range(1 if args.role=='coder' else 2):
        for start in range(0,len(tasks),32):
            batch=tasks[start:start+32]
            prompts=[]
            active=[]
            for task in batch:
                prompt=task['prompt']
                if round_index:
                    prompt+='\nPreviously generated tests (generate distinct additional cases):\n'+history.get(task['task_id'],'')
                formatted=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                                                        tokenize=False,add_generation_prompt=True)
                if len(tokenizer.encode(formatted,add_special_tokens=False))+1024>8192:
                    file.write(json.dumps({'benchmark':task['benchmark'],'task_id':task['task_id'],
                                           'round':round_index+1,'error':'prompt_exceeds_context','output':''})+'\n')
                    continue
                prompts.append(formatted)
                active.append(task)
            if not active:
                continue
            outputs=llm.generate(prompts,sampling,use_tqdm=False)
            for task,output in zip(active,outputs):
                text=output.outputs[0].text
                record={'benchmark':task['benchmark'],'task_id':task['task_id'],'round':round_index+1,
                        'output':text,'prompt_tokens':len(output.prompt_token_ids),
                        'generated_tokens':len(output.outputs[0].token_ids),
                        'finish_reason':output.outputs[0].finish_reason}
                file.write(json.dumps(record)+'\n')
                history[task['task_id']]=text
            file.flush()
            print(f'{args.variant} {args.role} round {round_index+1}: {min(start+32,len(tasks))}/{len(tasks)}',flush=True)
(run/f'{args.role}-generation-meta.json').write_text(json.dumps({
    'completed_at':datetime.now(timezone.utc).isoformat(),'model':str(model),'seed':42,
    'temperature':0,'max_new_tokens':1024,'feedback_at_evaluation':False,
    'tasks':len(tasks),'rounds':1 if args.role=='coder' else 2},indent=2)+'\n')
