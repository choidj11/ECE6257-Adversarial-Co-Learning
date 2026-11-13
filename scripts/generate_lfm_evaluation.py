#!/usr/bin/env python3
"""Resume inference without training; retain raw generations locally only."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/lfm3-evaluation'
parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['small','full'])
args=parser.parse_args()
protocol=json.loads((OUT/'protocol.json').read_text())
directory=OUT/args.stage
selection=json.loads((directory/'selection.json').read_text())
if args.stage=='full':
    assert json.loads((OUT/'small/gate.json').read_text())['passed'],'Feasibility gate has not passed.'
assert torch.cuda.is_available(),'Inference requires a GPU allocation.'
set_seed(protocol['seed'])
model_path=ROOT/'models/LFM3-1.7B'
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
tokenizer.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(
    model_path,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,
    attn_implementation=protocol['attention_implementation'],device_map='cuda')
model.eval()
started=time.monotonic()
events=[]


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


for role in ('coder','tester'):
    destination=directory/f'{role}-generations.jsonl'
    if args.stage=='full' and not destination.exists():
        cached=records(OUT/'small'/destination.name)
        destination.write_text(''.join(json.dumps(row)+'\n' for row in cached))
    existing=records(destination)
    done={(r['task_id'],r['round']) for r in existing}
    assert len(existing)==len(done),'Duplicate generation records.'
    assert all(r['revision']==protocol['revision'] for r in existing)
    history={r['task_id']:r['output'] for r in existing if r['round']==1}
    if role=='coder':
        tasks=[]
        for benchmark,folder,pattern,prompt_field in [
            ('HumanEval','data/HumanEval/openai_humaneval','*.parquet','prompt'),
            ('BigCodeBench','data/BigCodeBench/data','v0.1.4*.parquet','instruct_prompt')]:
            rows=pq.read_table(next((ROOT/folder).glob(pattern))).to_pylist()
            tasks.extend({'benchmark':benchmark,'task_id':r['task_id'],
                          'prompt':'Implement the requested Python function, including required imports. Return only the complete Python code inside a Python code block.\n'+r[prompt_field]}
                         for r in rows if r['task_id'] in selection[benchmark])
    else:
        rows=json.loads((ROOT/'data/UnLeakedTestBench/ULT_Lite.jsonl').read_text())
        tasks=[{'benchmark':'ULT_Lite','task_id':str(r['task_id']),
                'prompt':'Generate 8 assertion-based tests for the following Python function. The function is already defined; do not repeat or redefine it. Each test must be a standalone assert calling the function and comparing its result to an expected value. Return only assert statements inside one Python code block.\nDescription:\n'+r['prompt']+'\nFunction:\n'+r['code']}
               for r in rows if str(r['task_id']) in selection['ULT_Lite']]
    rounds=1 if role=='coder' else protocol['tester_queries']
    with destination.open('a') as file:
        for round_index in range(1,rounds+1):
            pending=[task for task in tasks if (task['task_id'],round_index) not in done]
            offset=0;batch_size=protocol.get('full_batch_size',protocol['batch_size']) if args.stage=='full' else protocol['batch_size']
            while offset<len(pending):
                batch=pending[offset:offset+batch_size]
                texts=[]
                for task in batch:
                    prompt=task['prompt']
                    if round_index>1:
                        prompt+='\nPreviously generated tests (generate distinct additional cases):\n'+history[task['task_id']]
                    text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                        tokenize=False,add_generation_prompt=True,enable_thinking=protocol['enable_thinking'])
                    assert len(tokenizer.encode(text,add_special_tokens=False))+protocol['max_new_tokens']<=protocol['context_tokens'],(role,task['task_id'],'context budget exceeded')
                    texts.append(text)
                inputs=tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
                try:
                    with torch.inference_mode():
                        outputs=model.generate(**inputs,max_new_tokens=protocol['max_new_tokens'],
                            do_sample=protocol['do_sample'],temperature=protocol['temperature'],
                            top_p=protocol['top_p'],top_k=protocol['top_k'],
                            repetition_penalty=protocol['repetition_penalty'],use_cache=True)
                except torch.cuda.OutOfMemoryError:
                    if batch_size==1:raise
                    batch_size=max(1,batch_size//2)
                    events.append({'event':'reduced_batch_after_oom','batch_size':batch_size,'role':role,'round':round_index})
                    del inputs
                    torch.cuda.empty_cache()
                    continue
                width=inputs['input_ids'].shape[1]
                eos=model.generation_config.eos_token_id
                eos={eos} if isinstance(eos,int) else set(eos)
                for task,tokens,mask in zip(batch,outputs[:,width:].tolist(),inputs['attention_mask']):
                    end=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens))
                    tokens=tokens[:end]
                    raw=tokenizer.decode(tokens,skip_special_tokens=True)
                    answer=raw.rsplit('</think>',1)[-1].strip()
                    record={'benchmark':task['benchmark'],'task_id':task['task_id'],'round':round_index,
                            'output':answer,'raw_output':raw,'revision':protocol['revision'],
                            'prompt_tokens':int(mask.sum().item()),'generated_tokens':len(tokens),
                            'finish_reason':'stop' if any(t in eos for t in tokens) else 'length'}
                    file.write(json.dumps(record)+'\n')
                    history[task['task_id']]=answer
                    done.add((task['task_id'],round_index))
                file.flush()
                offset+=len(batch)
                print(f'{args.stage} {role} round {round_index}: {len(done)}/{len(tasks)*rounds}',flush=True)
                del outputs,inputs
    assert len(done)==len(tasks)*rounds
    (directory/f'{role}-generation-meta.json').write_text(json.dumps({
        'completed_at':datetime.now(timezone.utc).isoformat(),'model':protocol['model_id'],
        'revision':protocol['revision'],'tasks':len(tasks),'rounds':rounds,
        'stage':args.stage,'training_steps':0},indent=2)+'\n')
(directory/'generation-resources.json').write_text(json.dumps({
    'completed_at':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
    'seconds':time.monotonic()-started,'gpu':torch.cuda.get_device_name(),
    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'events':events,
    'torch':torch.__version__,'initial_batch_size':protocol.get('full_batch_size',protocol['batch_size']) if args.stage=='full' else protocol['batch_size'],
},indent=2)+'\n')
