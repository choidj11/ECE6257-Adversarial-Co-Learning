#!/usr/bin/env python3
"""Publish only approved numeric final scores/resource summaries; verify readback."""
import json
import math
import os
from pathlib import Path

from dotenv import dotenv_values
import wandb

root=Path(__file__).resolve().parents[1]
study=root/'artifacts/study'
config=json.loads((study/'wandb-config.json').read_text())
assert config.get('authorized') is True and config.get('scope')=='metrics'
os.environ['WANDB_API_KEY']=dotenv_values(root/'.env')['WANDB_API_KEY']
os.environ['WANDB_MODE']='online';os.environ['WANDB_ERROR_REPORTING']='false'
jobs=json.loads((study/'jobs.json').read_text())
analysis=json.loads((study/'analysis.json').read_text())
resources={int(row['job_id']):row for row in analysis['resources']['jobs']}
api=wandb.Api(timeout=30)
published=[]
for variant in ('base','E0','E1','E2','E3'):
    identifier=jobs['generation']['base'] if variant=='base' else jobs['training'][variant]
    path=f"{config['entity']}/{config['project']}/a1-{variant}-{identifier}"
    run=api.run(path)
    metrics=analysis['summaries'][variant]
    payload={}
    for benchmark in ('HumanEval','BigCodeBench'):
        coder=metrics['coder'][benchmark]
        for field in ('raw_pass_at_1','qualified_pass_at_1','reference_qualified_tasks'):
            payload[f'eval/{benchmark}/{field}']=coder[field]
    for field in ('valid_tests_per_query','mean_branch_coverage','mean_line_coverage',
                  'bounded_mutation_score','measurement_failures'):
        payload[f'eval/ULT_Lite/{field}']=metrics['tester'][field]
    payload['compute/generation_gpu_hours']=resources[jobs['generation'][variant]]['gpu_hours']
    quota=analysis['resources']['scratch_quota']
    for field in ('used_gib','free_gib','used_files','free_files'):
        payload[f'storage/scratch_{field}']=quota[field]
    for role,fields in analysis['generation_summary'][variant].items():
        for field,value in fields.items():
            payload[f'eval/generation/{role}/{field}']=value
    if variant!='base':
        payload['compute/training_gpu_hours']=resources[jobs['training'][variant]]['gpu_hours']
        payload['training/completed_steps']=32
        for role in ('a','b2'):
            last=[row for row in analysis['training_curves'][variant] if f'actor_{role}/pg_loss' in row][-1]
            payload[f'loss/{role}_policy_final']=last[f'actor_{role}/pg_loss']
    if variant in analysis['feedback_cost']:
        costs=analysis['feedback_cost'][variant]
        payload['feedback/extra_input_tokens']=costs['input_tokens']
        payload['feedback/collection_error_count']=costs['collection_errors']
        payload['feedback/execution_seconds_sum']=costs['execution_seconds_sum']
    run.summary.update(payload)
    # Fetch via a new API object to verify the server rather than the local map.
    actual=wandb.Api(timeout=30).run(path)
    for key,value in payload.items():
        assert math.isclose(float(actual.summary[key]),float(value),rel_tol=1e-12,abs_tol=1e-12),(variant,key)
    allowed={'protocol.json','config.yaml','metrics-stream.jsonl','metrics.json'}
    artifacts=list(actual.logged_artifacts())
    assert artifacts,(variant,'missing numeric/configuration artifact')
    uploaded_files=[]
    for artifact in artifacts:
        names=[file.name for file in artifact.files()]
        assert set(names)<=allowed,(variant,'unexpected uploaded files',names)
        uploaded_files.extend(names)
    loss_steps=0
    if variant!='base':
        rows=list(actual.scan_history(keys=['train_step','actor_a/pg_loss','actor_b2/pg_loss']))
        steps={int(row['train_step']) for row in rows}
        assert steps==set(range(1,33)),(variant,'incomplete cloud loss history')
        loss_steps=len(steps)
    published.append({'variant':variant,'url':actual.url,'numeric_fields':len(payload),
                      'loss_history_steps':loss_steps,'verified_artifact_files':sorted(set(uploaded_files))})
(study/'wandb-final-verified.json').write_text(json.dumps({'passed':True,'runs':published},indent=2)+'\n')
print(json.dumps({'passed':True,'runs':published},indent=2))
