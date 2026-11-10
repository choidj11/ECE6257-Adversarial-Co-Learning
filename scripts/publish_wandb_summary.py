#!/usr/bin/env python3
"""Publish numeric finals to both history and summary; verify plot-ready readback."""
import json
import math
import os
from pathlib import Path
import re
import time

from dotenv import dotenv_values
import pyarrow as pa
import pyarrow.parquet as pq
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
    final_step=0 if variant=='base' else 32
    payload={'train_step':final_step,'training/completed_steps':final_step,
             'logged_train_steps':final_step}
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
    def matches(row):
        return all(type(row.get(key)) in (int,float) and
                   math.isclose(float(row[key]),float(value),rel_tol=1e-12,abs_tol=1e-12)
                   for key,value in payload.items())

    # Summary-only updates cannot populate line charts. Append one truthful final
    # observation with its custom X-axis; never replay or duplicate the loss curve.
    history_before=list(run.scan_history(use_cache=False))
    loss_keys=('actor_a/pg_loss','actor_b2/pg_loss')
    def loss_points(rows):
        # W&B may merge archived history and live rows with duplicate step IDs.
        # Require identical values for duplicates and compare canonical points.
        points={}
        for row in rows:
            if not all(type(row.get(key)) in (int,float) and math.isfinite(row[key])
                       for key in ('train_step',*loss_keys)):continue
            point={key:row[key] for key in ('train_step',*loss_keys)}
            step=int(row['train_step'])
            assert step not in points or points[step]==point,(variant,'conflicting loss records',step)
            points[step]=point
        return [points[step] for step in sorted(points)]

    losses_before=loss_points(history_before)
    plot_keys=[key for key in payload if key!='train_step']
    # The chart endpoint reads live history; archived full-history snapshots may
    # lag behind a resumed run even when the SDK disk cache is bypassed.
    plotted_before=run.history(keys=plot_keys,x_axis='train_step',pandas=False,samples=1000)
    already_logged=any(matches(row) for row in plotted_before)
    if not already_logged:
        assert run.state=='finished',(variant,'refusing concurrent writes to an active run',run.state)
        with wandb.init(entity=config['entity'],project=config['project'],id=run.id,
                        resume='must',mode='online',reinit='create_new',
                        dir=str(root/'artifacts/wandb'),
                        settings=wandb.Settings(init_timeout=60,x_disable_stats=True,
                                               disable_git=True,save_code=False,console='off')) as writer:
            writer.define_metric('train_step',hidden=True)
            for key in payload:
                if key!='train_step':writer.define_metric(key,step_metric='train_step',step_sync=False)
            writer.log(payload)
            writer.summary.update(payload)

    # Verify through the same sampled-history interface used for line charts.
    for attempt in range(30):
        actual=wandb.Api(timeout=30).run(path)
        plotted_after=actual.history(keys=plot_keys,x_axis='train_step',pandas=False,samples=1000)
        if any(matches(row) for row in plotted_after):break
        if attempt==29:raise AssertionError((variant,'final metrics missing aligned chart history'))
        time.sleep(2)
    history_after=list(actual.scan_history(use_cache=False))
    for key,value in payload.items():
        assert math.isclose(float(actual.summary[key]),float(value),rel_tol=1e-12,abs_tol=1e-12),(variant,key)
    losses_after=loss_points(history_after)
    assert losses_after==losses_before,(variant,'original loss history changed')
    history_keys=set(payload).union(*(row.keys() for row in history_after))
    plottable_keys={key for row in history_after
                    if type(row.get('train_step')) in (int,float) and math.isfinite(row['train_step'])
                    for key,value in row.items() if type(value) in (int,float) and math.isfinite(value)}
    plottable_keys.update(payload)  # Already verified together through the chart endpoint.
    summary_only=[key for key,value in actual.summary.items()
                  if not key.startswith('_') and type(value) in (int,float) and key not in plottable_keys]
    assert not summary_only,(variant,'remaining unplottable summary-only metrics',summary_only)
    allowed={'protocol.json','config.yaml','metrics-stream.jsonl','metrics.json'}
    artifacts=list(actual.logged_artifacts())
    assert artifacts,(variant,'missing numeric/configuration artifact')
    uploaded_files=[]
    internal_histories=[]
    for artifact in artifacts:
        files=list(artifact.files());names=[file.name for file in files]
        if artifact.type=='wandb-history':
            # W&B itself archives numeric run histories as Parquet artifacts.
            assert artifact.name.split(':')[0]==f'run-{actual.id}-history'
            assert all(re.fullmatch(r'\d+\.parquet',name) for name in names)
            assert all(file.size<=16*2**20 for file in files)
            folder=Path(artifact.download(root=str(study/'wandb-internal-history'/artifact.name.replace(':','-'))))
            for name in names:
                schema=pq.read_schema(folder/name)
                assert set(schema.names)<=history_keys,(variant,'unknown archived history columns')
                assert all(pa.types.is_floating(field.type) or pa.types.is_integer(field.type)
                           or pa.types.is_boolean(field.type) or pa.types.is_null(field.type)
                           for field in schema),(variant,'non-numeric archived history')
            internal_histories.append({'name':artifact.name,'numeric_columns_verified':True,'files':names})
        else:
            assert artifact.type=='experiment-metrics' and set(names)<=allowed,(variant,'unexpected uploaded files',names)
            uploaded_files.extend(names)
    loss_steps=0
    if variant!='base':
        steps={int(row['train_step']) for row in losses_after}
        assert steps==set(range(1,33)),(variant,'incomplete cloud loss history')
        loss_steps=len(steps)
    plot_counts={}
    for key in ('training/completed_steps','eval/HumanEval/raw_pass_at_1',*(loss_keys if variant!='base' else ())):
        for attempt in range(6):
            points=actual.history(keys=[key],x_axis='train_step',pandas=False,samples=1000)
            valid=[row for row in points if type(row.get(key)) in (int,float)
                   and type(row.get('train_step')) in (int,float)]
            if valid:break
            if attempt==5:raise AssertionError((variant,key,'chart query returned no data'))
            time.sleep(2)
        plot_counts[key]=len({row['train_step'] for row in valid})
        if key in loss_keys:assert {int(row['train_step']) for row in valid}==set(range(1,33))
    published.append({'variant':variant,'url':actual.url,'numeric_fields':len(payload),
                      'loss_history_steps':loss_steps,'verified_artifact_files':sorted(set(uploaded_files)),
                      'final_history_step':final_step,'history_fields_verified':len(payload),
                      'appended_history_row':not already_logged,'original_loss_history_preserved':True,
                      'remaining_summary_only_numeric_metrics':summary_only,
                      'chart_query_points':plot_counts,'internal_numeric_history_artifacts':internal_histories})
    print(f'{variant}: final history and chart queries verified; loss steps={loss_steps}',flush=True)
(study/'wandb-final-verified.json').write_text(json.dumps({'passed':True,'runs':published},indent=2)+'\n')
print(json.dumps({'passed':True,'runs':published},indent=2))
