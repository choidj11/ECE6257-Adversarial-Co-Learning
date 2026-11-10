#!/usr/bin/env python3
"""Backfill and stream this study to W&B without occupying or restarting GPUs.

Auto mode waits for credentials in the user's environment/netrc. If the study
finishes first, it creates offline W&B runs for later `wandb sync`.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import netrc
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from omegaconf import OmegaConf
import wandb
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/study'
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['auto','online','offline'],default='auto')
parser.add_argument('--poll-seconds',type=float,default=30)
parser.add_argument('--max-hours',type=float,default=4)
parser.add_argument('--target-config',type=Path,default=STUDY/'wandb-config.json')
parser.add_argument('--status-file',type=Path,default=STUDY/'wandb-status.json')
args=parser.parse_args()


def credentials():
    key=dotenv_values(ROOT/'.env').get('WANDB_API_KEY') or os.environ.get('WANDB_API_KEY')
    if key: return key
    host=urlsplit(os.environ.get('WANDB_BASE_URL','https://api.wandb.ai')).hostname
    try:
        auth=netrc.netrc(os.environ.get('NETRC',str(Path.home()/'.netrc'))).authenticators(host)
        return auth[2] if auth else None
    except (OSError,netrc.NetrcParseError):
        return None


def metrics_for(variant,job):
    steps={}
    path=ROOT/f'artifacts/setup/study-train-{job}.log'
    if path.exists():
        content=re.sub(r'\x1b\[[0-9;]*m','',path.read_text(errors='replace'))
        for line in content.splitlines():
            match=re.search(r'step:(\d+) - (.*)',line)
            if not match: continue
            row=steps.setdefault(int(match.group(1)),{})
            for part in match.group(2).split(' - '):
                if ':' not in part: continue
                name,value=part.rsplit(':',1)
                try: value=float(value)
                except ValueError: continue
                # Console formatting rounds a real 1e-6 learning rate to 0.000.
                # Preserve it in config; only a full-precision stream logs /lr.
                if name.endswith('/lr') and value==0:
                    continue
                if math.isfinite(value): row[name]=value
    stream=STUDY/variant/'metrics-stream.jsonl'
    if stream.exists():
        for line in stream.read_text().splitlines():
            try:
                row=json.loads(line)
                steps.setdefault(int(row['step']),{}).update(row['metrics'])
            except (ValueError,json.JSONDecodeError):
                continue
    return steps


def flattened(value,prefix='evaluation'):
    result={}
    for key,item in value.items():
        name=prefix+'/'+key
        if isinstance(item,dict): result.update(flattened(item,name))
        elif type(item) in (int,float,bool) and math.isfinite(float(item)): result[name]=item
    return result


def upload_data(run,variant,include_text=False):
    directory=STUDY/variant
    artifact=wandb.Artifact(f'{variant}-study-metrics',type='experiment-metrics',
                            description='Training configurations and numeric metrics only; no raw prompts, outputs or model weights.')
    artifact.add_file(str(STUDY/'protocol.json'),name='protocol.json')
    filenames=['config.yaml','metrics-stream.jsonl','metrics.json']
    if include_text:
        filenames+=['coder-generations.jsonl','tester-generations.jsonl','coder-results.jsonl','tester-results.jsonl']
    for filename in filenames:
        path=directory/filename
        if path.is_file(): artifact.add_file(str(path),name=filename)
    if include_text:
        table=wandb.Table(columns=['step','role','input','output','score'])
        for role in ('coder','tester'):
            path=directory/f'{role}-rollouts'
            if path.exists():
                artifact.add_dir(str(path),name=f'{role}-rollouts')
                for file in sorted(path.glob('**/*.jsonl')):
                    for line in file.read_text().splitlines()[:2]:
                        row=json.loads(line)
                        table.add_data(int(file.stem),role,str(row.get('input',''))[:4000],
                                       str(row.get('output',''))[:4000],row.get('score'))
        if (directory/'feedback').exists(): artifact.add_dir(str(directory/'feedback'),name='feedback')
        run.log({'training_samples':table})
    run.log_artifact(artifact)


started=time.monotonic()
runs={}
last_steps={}
logged_evaluation=set()
status={'project':'6257-adversarial-co-training','runs':{},'state':'waiting_for_login'}
mode=None
try:
    while True:
        jobs=json.loads((STUDY/'jobs.json').read_text())
        options_path=args.target_config
        options=json.loads(options_path.read_text()) if options_path.exists() else {}
        target_authorized=(options.get('authorized') is True and bool(options.get('entity'))
                           and bool(options.get('project')) and options.get('scope') in ('metrics','text'))
        project=options.get('project',os.environ.get('WANDB_PROJECT','6257-adversarial-co-training'))
        entity=options.get('entity',os.environ.get('WANDB_ENTITY'))
        complete=all((STUDY/v/'metrics.json').exists() for v in ('base','E0','E1','E2','E3'))
        timed_out=time.monotonic()-started>args.max_hours*3600
        if mode is None:
            key=credentials()
            if args.mode=='offline': mode='offline'
            elif key and target_authorized:
                os.environ['WANDB_API_KEY']=key
                mode='online'
            elif complete or timed_out:
                mode='offline'
            elif args.mode=='online':
                raise RuntimeError('Explicit entity/project/payload authorization and W&B login are required')
        if mode:
            for variant in ('base','E0','E1','E2','E3'):
                if variant not in runs:
                    config_path=STUDY/variant/'config.yaml'
                    config=OmegaConf.to_container(OmegaConf.load(config_path),resolve=True) if config_path.exists() else {}
                    config['study_protocol']=json.loads((STUDY/'protocol.json').read_text())
                    config['slurm_job_id']=jobs['generation']['base'] if variant=='base' else jobs['training'][variant]
                    config['metric_source']='full-precision stream where present; early console-only metrics rounded to 3 decimals'
                    run=wandb.init(project=project,entity=entity,name=variant,
                                   id=f'a1-{variant}-{config["slurm_job_id"]}',group='study-20261108',
                                   job_type='evaluation' if variant=='base' else 'training',
                                   config=config,mode=mode,reinit='create_new',resume='allow' if mode=='online' else None,
                                   dir=str(ROOT/'artifacts/wandb'),
                                   settings=wandb.Settings(init_timeout=60,x_disable_stats=True,
                                                          disable_git=True,save_code=False,console='off'))
                    run.define_metric('train_step',hidden=True)
                    run.define_metric('*',step_metric='train_step')
                    runs[variant]=run
                    last_steps[variant]=0
                    status['runs'][variant]={'id':run.id,'url':run.url if mode=='online' else None,
                                             'local_dir':str(Path(run.dir).parent)}
                run=runs[variant]
                if variant!='base':
                    for step,values in sorted(metrics_for(variant,jobs['training'][variant]).items()):
                        if step<=last_steps[variant] or not all(f'actor_{role}/grad_norm' in values for role in ('a','b2')):
                            continue
                        run.log({'train_step':step,'training/completed_steps':step,
                                 'logged_train_steps':step,**values})
                        last_steps[variant]=step
                metrics_path=STUDY/variant/'metrics.json'
                if metrics_path.exists() and variant not in logged_evaluation:
                    result=json.loads(metrics_path.read_text())
                    run.log({'train_step':last_steps.get(variant,0),
                             'training/completed_steps':last_steps.get(variant,0),
                             'logged_train_steps':last_steps.get(variant,0),**flattened(result)})
                    run.summary.update(flattened(result))
                    upload_data(run,variant,include_text=options.get('scope')=='text' and target_authorized)
                    logged_evaluation.add(variant)
            status['state']='streaming_online' if mode=='online' else 'offline_ready'
        status.update({'project':project,'entity':entity,'mode':mode,
                       'updated_at':datetime.now(timezone.utc).isoformat(),
                       'last_train_steps':last_steps,'evaluation_uploaded':sorted(logged_evaluation)})
        args.status_file.write_text(json.dumps(status,indent=2)+'\n')
        if complete or timed_out:
            break
        time.sleep(args.poll_seconds)
finally:
    for run in runs.values(): run.finish()
if mode=='online':
    status['state']='synced'
    # Verify each created cloud run through the authenticated public API.
    api=wandb.Api()
    for run in runs.values():
        verified=api.run(f'{run.entity}/{run.project}/{run.id}')
        assert verified.id==run.id
elif mode=='offline':
    status['state']='offline_ready'
status['finished_at']=datetime.now(timezone.utc).isoformat()
args.status_file.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
