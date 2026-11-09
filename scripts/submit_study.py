#!/usr/bin/env python3
"""Submit the frozen study DAG in a fresh prepared workspace; never overwrite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
study=root/'artifacts/study'
parser=argparse.ArgumentParser()
parser.add_argument('--wandb',action='store_true',help='Use only the explicitly authorized metrics-only W&B project')
args=parser.parse_args()
assert not (study/'jobs.json').exists(),'This study already has jobs; use a fresh workspace or archive the finished study first.'
if args.wandb:
    config=json.loads((study/'wandb-config.json').read_text())
    assert config.get('authorized') is True and config.get('scope')=='metrics'
subprocess.run([sys.executable,str(root/'scripts/prepare_experiments.py')],check=True,cwd=root)
jobs={'training':{},'generation':{},'evaluation':{},'attempts':[]}


def submit(script,*arguments,after=()):
    command=['sbatch','--parsable']
    if after: command+=['--dependency=afterok:'+':'.join(map(str,after))]
    command += [str(root/'scripts'/script),*arguments]
    result=subprocess.run(command,check=True,capture_output=True,text=True,cwd=root)
    return int(result.stdout.strip().split(';',1)[0])


def save():
    (study/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')


try:
    jobs['calibration']=submit('calibrate_study.sbatch');save()
    jobs['generation']['base']=submit('generate_study.sbatch','base');save()
    jobs['evaluation']['base']=submit('evaluate_study.sbatch','base',after=[jobs['calibration'],jobs['generation']['base']]);save()
    for variant in ('E0','E1','E2','E3'):
        jobs['training'][variant]=submit('train_study.sbatch',variant);save()
        jobs['generation'][variant]=submit('generate_study.sbatch',variant,after=[jobs['training'][variant]]);save()
        jobs['evaluation'][variant]=submit('evaluate_study.sbatch',variant,after=[jobs['calibration'],jobs['generation'][variant]]);save()
    jobs['analysis']=submit('analyze_study.sbatch',after=list(jobs['evaluation'].values()));save()
    if args.wandb:
        jobs['wandb_monitor']=submit('wandb_monitor.sbatch','online');save()
        jobs['wandb_final']=submit('publish_wandb_summary.sbatch',after=[jobs['analysis'],jobs['wandb_monitor']]);save()
finally:
    save()
print(json.dumps(jobs,indent=2))
