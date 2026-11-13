#!/usr/bin/env python3
"""Submit a gated inference-only feasibility/full evaluation DAG."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/lfm3-evaluation'
assert (ROOT/'models/LFM3-1.7B/model.safetensors').exists(),'Model download is not complete.'
assert not (OUT/'jobs.json').exists(),'Jobs already recorded; inspect before resuming.'
jobs={'training':{},'mode':'inference_only','generation':{},'evaluation':{},'attempts':[]}


def submit(script,stage,after=None,time_limit=None):
    command=['sbatch','--parsable']
    if after:command+=['--dependency=afterok:'+str(after)]
    if time_limit:command+=['--time='+time_limit]
    result=subprocess.run(command+[str(ROOT/'scripts'/script),stage],check=True,capture_output=True,text=True,cwd=ROOT)
    return int(result.stdout.strip().split(';')[0])


def save():
    (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')


try:
    jobs['generation']['small']=submit('generate_lfm_evaluation.sbatch','small',time_limit='01:00:00');save()
    jobs['evaluation']['small']=submit('evaluate_lfm_evaluation.sbatch','small',after=jobs['generation']['small']);save()
    jobs['generation']['full']=submit('generate_lfm_evaluation.sbatch','full',after=jobs['evaluation']['small']);save()
    jobs['evaluation']['full']=submit('evaluate_lfm_evaluation.sbatch','full',after=jobs['generation']['full']);save()
    jobs['report']=submit('report_lfm_evaluation.sbatch','full',after=jobs['evaluation']['full']);save()
finally:
    save()
print(json.dumps(jobs,indent=2))
