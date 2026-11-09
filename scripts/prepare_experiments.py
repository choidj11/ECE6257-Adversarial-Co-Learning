#!/usr/bin/env python3
"""Freeze a bounded four-arm study before looking at its evaluation results."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/study'
OUT.mkdir(parents=True, exist_ok=True)
model = str(ROOT / 'models/Qwen2.5-Coder-1.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
source = next((ROOT / 'data/Code-A1/train_data').glob('*.parquet'))
table = pq.read_table(source)
rows = table.to_pylist()
eligible = [i for i,r in enumerate(rows)
            if len(tokenizer.apply_chat_template(r['prompt'], add_generation_prompt=True)) <= 1024]
random.Random(42).shuffle(eligible)
selected = eligible[:128]
assert len(selected) == 128
pq.write_table(table.take(pa.array(selected)), OUT / 'train.parquet')
val = next((ROOT / 'data/Code-A1/val_data').glob('*.parquet'))
pq.write_table(pq.read_table(val).slice(0, 2), OUT / 'val.parquet')
if not (ROOT / 'artifacts/week12/smoke.yaml').exists():
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_week12.py')],check=True)
base = OmegaConf.load(ROOT / 'artifacts/week12/smoke.yaml')
protocol = {'seed':42, 'variants':['E0','E1','E2','E3'], 'steps':32, 'batch_size':4,
            'coder_samples_per_problem':2, 'tester_samples_per_code':2,
            'max_response_tokens':1024, 'training_problem_exposures_per_arm':128,
            'max_feedback_tokens':384, 'feedback_probes_per_code':2,
            'training_job_walltime_seconds':3600,'generation_job_walltime_seconds':7200,
            'source_train_rows':len(rows), 'eligible_rows':len(eligible), 'selected_source_rows':selected,
            'train_sha256':hashlib.sha256((OUT/'train.parquet').read_bytes()).hexdigest(),
            'evaluation':{'HumanEval':164,'BigCodeBench':1140,'ULT_Lite':200,
                          'coder_samples':1,'tester_queries':2,'max_new_tokens':1024,'temperature':0},
            'limitations':['Resource-limited reproduction, not the paper full-budget training.',
                           'One training seed; paired task uncertainty does not estimate seed variance.',
                           'Fixed output-token caps; extra feedback input and execution costs recorded separately.',
                           'Only final inference checkpoints; no optimizer resume checkpoint.']}
(OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
for variant in protocol['variants']:
    run = OUT/variant
    run.mkdir(exist_ok=True)
    config = OmegaConf.merge(base, {
        'data':{'train_files':str(OUT/'train.parquet'),'val_files':str(OUT/'val.parquet'),
                'train_batch_size':4,'max_response_length':1024,'shuffle':False,'seed':42},
        'actor_rollout_ref':{
            'actor':{'ppo_mini_batch_size':4,'ppo_max_token_len_per_gpu':4096,
                     'checkpoint':{'save_contents':['hf_model'],'load_contents':[]},
                     'optim':{'lr':1e-6,'lr_warmup_steps':0}},
            'rollout':{'response_length':1024,'seed':42}},
        'trainer':{'total_training_steps':32,'total_epochs':1,'save_freq':32,
                   'experiment_name':variant,'default_local_dir':str(run/'coder'),
                   'rollout_data_dir':str(run/'coder-rollouts'),
                   'validation_data_dir':str(run/'coder-validation')},
        'other':{'data':{'max_prompt_length':3072},
                 'trainer':{'default_local_dir':str(run/'tester'),
                            'rollout_data_dir':str(run/'tester-rollouts'),
                            'validation_data_dir':str(run/'tester-validation')}},
        'execution_feedback':{'variant':variant,'max_tokens':384,'probes':2,'run_dir':str(run)},
    })
    OmegaConf.save(config, run/'config.yaml')
print(json.dumps(protocol, indent=2))
