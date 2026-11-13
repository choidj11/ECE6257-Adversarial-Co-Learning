#!/usr/bin/env python3
"""Export verified aggregate results for Git; retain raw experiment files locally."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/lfm3-evaluation'
evidence=json.loads((OUT/'report-evidence.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
verification=json.loads((OUT/'final-verification.json').read_text())
assert evidence['passed'] and verification['passed']
assert evidence['training_steps']==0 and evidence['wandb_mode']=='disabled'
assert {k:v['tasks'] for k,v in evidence['full']['coder'].items()}=={'HumanEval':164,'BigCodeBench':1140}
assert evidence['full']['tester']['tasks']==200
configuration_keys=(
    'seed','context_tokens','max_new_tokens','batch_size','full_batch_size',
    'do_sample','temperature','top_k','top_p','repetition_penalty','enable_thinking',
    'dtype','attention_implementation','coder_samples_per_task','tester_queries',
    'assertions_per_query','small_counts','full_counts','benchmark_variants',
    'dataset_sha256','reference_calibration_sha256','limitations',
)
summary={
    'schema_version':1,'model_id':evidence['model_id'],'model_revision':evidence['revision'],
    'evaluation_completed_at':evidence['full']['completed_at'],'mode':evidence['mode'],
    'training_steps':0,'wandb_mode':'disabled',
    'protocol':{key:protocol[key] for key in configuration_keys},
    'results':{stage:evidence[stage] for stage in ('small','full')},
    'generation_diagnostics':evidence['generation_diagnostics'],
    'gpu_hours_including_cancelled_attempts':evidence['gpu_hours'],
    'verification':verification,
}
destination=ROOT/'report/lfm3/summary.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(summary,indent=2)+'\n')
print(f'Exported aggregate results to {destination.relative_to(ROOT)}')
