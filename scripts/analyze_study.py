#!/usr/bin/env python3
"""Build evidence tables, paired task intervals, figures, and the final report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/study'
variants=['base','E0','E1','E2','E3']
summaries={variant:json.loads((STUDY/variant/'metrics.json').read_text()) for variant in variants}
jobs=json.loads((STUDY/'jobs.json').read_text())
protocol=json.loads((STUDY/'protocol.json').read_text())
resources=json.loads((STUDY/'resources.json').read_text())
subprocess.run([sys.executable,str(ROOT/'scripts/audit_study_overlap.py')],check=True,cwd=ROOT)
overlap=json.loads((STUDY/'overlap-audit.json').read_text())
curves={}
for variant in variants[1:]:
    path=ROOT/f"artifacts/setup/study-train-{jobs['training'][variant]}.log"
    log=re.sub(r'\x1b\[[0-9;]*m','',path.read_text(errors='replace'))
    assert 'Dual model collaborative training completed!' in log,(variant,'training did not complete')
    records=[]
    for line in log.splitlines():
        match=re.search(r'step:(\d+) - (.*)',line)
        if not match:
            continue
        metrics={}
        for part in match.group(2).split(' - '):
            if ':' in part:
                key,value=part.rsplit(':',1)
                try: metrics[key]=float(value)
                except ValueError: pass
        metrics['step']=int(match.group(1))
        records.append(metrics)
    for role in ('a','b2'):
        role_records=[r for r in records if f'actor_{role}/grad_norm' in r]
        assert {r['step'] for r in role_records}==set(range(1,33)),(variant,role,'missing steps')
        assert all(np.isfinite(r[f'actor_{role}/grad_norm']) for r in role_records)
    curves[variant]=records


def load_results(variant,role):
    return {r['task_id']:r for r in map(json.loads,(STUDY/variant/f'{role}-results.jsonl').read_text().splitlines())}


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(42)
    means=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(2000)])
    return {'delta':float(values.mean()),'lower':float(np.quantile(means,.025)),
            'upper':float(np.quantile(means,.975)),'tasks':len(values)}


baseline_coder=load_results('E0','coder')
baseline_tester=load_results('E0','tester')
comparisons={}
examples={}
for variant in variants[2:]:
    coder=load_results(variant,'coder')
    tester=load_results(variant,'tester')
    comparisons[variant]={}
    examples[variant]={}
    for benchmark in ('HumanEval','BigCodeBench'):
        keys=sorted(key for key,row in baseline_coder.items() if row['benchmark']==benchmark and row['reference_passed'])
        differences=[int(coder[key]['execution']['passed'])-int(baseline_coder[key]['execution']['passed']) for key in keys]
        comparisons[variant][benchmark]=interval(differences)
        examples[variant][benchmark]={'improved':[key for key,d in zip(keys,differences) if d>0][:3],
                                       'regressed':[key for key,d in zip(keys,differences) if d<0][:3]}
    def tester_value(row,metric):
        m=row.get('metrics')
        if not m: return 0.0
        if metric=='valid_tests_per_query': return m['valid_tests']/2
        if metric=='bounded_mutation_score': return m['killed_mutants']/max(1,m['mutants'])
        c=m['coverage']
        return (c.get('branch_covered') or 0)/max(1,c.get('branch_total') or 0)
    for metric in ('valid_tests_per_query','bounded_mutation_score','branch_coverage'):
        comparisons[variant][metric]=interval([tester_value(tester[key],metric)-tester_value(baseline_tester[key],metric)
                                               for key in sorted(baseline_tester)])

feedback={}
for variant in variants[2:]:
    records=[]
    for path in sorted((STUDY/variant/'feedback').glob('*.jsonl')):
        records.extend(map(json.loads,path.read_text().splitlines()))
    assert len(records)==32*4*2,(variant,'missing feedback records',len(records))
    feedback[variant]={'records':len(records),'input_tokens':sum(r['input_tokens'] for r in records),
                       'execution_seconds_sum':sum(r['trace']['execution_seconds'] for r in records),
                       'collection_errors':sum(bool(r['trace'].get('error')) for r in records)}
generation_summary={}
for variant in variants:
    generation_summary[variant]={}
    for role,expected in (('coder',1304),('tester',400)):
        rows=[json.loads(line) for line in (STUDY/variant/f'{role}-generations.jsonl').read_text().splitlines()]
        assert len(rows)==expected,(variant,role,'incomplete generations')
        generation_summary[variant][role]={
            'outputs':len(rows),'length_limited_outputs':sum(r.get('finish_reason')=='length' for r in rows),
            'context_failures':sum(r.get('error')=='prompt_exceeds_context' for r in rows),
            'prompt_tokens':sum(r.get('prompt_tokens',0) for r in rows),
            'generated_tokens':sum(r.get('generated_tokens',0) for r in rows)}
analysis={'completed_at':datetime.now(timezone.utc).isoformat(),'summaries':summaries,
          'paired_task_bootstrap_95ci':comparisons,'examples':examples,'feedback_cost':feedback,
          'training_curves':curves,'training_seed_count':1,'bootstrap_replicates':2000,
          'resources':resources,'generation_summary':generation_summary,'overlap_audit':overlap}
(STUDY/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=ROOT/'report/figures'
FIG.mkdir(parents=True,exist_ok=True)
colors=['#7c8795','#245a81','#469989','#b58a42','#8862a0']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,benchmark in zip(axes,('HumanEval','BigCodeBench')):
    ax.bar(variants,[100*summaries[v]['coder'][benchmark]['qualified_pass_at_1'] for v in variants],color=colors)
    ax.set(title=benchmark+' (reference-qualified tasks)',ylabel='Pass@1 (%)',ylim=(0,100))
fig.savefig(FIG/'coder-results.png',dpi=180); plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,metric,label in zip(axes,('valid_tests_per_query','mean_branch_coverage','bounded_mutation_score'),
                           ('Valid assertions / query','Mean branch coverage','Bounded mutation kill rate')):
    factor=1 if metric=='valid_tests_per_query' else 100
    ax.bar(variants,[factor*summaries[v]['tester'][metric] for v in variants],color=colors)
    ax.set(title=label,ylabel='Count' if factor==1 else 'Percent')
fig.savefig(FIG/'tester-results.png',dpi=180); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for variant,color in zip(variants[1:],colors[1:]):
    for column,role in enumerate(('a','b2')):
        rs=[r for r in curves[variant] if f'reward_{role}/mean' in r]
        axes[0,column].plot([r['step'] for r in rs],[r[f'reward_{role}/mean'] for r in rs],label=variant,color=color)
        ls=[r for r in curves[variant] if f'actor_{role}/pg_loss' in r]
        axes[1,column].plot([r['step'] for r in ls],[r[f'actor_{role}/pg_loss'] for r in ls],label=variant,color=color)
for column,role in enumerate(('Coder','Tester')):
    axes[0,column].set(title=role+' training reward',xlabel='Training step',ylabel='Reward')
    axes[1,column].set(title=role+' policy loss (console precision)',xlabel='Training step',ylabel='Policy loss')
    axes[0,column].legend(); axes[1,column].legend()
fig.savefig(FIG/'training-curves.png',dpi=180); plt.close(fig)

reference=json.loads((STUDY/'reference-harnesses-summary.json').read_text())
lines=['# Adversarial Co-Training: Fixed-Budget Study Report','',
       'This report compares the untrained model, the Code-A1 baseline, and three execution-feedback variants. Conclusions apply only to this resource-limited, 32-step, single-seed study; they do not establish full-budget paper reproduction or official leaderboard scores.','',
       '## Goal and implementation','',
       'Test whether branch coverage and return-time locals from candidate code improve tester quality and coder performance under a fixed output budget. All four arms share the base model, 128 training problems, 32 steps, batch size 4, n=2 per coder/tester, and a 1,024-token response cap. The learning rate is 1e-6 and the KL coefficient is 0.001.',
       '', 'E0 runs official Code-A1 directly. E1 supplies coverage, E2 runtime values, and E3 both. Each enhanced arm probes candidate code with the first two fixed training tests and adds at most 384 feedback tokens. Expected assertion answers and reference implementations are excluded from tester prompts. Feedback input and execution add costs.',
       '', 'See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the full protocol and [CODE_A1_METHOD.md](CODE_A1_METHOD.md) for the upstream implementation review.',
       '', f'An exact AST comparison checked {overlap["training_rows"]} training sources against {overlap["evaluation_tasks"]} reference implementations across the three evaluation sets, removing docstrings/comments/formatting while retaining identifiers and imports. It found {len(overlap["exact_normalized_source_matches"])} identical-source matches; unparsed training/evaluation sources: {overlap["unparsed_training_sources"]}/{overlap["unparsed_evaluation_sources"]}. This does not rule out paraphrased or semantic overlap, or base-model pretraining contamination. See `artifacts/study/overlap-audit.json`.','',
       '## Training evidence and reproducibility','',
       'All four arms must have finite-gradient records for both models at steps 1–32, final HF weights, and per-round rollouts; the report script exits if this evidence is missing. Weights, data, and raw logs are in Git-ignored `artifacts/`, `data/`, and `models/` directories and are reproduced through the scripts.','',
       '| Arm | Added feedback | Training job | Final step |','| --- | --- | --- | --- |']
for variant,description in zip(variants[1:],('None','Coverage','Runtime values','Both')):
    lines.append(f'| {variant} | {description} | {jobs["training"][variant]} | 32 |')
lines += ['', '![Training curves](report/figures/training-curves.png)','',
          'Training rewards reflect the current coder/tester interaction and change as both models change; they alone are not evidence of benchmark improvement. GRPO policy loss can be negative, and its sign or magnitude does not directly rank model quality. This study\'s console-only curves retain three decimals.','',
          '## Coder evaluation','',
          'All variants use the same greedy prompts, one output per task, and a 1,024-token cap. HumanEval has 164 tasks; BigCodeBench v0.1.4 Instruct has 1,140. Each candidate runs the same complete tests. Reference implementations are calibrated first. Report both the raw full-denominator pass rate and the rate on the subset whose reference implementations pass; a failed reference calibration is not treated as conclusive evidence of model failure.','',
          '| Variant | HumanEval raw pass@1 | HumanEval qualified subset | BigCodeBench raw pass@1 | BigCodeBench qualified subset |',
          '| --- | --- | --- | --- | --- |']
for v in variants:
    he=summaries[v]['coder']['HumanEval']; bcb=summaries[v]['coder']['BigCodeBench']
    lines.append(f'| {v} | {he["raw_pass_at_1"]:.2%} | {he["qualified_pass_at_1"]:.2%} | {bcb["raw_pass_at_1"]:.2%} | {bcb["qualified_pass_at_1"]:.2%} |')
lines += ['',f'Reference calibration: HumanEval {reference["benchmarks"]["HumanEval"]["reference_passed"]}/164; BigCodeBench {reference["benchmarks"]["BigCodeBench"]["reference_passed"]}/1140. Failed task IDs and raw stderr are retained in `artifacts/study/reference-harnesses.jsonl`.',
          '', '![Coder results](report/figures/coder-results.png)','',
          '## Tester evaluation and ablations','',
          'ULT Lite uses the fixed 200 functions, K=2 rounds, and at most 8 assertions per round. Round two sees round one\'s output. No variant receives extra execution feedback during evaluation, so the comparison measures capabilities after training. Only tests passing on the reference function contribute to coverage and mutant kills. Each function has at most 4 deterministic AST comparison/arithmetic/Boolean mutants. This is a bounded mutation metric, not the official full Cosmic-Ray score.','',
          '| Variant | Valid/submitted tests | Valid tests/query | Mean line coverage | Mean branch coverage | Bounded mutation score | Measurement failures |',
          '| --- | --- | --- | --- | --- | --- | --- |']
for v in variants:
    t=summaries[v]['tester']
    lines.append(f'| {v} | {t["valid_tests"]}/{t["submitted_tests"]} | {t["valid_tests_per_query"]:.3f} | {t["mean_line_coverage"]:.2%} | {t["mean_branch_coverage"]:.2%} | {t["bounded_mutation_score"]:.2%} | {t["measurement_failures"]} |')
lines += ['', 'Count at most 8 top-level assertions per round. A whole output with syntax errors, redefined functions, or no top-level assertions counts as one invalid submission. Valid tests/query always uses the fixed denominator of 400 queries. Mean coverage weights all 200 functions equally, with zero for measurement failures; branchless functions also receive zero branch coverage, and line coverage is reported separately. Test-validity and aggregate mutation ratios use successfully measured functions and must be read alongside failure counts. Paired analysis instead weights per-function ratios equally, assigning zero to failures or functions with no mutants; its mutation difference need not equal the difference between the aggregate ratios above.','',
          '![Tester results](report/figures/tester-results.png)','',
          '## Paired differences from the baseline','',
          'Pair the same tasks and report 95% intervals from 2,000 task bootstrap resamples. These describe task-sampling uncertainty, not training-seed variance. All four arms use the same training seed, and no multiple-comparison correction is applied.','',
          '| Arm | Metric | Difference | 95% interval |','| --- | --- | --- | --- |']
for v in variants[2:]:
    for metric,d in comparisons[v].items():
        scale=1 if metric=='valid_tests_per_query' else 100
        unit='' if scale==1 else ' pp'
        lines.append(f'| {v} | {metric} | {d["delta"]*scale:+.3f}{unit} | [{d["lower"]*scale:+.3f}, {d["upper"]*scale:+.3f}]{unit} |')
lines += ['', '## Feedback costs and failure cases','',
          '| Arm | Traces | Extra input tokens | Summed probe time (seconds) | Collection errors |',
          '| --- | --- | --- | --- | --- |']
for v,c in feedback.items():
    lines.append(f'| {v} | {c["records"]} | {c["input_tokens"]} | {c["execution_seconds_sum"]:.1f} | {c["collection_errors"]} |')
lines += ['', 'Actual evaluation generation usage: 1,304 coder outputs and 400 tester outputs. A length-limit stop can leave code incomplete.','',
          '| Variant | Coder input/output tokens | Coder length-limited | Tester input/output tokens | Tester length-limited | Oversized-input failures |',
          '| --- | --- | --- | --- | --- | --- |']
for v in variants:
    c=generation_summary[v]['coder'];t=generation_summary[v]['tester']
    lines.append(f'| {v} | {c["prompt_tokens"]}/{c["generated_tokens"]} | {c["length_limited_outputs"]}/1304 | {t["prompt_tokens"]}/{t["generated_tokens"]} | {t["length_limited_outputs"]}/400 | {c["context_failures"]+t["context_failures"]} |')
quota=resources['scratch_quota']
lines += ['',f'Post-study Scratch quota: {quota["used_gib"]:.1f}/{quota["limit_gib"]:.0f} GiB used, {quota["free_gib"]:.1f} GiB free; {quota["used_files"]:,}/{quota["limit_files"]:,} files. This quota is shared by the user\'s entire Scratch space. Startup guards reserve 20 GiB and 20,000 files.']
lines += ['', 'The three feedback arms use the same probe counts and collection settings, but actual execution costs and feedback lengths differ. Summed request durations are not concurrent walltime. Per-example feedback/code links are in `artifacts/study/E*/feedback/`; raw generation files record actual prompt/output tokens and truncation status.','',
          f'Actual Slurm allocation totals **{resources["gpu_hours"]:.3f} GPU-hours** for this study. Per-job GPU type, duration, and exit state are in `artifacts/study/resources.json`. E1 trained on H200; the other three arms trained on H100. A single seed does not control hardware-dependent numerical/generation differences, so differences cannot be attributed entirely to feedback. Base CPU scoring requested 8 cores, later variants 4, all with four concurrent sandboxes; walltime differences do not establish algorithmic efficiency.','',
          'Representative paired examples, selected by sorted task ID rather than only showing successes:','']
for v,benchmarks in examples.items():
    for benchmark,example in benchmarks.items():
        lines.append(f'- {v} / {benchmark}: newly passed {", ".join(example["improved"]) or "none"}; regressed {", ".join(example["regressed"]) or "none"}.')
lines += ['', '## Conclusions and limitations','']
for v in variants[2:]:
    d=comparisons[v]['bounded_mutation_score']
    statement='The interval includes zero; current evidence does not establish a stable improvement.' if d['lower']<=0<=d['upper'] else ('The interval is above zero, indicating a positive difference under this study\'s task resampling.' if d['lower']>0 else 'The interval is below zero, indicating a negative difference under this study\'s task resampling.')
    lines.append(f'{v}: mean per-function bounded mutation-score difference from E0 is {d["delta"]*100:+.3f} percentage points. {statement} One seed, a small training subset, and limited generation length restrict generalization.\n')
lines += ['The training budget differs substantially from the paper. Some BigCodeBench dependencies, network requirements, or reference tests do not fit this sandbox. ULT uses an assertion interface and a small mutation-operator set, with equivalent-mutant and denominator limitations. Lack of improvement, truncation, timeouts, and measurement failures remain visible in the results. Future work should increase training scale and seeds, expand probes/mutation operators, and verify results in the complete official evaluation environment.','',
          '## Reproduction entrypoints','',
          '```bash','source scripts/activate.sh rl','python scripts/check_quota.py',
          '# Fresh workspace only: environment/data prepared and no jobs.json present.',
          'python scripts/submit_study.py --wandb',
          '# --wandb requires an explicitly authorized metrics configuration; omit for no upload.',
          '# Reanalyze the complete existing study:',
          'python scripts/record_study_resources.py','python scripts/analyze_study.py','```','',
          'Job mapping: `artifacts/study/jobs.json`; complete statistics and curves: `artifacts/study/analysis.json`. Use a separate workspace or archive this study before rerunning; generation refuses to overwrite original outputs.','']
wandb_config_path=STUDY/'wandb-config.json'
if wandb_config_path.exists():
    monitoring=json.loads(wandb_config_path.read_text())
    lines += ['## W&B monitoring','',f'Project: [6257-adversarial-co-training]({monitoring["url"]}). Uploads are limited to the user-approved numeric training metrics, configurations, resource statistics, and aggregate evaluation summaries. Raw prompts, model outputs, per-example logs, datasets, and weights are excluded.',
              '', 'Loss histories: `actor_a/pg_loss` and `actor_b2/pg_loss`. Final evaluation summaries: `eval/` and `evaluation/`. Earlier console-only data has three-decimal precision, documented in the run configuration; future structured streams can preserve full precision. Final cloud readback evidence is `artifacts/study/wandb-final-verified.json`.','']
(ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines))
print('PASS: completed evidence-backed report, figures and paired statistics')
