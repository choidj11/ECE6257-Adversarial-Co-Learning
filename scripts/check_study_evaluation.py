#!/usr/bin/env python3
"""Known-answer integration check of validity, mutation kills, and coverage."""
import json
from pathlib import Path
from study_evaluation import ROOT, execute, mutations

source='def choose(x):\n    if x > 0:\n        return x + 1\n    return -1\n'
mutants=mutations(source)
assert len(mutants)==2
generated=['assert choose(2) == 3\nassert choose(2) == 999','assert choose(0) == -1']
profile=Path(__file__).with_name('profile_probe.py').read_text()
measurement=Path(__file__).with_name('ult_measure.py').read_text()
script=(f'SOURCE={source!r}\nGENERATED={generated!r}\nMUTANTS={mutants!r}\n'
        f'FUT="choose"\nPROFILE_SOURCE={profile!r}\n'+measurement)
result=execute(script,timeout=20)
assert result['passed'],result
metrics=json.loads([line[len('__A1_ULT__'):] for line in result['stdout'].splitlines() if line.startswith('__A1_ULT__')][-1])
assert metrics['submitted_tests']==3 and metrics['valid_tests']==2,metrics
assert metrics['mutants']==metrics['killed_mutants']==2,metrics
assert metrics['coverage']['branch_total']==metrics['coverage']['branch_covered']==2,metrics
(ROOT/'artifacts/setup/study-evaluation-check.json').write_text(json.dumps({'passed':True,'metrics':metrics},indent=2)+'\n')
print('PASS: invalid assertion rejected; both known mutants killed; both branches covered')
