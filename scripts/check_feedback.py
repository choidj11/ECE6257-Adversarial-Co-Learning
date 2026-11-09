#!/usr/bin/env python3
import json
from pathlib import Path
from execution_feedback import collect_feedback, format_feedback
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[1]
code='def choose(x):\n    if x > 0:\n        return x + 1\n    return -1\n'
half=collect_feedback(code,['assert choose(2) == 3'])
full=collect_feedback(code,['assert choose(2) == 3','assert choose(0) == -1'])
assert half['branch_covered'] < half['branch_total'],half
assert full['branch_covered']==full['branch_total'],full
assert half['snapshots'][0]['return']==3,half
tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen2.5-Coder-1.5B-Instruct',local_files_only=True)
formatted={}
for variant in ('E1','E2','E3'):
    text,tokens=format_feedback(half,variant,tokenizer)
    assert tokens<=384
    assert ('missing_arcs' in text)==(variant in ('E1','E3'))
    assert ('runtime_values' in text)==(variant in ('E2','E3'))
    assert 'assert choose' not in text
    formatted[variant]={'text':text,'tokens':tokens}
report={'passed':True,'partial':half,'full':full,'formats':formatted}
(root/'artifacts/setup/feedback-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
