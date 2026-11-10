"""Collect and format bounded execution feedback without exposing test oracles."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from sandbox_fusion import run_code, RunCodeRequest

PROFILER = Path(__file__).with_name('profile_probe.py').read_text()


def collect_feedback(code, probes):
    started=time.monotonic()
    script=f'SOURCE={str(code)!r}\nTESTS={list(map(str,probes))!r}\n'+PROFILER
    try:
        response=run_code(RunCodeRequest(code=script, language='python', run_timeout=10),
                          max_attempts=1, client_timeout=20)
        lines=(response.run_result.stdout or '').splitlines()
        messages=[line[len('__A1_FEEDBACK__'):] for line in lines if line.startswith('__A1_FEEDBACK__')]
        result=json.loads(messages[-1]) if messages else {'error':str(response.status)}
    except Exception as error:
        result={'error':type(error).__name__}
    result['execution_seconds']=time.monotonic()-started
    return result


def format_feedback(trace, variant, tokenizer, max_tokens=384):
    selected={}
    if variant in ('E1','E3'):
        selected.update({k:trace.get(k) for k in ('branch_total','branch_covered')})
        selected['covered_arcs']=trace.get('covered_arcs',[])[:12]
        selected['missing_arcs']=trace.get('missing_arcs',[])[:12]
    if variant in ('E2','E3'):
        selected['runtime_values']=trace.get('snapshots',[])[:4]
    if trace.get('error'):
        selected['collection_error']=trace['error']
    prefix='\n\nExecution feedback from fixed seed probes (line numbers refer to Buggy Code; expected test answers are withheld):\n'
    while True:
        text=prefix+json.dumps(selected,separators=(',',':'),ensure_ascii=True)
        tokens=len(tokenizer.encode(text,add_special_tokens=False))
        if tokens <= max_tokens:
            return text,tokens
        reducible=[key for key,value in selected.items() if isinstance(value,list) and value]
        if reducible:
            longest=max(reducible,key=lambda k:len(json.dumps(selected[k])))
            selected[longest].pop()
        else:
            return prefix+'{"collection_error":"feedback too long"}',len(tokenizer.encode(prefix))+12


def collect_batch(codes, probes):
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(collect_feedback,codes,probes))
