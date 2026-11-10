"""Embedded in an isolated execution request with SOURCE and TESTS supplied.

Coverage traces branch arcs; the independent profiler records bounded function
return-time locals. Only the candidate's frames are exposed, never test frames.
"""
import coverage
import json
from pathlib import Path
import sys


def limited_value(value, depth=0):
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return value[:64] if kind is str else value
    if depth < 2 and kind in (list, tuple):
        return [limited_value(v, depth+1) for v in value[:6]]
    if depth < 2 and kind is dict:
        return {str(k)[:24]:limited_value(v, depth+1) for k,v in list(value.items())[:6]
                if type(k) in (str,int,float,bool)}
    return {'type':kind.__name__}


filename = str(Path('candidate.py').resolve())
Path(filename).write_text(SOURCE)
snapshots = []


def observe(frame, event, value):
    if event == 'return' and frame.f_code.co_filename == filename and frame.f_code.co_name != '<module>':
        if len(snapshots) < 16:
            snapshots.append({'function':frame.f_code.co_name, 'line':frame.f_lineno,
                              'locals':{k:limited_value(v) for k,v in list(frame.f_locals.items())[:6]
                                        if not k.startswith('_')},
                              'return':limited_value(value)})


cov = coverage.Coverage(branch=True, data_file=None, config_file=False, include=[filename])
cov.start()
sys.setprofile(observe)
outcomes=[]
for test in TESTS:
    namespace = {'__name__':'candidate'}
    try:
        exec(compile(SOURCE, filename, 'exec'), namespace)
        exec(compile(test, 'probe.py', 'exec'), namespace)
        outcomes.append('passed')
    except BaseException as error:
        outcomes.append(type(error).__name__)
sys.setprofile(None)
cov.stop()
try:
    analysis = cov._analyze(filename)
    stats = analysis.branch_stats()
    branch_lines = set(stats)
    result = {'branch_total':sum(v[0] for v in stats.values()),
              'branch_covered':sum(v[1] for v in stats.values()),
              'covered_arcs':[list(a) for a in analysis.arcs_executed if a[0] in branch_lines],
              'missing_arcs':[list(a) for a in analysis.arcs_missing() if a[0] in branch_lines],
              'line_total':len(analysis.statements),'line_covered':len(analysis.executed),
              'snapshots':snapshots,'probe_outcomes':outcomes}
except Exception as error:
    result={'error':type(error).__name__,'snapshots':snapshots,'probe_outcomes':outcomes}
print('__A1_FEEDBACK__'+json.dumps(result, allow_nan=False, default=str))
