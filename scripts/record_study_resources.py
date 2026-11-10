#!/usr/bin/env python3
"""Record this study's allocated GPU-hours from Slurm, including failures."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
study=root/'artifacts/study'
jobs=json.loads((study/'jobs.json').read_text())
ids=[]
for category in ('training','generation','evaluation'):
    ids.extend(jobs[category].values())
ids.extend([jobs['calibration'],*jobs.get('integration_checks',[])])
for attempt in jobs.get('attempts',[]):
    ids.append(attempt['job_id'])
ids=list(dict.fromkeys(ids))
response=subprocess.run(['sacct','-X','-P','-n','-j',','.join(map(str,ids)),
                         '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES'],
                        check=True,text=True,capture_output=True,timeout=30)
(study/'slurm-accounting.txt').write_text(response.stdout)
rows=[]
for line in response.stdout.splitlines():
    fields=line.split('|')
    if len(fields)<6: continue
    identifier,name,state,exitcode,elapsed,tres=fields[:6]
    match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
    gpus=int(match.group(1)) if match else 0
    rows.append({'job_id':identifier,'name':name,'state':state,'exitcode':exitcode,
                 'elapsed_seconds':int(elapsed),'allocated_gpus':gpus,
                 'gpu_hours':gpus*int(elapsed)/3600,'allocated_tres':tres})
report={'recorded_at':datetime.now(timezone.utc).isoformat(),'jobs':rows,
        'gpu_hours':sum(r['gpu_hours'] for r in rows),
        'scope':'Current frozen study jobs, not all historical account usage.'}
quota=subprocess.run([sys.executable,str(root/'scripts/check_quota.py')],
                     capture_output=True,text=True,timeout=45)
if quota.returncode not in (0,1):
    raise RuntimeError('Unable to read final storage quota')
report['scratch_quota']=json.loads(quota.stdout)
(study/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
