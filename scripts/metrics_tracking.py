"""Preserve full-precision trainer metrics locally for asynchronous W&B upload."""
import json
from pathlib import Path

from verl.utils.tracking import Tracking


class MetricsTracking(Tracking):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        config=kwargs.get('config') or {}
        directory=(config.get('execution_feedback') or {}).get('run_dir')
        self.metric_file=Path(directory)/'metrics-stream.jsonl' if directory else None

    def log(self,data,step,backend=None):
        if self.metric_file:
            with self.metric_file.open('a') as output:
                output.write(json.dumps({'step':step,'metrics':data},default=float)+'\n')
        return super().log(data,step,backend)
