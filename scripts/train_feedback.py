#!/usr/bin/env python3
"""Run the official entrypoint with a local trainer subclass, without editing it."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
upstream=root/'external/Code-A1/Code-A1/code/rl/main_a1.py'
sys.path.insert(0,str(upstream.parent))
source=upstream.read_text()
old='from trainer_a1 import ACGRayTrainer'
assert source.count(old)==1
source=source.replace(old,'from feedback_trainer import FeedbackTrainer as ACGRayTrainer')
exec(compile(source,str(upstream),'exec'),globals())
