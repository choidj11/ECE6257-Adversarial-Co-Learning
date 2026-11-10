#!/usr/bin/env python3
"""Create the user-authorized metrics-only W&B project; never print the key."""
import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values
import wandb
from wandb.apis._generated import CREATE_PROJECT_GQL, UpsertModelInput

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--entity')
parser.add_argument('--project')
parser.add_argument('--config-file',type=Path,default=root/'artifacts/study/wandb-config.json')
args=parser.parse_args()
values=dotenv_values(root/'.env')
key=values.get('WANDB_API_KEY') or os.environ.get('WANDB_API_KEY')
assert key,'WANDB_API_KEY is missing from .env/environment'
os.environ['WANDB_API_KEY']=key
os.environ['WANDB_MODE']='online'
os.environ['WANDB_ERROR_REPORTING']='false'
api=wandb.Api(timeout=30)
entity=args.entity or values.get('WANDB_ENTITY') or api.default_entity
assert entity,'No default W&B entity found for the supplied account'
name=args.project or values.get('WANDB_PROJECT') or '6257-adversarial-co-training'
request=UpsertModelInput(name=name,entity_name=entity,access='PRIVATE',
                         description='6257 adversarial co-training: training/evaluation metrics and configurations only.')
api._service_api.execute_graphql(CREATE_PROJECT_GQL,{'input':request.model_dump(by_alias=True,exclude_none=True)})
project=api.project(name,entity=entity)
result={'authorized':True,'entity':entity,'project':project.name,'scope':'metrics',
        'url':project.url,'payload':['numeric training metrics','training configuration',
                                    'numeric resource monitoring','aggregate evaluation scores'],
        'excluded':['raw prompts','model outputs','per-example logs','datasets','model weights'],
        'authorization':'User explicitly requested project creation and metrics such as loss/eval score.'}
args.config_file.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
