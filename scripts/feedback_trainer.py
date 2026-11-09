"""Extend only the tester prompt; preserve the upstream Code-A1 reward/update."""
import json
from pathlib import Path
import time

from trainer_a1 import ACGRayTrainer
from execution_feedback import collect_batch, format_feedback
import trainer_a1
from metrics_tracking import MetricsTracking

# Training never waits for network logging. A CPU monitor uploads this stream.
trainer_a1.Tracking=MetricsTracking


class PromptTokenizer:
    def __init__(self, tokenizer, feedback):
        self.tokenizer=tokenizer
        self.feedback=iter(feedback)

    def __getattr__(self,name):
        return getattr(self.tokenizer,name)

    def __call__(self,*args,**kwargs):
        return self.tokenizer(*args,**kwargs)

    def apply_chat_template(self,messages,*args,**kwargs):
        messages=[dict(message) for message in messages]
        messages[-1]['content']+=next(self.feedback)
        return self.tokenizer.apply_chat_template(messages,*args,**kwargs)


class FeedbackTrainer(ACGRayTrainer):
    def _prepare_model_b_batch_phase2(self,batch,a_gen_code):
        settings=self.config_a.execution_feedback
        n=self.config_a.actor_rollout_ref.rollout.n
        probes=[]
        for tests in batch.non_tensor_batch['GT_test']:
            tests=list(tests)
            probes.extend([tests[:int(settings.probes)]]*n)
        traces=collect_batch(a_gen_code,probes)
        tokenizer=self.tokenizer_b
        formatted=[format_feedback(trace,settings.variant,tokenizer,int(settings.max_tokens))
                   for trace in traces]
        directory=Path(settings.run_dir)/'feedback'
        directory.mkdir(exist_ok=True)
        with (directory/f'step-{getattr(self,"global_steps",0)}.jsonl').open('a') as output:
            for code,trace,(text,tokens) in zip(a_gen_code,traces,formatted):
                output.write(json.dumps({'variant':settings.variant,'code':code,'trace':trace,
                                         'feedback':text,'input_tokens':tokens})+'\n')
        self.tokenizer_b=PromptTokenizer(tokenizer,[text for text,_ in formatted])
        try:
            result=super()._prepare_model_b_batch_phase2(batch,a_gen_code)
        finally:
            self.tokenizer_b=tokenizer
        return result
