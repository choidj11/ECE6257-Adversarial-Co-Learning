#!/usr/bin/env python3
"""One example per benchmark: local generation -> isolated execution.

This checks input/output plumbing, not official benchmark scores. ULT runs K=1;
coverage/mutation and official full-suite evaluation belong to the experiment stage.
"""
from datetime import datetime, timezone
import ast
import json
import os
from pathlib import Path
import re

import pyarrow.parquet as pq
from sandbox_fusion import run_code, RunCodeRequest, RunStatus
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/week12"
OUT.mkdir(parents=True, exist_ok=True)
model = str(ROOT / "models/Qwen2.5-Coder-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
he = pq.read_table(next((ROOT / "data/HumanEval/openai_humaneval").glob("*.parquet"))).slice(0, 1).to_pylist()[0]
# Fixed lightweight stdlib example; /0 contains factorial-scale permutation tests
# and exceeded the short sandbox budget. Neither selection estimates benchmark scores.
bcb = pq.read_table(next((ROOT / "data/BigCodeBench/data").glob("v0.1.4*.parquet"))).slice(4, 1).to_pylist()[0]
assert bcb["task_id"] == "BigCodeBench/4"
ult = json.loads((ROOT / "data/UnLeakedTestBench/ULT_Lite.jsonl").read_text())[0]


def execute(code):
    response = run_code(RunCodeRequest(code=code, language="python", run_timeout=10),
                        endpoint=os.environ["SANDBOX_FUSION_ENDPOINT"],
                        max_attempts=1, client_timeout=20)
    return response.dict()


def bcb_test(code):
    return (code + "\n" + bcb["test"] + "\n"
            "if __name__ == '__main__':\n"
            "    unittest.main(exit=True)\n")


canonical = {
    "HumanEval": execute(he["prompt"] + he["canonical_solution"] + "\n" + he["test"] +
                         f"\ncheck({he['entry_point']})\n"),
    "BigCodeBench": execute(bcb_test(bcb["complete_prompt"] + bcb["canonical_solution"])),
}
assert all(r["status"] == "Success" for r in canonical.values()), canonical

requests = [
    ("HumanEval", he["task_id"], "Implement this Python function. Return only the complete function in a Python code block.\n" + he["prompt"]),
    ("BigCodeBench", bcb["task_id"], "Implement this Python function. Include required imports and return only Python code.\n" + bcb["instruct_prompt"]),
    ("UnLeakedTestBench", ult["task_id"], "Write exactly one standalone pytest test function for the Python function below. "
     "The function is already defined; do not repeat its implementation. Return only Python code.\n" + ult["code"]),
]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
           for _, _, prompt in requests]
llm = LLM(model=model, tokenizer=model, dtype="bfloat16", max_model_len=4096,
          max_num_seqs=1, enforce_eager=True, gpu_memory_utilization=0.2,
          disable_log_stats=True, tensor_parallel_size=1, seed=42)
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=512), use_tqdm=False)
results = []
for (benchmark, task_id, prompt), output in zip(requests, outputs):
    raw = output.outputs[0].text
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.S)
    candidate = "\n".join(blocks) if blocks else raw.strip()
    item = {"benchmark": benchmark, "task_id": task_id, "raw_output": raw,
            "prompt_tokens": len(output.prompt_token_ids),
            "generated_tokens": len(output.outputs[0].token_ids), "finish_reason": output.outputs[0].finish_reason}
    try:
        ast.parse(candidate)
        if benchmark == "HumanEval":
            code = candidate + "\n" + he["test"] + f"\ncheck({he['entry_point']})\n"
        elif benchmark == "BigCodeBench":
            code = bcb_test(candidate)
        else:
            tree = ast.parse(candidate)
            assert any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in tree.body), "No pytest test function"
            assert not any(isinstance(n, ast.FunctionDef) and n.name == ult["func_name"] for n in tree.body), "Generated test redefines FUT"
            source = ult["code"] + "\n" + candidate
            code = ("import pathlib, pytest\n" +
                    f"pathlib.Path('test_generated.py').write_text({source!r})\n" +
                    "raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', 'test_generated.py']))\n")
        item["execution"] = execute(code)
        item["sample_passed"] = item["execution"]["status"] == "Success"
    except (SyntaxError, AssertionError) as error:
        item["sample_passed"] = False
        item["candidate_error"] = f"{type(error).__name__}: {error}"
    results.append(item)
    print(f"{benchmark} {task_id}: sample_passed={item['sample_passed']}", flush=True)
report = {"checked_at": datetime.now(timezone.utc).isoformat(), "model": model, "seed": 42,
          "canonical_harness_checks": canonical, "results": results,
          "passed": all(bool(r["raw_output"].strip()) for r in results),
          "scope": "Infrastructure smoke only: 1 sample/benchmark, greedy 512 tokens, ULT K=1. Not official scores."}
(ROOT / "artifacts/setup/benchmark-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if report["passed"] else 1)
