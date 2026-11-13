# LFM3-1.7B Inference Evaluation

Model: `LiquidAI/LFM3-1.7B` at revision `1da06bec6a819da10b54c67e5c90972b46d81871`.

The [aggregate results and protocol](report/lfm3/summary.json) are available as JSON. Reproduction scripts and pinned inference dependencies are included in this repository; raw generations, per-task records, datasets, and weights remain local.
The original checkpoint serves both roles. There is no fine-tuning, RL, parameter update, candidate reranking, or inference-time code repair. Coder and tester are scored on their respective benchmarks; the result is not a jointly trained system score.

HumanEval uses the original OpenAI test set, not HumanEval+. BigCodeBench uses the full v0.1.4 Instruct variant. ULT Lite uses the existing 200-task bounded adaptation.

## Coder results

| Stage | Benchmark | Passed / total | Raw pass@1 | Reference-qualified pass@1 |
| --- | --- | ---: | ---: | ---: |
| 128-task feasibility | HumanEval | 22/32 | 68.75% | 68.75% (32 tasks) |
| 128-task feasibility | BigCodeBench | 18/64 | 28.13% | 29.03% (62 tasks) |
| Full | HumanEval | 106/164 | 64.63% | 64.63% (164 tasks) |
| Full | BigCodeBench | 290/1140 | 25.44% | 26.49% (1087 tasks) |

The feasibility subset contains 32 HumanEval, 64 BigCodeBench, and 32 ULT Lite tasks. Full evaluation includes all 164, 1140, and 200 tasks respectively, reusing feasibility outputs/results. Raw scores retain every task in the denominator. Qualified scores exclude tasks whose reference solution failed in the unchanged sandbox calibration.

## Tester results

| Stage | Tasks measured / total | Valid assertions / submitted | Valid / query | Mean branch coverage | Bounded mutation score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128-task feasibility | 32/32 | 47/265 | 0.7344 | 17.52% | 18.75% (24/128) |
| Full | 197/200 | 383/1717 | 0.9575 | 21.74% | 22.99% (180/783) |

Tester evaluation makes two queries per function, requesting up to eight assertions per query. Incorrect assertions are rejected against the reference function. Coverage uses valid assertions; fault detection uses at most four deterministic AST mutants per function. This is the existing bounded ULT Lite adaptation, not the official full Cosmic Ray score. Measurement failures are retained in the report rather than silently dropped.

Full evaluation attempted all 200 functions; 3 whole-function measurements reached the 120-second execution limit. Assertion and mutation totals use the 197 measured functions. Coverage means and per-query counts keep all 200 functions in their denominators, with zero contribution from failed measurements. Full assertion validity is 22.31% (383/1717).

## Protocol and resources

- Seed 42; BF16; Transformers/SDPA; context cap 16384 tokens; output cap 2048 tokens.
- Sampling: temperature 0.1, top_k 50, top_p 1.0, repetition penalty 1.1; thinking requested off through the official chat template.
- Full-run responses reaching the output cap: coder 127/1304; tester 53/400. Truncations remain in the score denominators.
- One coder response per task. A single sampled response is reported as pass@1; repeated-sampling uncertainty is not estimated.
- Generation allocation: 0.8731 GPU-hours; job details in `artifacts/lfm3-evaluation/report-evidence.json`.
- Feasibility started at batch 8 and resumed at batch 32 after resource monitoring. A full-run batch-64 attempt was stopped after slow initial progress, and full inference resumed at the verified batch 32 while retaining outputs. This was not a controlled steady-state throughput comparison. Cancelled GPU allocation time is included. Sampling randomness depends on batching and process restarts; retained outputs and the allocation history are recorded locally.
- W&B reporting is disabled for this evaluation. The report and aggregate metrics are versioned in this repository; raw generations and per-task results stay local under ignored `artifacts/lfm3-evaluation/`.
- User-authorized Qwen cleanup removed 55.97 GiB by file size, including the original checkpoint cache and eight fine-tuned coder/tester checkpoint trees. Historical metrics, outputs, and reports remain available; trained Qwen weights no longer exist locally.
- The earlier Qwen study used different sampling/output settings. Its results are historical context, not a controlled model comparison.

## Local evidence

- `protocol.json`, `model-revision.json`, and `custom-code-sha256.json` under `artifacts/lfm3-evaluation/`.
- `small/metrics.json`, `small/gate.json`, `full/metrics.json`, `report-evidence.json`, and `final-verification.json` in that directory.
- Weight removal manifest: `artifacts/setup/qwen-weight-cleanup.json`.
