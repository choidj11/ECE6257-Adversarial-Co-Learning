# Code-A1 Implementation Review

Reviewed version: ZJU-REAL/Code-A1 commit `be86d9315bab10a97d006aa25eb84204b25aa129`. The entrypoint is `external/Code-A1/Code-A1/code/rl/main_a1.py`; the local baseline configuration is `artifacts/study/E0/config.yaml`. Source: [official repository](https://github.com/ZJU-REAL/Code-A1).

## Actual training path

1. `main_a1.py` creates separate coder and tester resource pools, initializes their actor/rollout and reference policies, and uses the official customized verl FSDP2 workers.
2. `trainer_a1.py::_dual_model_rollout` generates multiple code candidates per problem, then calls `_prepare_model_b_batch_phase2` to give the problem and each candidate to the tester, which generates assertions.
3. `reward.py` uses the Sandbox Fusion execution API to validate tests against the reference implementation, then evaluates candidate code against valid new tests and historical tests in the Mistake Book.
4. The trainer computes old-policy and reference-policy log probabilities, constructs GRPO advantages, and updates both actors with the official PPO clipped loss and KL loss.
5. Rollouts and the Mistake Book are saved. To control storage, this study saves only the final Hugging Face inference weights for both models.

## Rewards

The training data uses `data_source=kodcode_light`, the reward class registered as `acg_llm`, and `alpha=0.5`. For valid candidate code:

- The coder reward is the pass rate on new tests. If historical attack tests also exist, it is averaged with the pass rate on those historical tests.
- The tester validity component is `max(0, min(1, valid_test_count/8, 1-invalid_test_count/8))`.
- With no historical attack tests, the attack component is the number of failures on new tests divided by 8. When historical attacks exist, the official implementation combines this with the historical attack failure rate.
- The tester reward weights validity and attack components by `alpha`; non-printable output receives an additional penalty. Invalid code does not receive the normal validity/attack reward.

The trainer aggregates multiple tester rewards for each code candidate and retains tester groups with higher variance for updates. The Mistake Book stores valid tests that previously caused candidates to fail and replays them in later rounds. Describing this implementation simply as RL on the pass rate of official tests would omit these mechanisms.

## Preserved mechanisms and study deviations

The study preserves the rewards, Mistake Book, FSDP2, GRPO, dual-model updates, KL coefficient, learning rate, and base model. E0 runs the upstream entrypoint directly. E1–E3 extend only the tester prompt through a local subclass, without modifying upstream source files.

Resource limits reduce the upstream configuration of 3 epochs, batch size 256, n=8 per role, and 4 GPUs per role to 128 fixed problems, 32 steps, batch size 4, n=2 per role, and 1 GPU per role. The output cap is reduced from 4,096 to 1,024 tokens. Conclusions apply to this fixed-budget study; they do not claim reproduction of the paper's absolute scores.

## Validation evidence

Weeks 1–2 job 6888089 completed a full update with actor gradient norms of 2.010 and 2.977. `artifacts/setup/feedback-check.json` validates partial/full branch coverage and return-time locals for the execution feedback. `study-evaluation-check.json` validates rejection of an incorrect assertion, detection of known mutants, and complete branch coverage. The subsequent 32-step training and evaluation are supported by the corresponding jobs, checkpoints, per-task JSONL records, and final report.
