# Phase 2 — Distributed RL (cloud)

**Goal:** Compose PyTorch-native distributed training with LLM RL. 2–3 weeks.

**Hardware:** 4–8 H100s on Lambda/Runpod. Budget $200–400 over a weekend for experiments.

**Prerequisites:** Phase 1 complete. You have a working GRPO implementation end-to-end.

## Two flavors — pick one

### Flavor A — nanochat + your own distributed layer

Take [nanochat](https://github.com/karpathy/nanochat)'s `scripts/chat_rl.py` (which uses simple torchrun + DDP) and upgrade it:

1. Replace DDP with FSDP2 (`fully_shard`) for the policy model
2. Keep a replicated reference model on a subset of ranks
3. Integrate vLLM as the rollout engine on dedicated GPUs (rollout/training disaggregation — this is what real RL infra does)
4. Implement weight syncing between the training model (DTensor-sharded) and the vLLM model (replicated)
5. Schedule alternating rollout and update phases

This is a portfolio-worthy project. Karpathy himself says nanochat's RL support is "in its infancy," which means you're not just running a tutorial — you're improving a real repo.

Milestones:
- [ ] FSDP2 policy runs end-to-end on nanochat's GSM8K setup
- [ ] Rollout throughput measured with/without disaggregation
- [ ] Weight sync working; policy updates reflected in next rollout
- [ ] Ablation: FSDP2 vs FSDP2+TP on a 1B model

### Flavor B — Study RL2 or verl

Read [RL2](https://github.com/ChenmienTan/RL2) end-to-end (smaller than verl, more readable). It supports FSDP DP/CP/TP composition.

1. Run a GRPO job on a 1.5B–7B model across 4–8 H100s
2. Modify the parallelism config: try different DP/TP combinations, measure throughput
3. Write up findings: when does TP help? When does it hurt?

## Recommendation

Flavor A, unless you're time-constrained. Given your `torch.export` / ExecuTorch background, you'll get more out of *building* the parallelism than configuring it.

## References

- [nanochat repo](https://github.com/karpathy/nanochat)
- [nanochat DeepWiki](https://deepwiki.com/karpathy/nanochat)
- [RL2 repo](https://github.com/ChenmienTan/RL2)
- [verl repo](https://github.com/verl-project/verl)
- [Rollout-training disaggregation blog posts](https://www.anyscale.com/blog/ray-rlhf) (various)
