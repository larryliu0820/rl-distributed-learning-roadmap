# Phase 3 — Capstone

**Goal:** Pick one meaty problem to publish or write up. Open-ended.

**Prerequisites:** Phase 2 complete.

## Options

### Option 1 — Distributed checkpointing for RL state

The full RL training state includes: policy weights, reference model weights, optimizer state, replay-buffer or rollout-buffer equivalent, RNG state, and possibly vLLM KV cache state. Sharded across DP/TP and resumable across preemption.

Why it matters: cloud spot instances are 3–4x cheaper but get preempted. A robust checkpoint/resume story is the difference between $1000 and $4000 per experiment.

Reference: [PyTorch Distributed Checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

### Option 2 — Float8 / MXFP8 GRPO on Blackwell

TorchTitan has MXFP8 for dense and MoE models on Blackwell. No clean open-source combination of this with GRPO exists yet.

- Sanity-check on your 5080 (FP8-capable)
- Scale on rented H100/B200
- Write up: loss curves, throughput, numerical stability issues, what breaks

References:
- [TorchAO FP8 docs](https://github.com/pytorch/ao)
- [Unsloth FP8 RL blog](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning)

### Option 3 — Pipeline parallel for the rollout phase

Most RL frameworks pipeline-parallel the *training* step but replicate the rollout/generation step. PP-ing generation (decoder layers across stages with KV cache handoff) is harder and more novel.

Hardest of the three. Most portfolio upside.

## Output

Write it up as a blog post or arXiv note. You'll have the numbers and the code to make it real.
