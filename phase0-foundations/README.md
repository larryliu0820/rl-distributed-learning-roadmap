# Phase 0 — Foundations

**Goal:** Separate RL learning from distributed learning, so Phase 1 (GRPO) doesn't force you to debug both at once. Broken RL fails silently; broken distributed deadlocks mysteriously. Fluency in each, in isolation, is the prerequisite.

**Duration:** 1–2 weeks part-time, 4–5 days full-time.

**Hardware:** RTX 5080 for most of it. One evening on a rented 8xGPU node (~$25–50) for FSDP2 and 2D parallelism exercises.

## Tracks

Run these in parallel — they use different mental energy and reinforce each other.

### [Track A — RL fundamentals](./track-a-rl/)
REINFORCE → PPO, all on a single GPU or CPU. ~4–5 days.

### [Track B — Distributed fundamentals](./track-b-distributed/)
NCCL collectives → DDP → FSDP2 → 2D parallelism. ~4–5 days.

## Deliverable

A small repo of your own reference implementations:
- `reinforce.py` — ~80 lines, CartPole
- `ppo_modified.py` — your fork of CleanRL's `ppo.py`
- `collectives_demo.py` — all_reduce, all_gather, reduce_scatter, broadcast, all_to_all
- `ddp_manual.py` — DDP implemented by hand with all_reduce
- `fsdp_transformer.py` — small Transformer with `fully_shard`
- `tp_fsdp_2d.py` — 2D DeviceMesh with TP + FSDP

~1500 lines total. This is your personal reference library for Phases 1–3.

## Success criteria

After Phase 0, you should be able to answer without looking anything up:

**RL:**
- Why does the policy gradient contain `log_prob` and not `prob`?
- What does GAE compute, and what does λ control?
- Why does PPO clip the ratio instead of using a hard KL constraint?
- What's the difference between on-policy and off-policy, and why does PPO need an importance ratio?

**Distributed:**
- Walk through which NCCL collectives fire during one DDP step vs one FSDP step.
- Why is FSDP's per-step communication ~1.5x DDP's, and when is it still worth it?
- What's a DTensor, and what does `Shard(0)` mean operationally?
- Why does TP need fast intra-node interconnect but DP/FSDP tolerates slower links?
