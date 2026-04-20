# RL + Distributed Training Learning Roadmap

A hands-on, project-driven roadmap to learn **Reinforcement Learning** and **Distributed Training** in PyTorch, culminating in a distributed GRPO implementation for LLMs.

## Background & motivation

Built for someone with a PyTorch internals background (e.g., ExecuTorch, `torch.export`) who wants to branch into two adjacent areas:
- **Reinforcement Learning**, particularly RL for LLMs (PPO, GRPO)
- **Distributed training**, particularly PyTorch-native (FSDP2, TP, PP, DTensor)

Hardware assumptions:
- **Local:** RTX 5080 (16GB VRAM, Blackwell)
- **Cloud:** Occasional 4–8x H100 node (Lambda / Runpod, ~$24/hr)

## Roadmap at a glance

| Phase | Focus | Duration | Hardware |
|-------|-------|----------|----------|
| **0** | Foundations: REINFORCE, PPO, NCCL, DDP, FSDP2, 2D parallelism | 1–2 weeks | 5080 + 1 evening rented 8xGPU |
| **1** | GRPO on a small LLM: Unsloth → hand-rolled | 1–2 weeks | 5080 |
| **2** | Distributed RL: nanochat RL stage + FSDP2/TP, or RL2/verl study | 2–3 weeks | 8xH100 (cloud) |
| **3** | Capstone: distributed checkpointing for RL, or MXFP8 GRPO on Blackwell, or pipeline-parallel rollouts | Open-ended | Mix |

## Phase 0 — Foundations

Two parallel tracks, run concurrently:

- **[Track A: RL fundamentals](./phase0-foundations/track-a-rl/)** — REINFORCE → PPO from scratch
- **[Track B: Distributed fundamentals](./phase0-foundations/track-b-distributed/)** — NCCL → DDP → FSDP2 → 2D parallelism

See [phase0-foundations/README.md](./phase0-foundations/README.md) for the full walkthrough.

## Phase 1 — GRPO on a small LLM (local)

See [phase1-grpo-local/README.md](./phase1-grpo-local/README.md).

## Phase 2 — Distributed RL (cloud)

See [phase2-distributed-rl/README.md](./phase2-distributed-rl/README.md).

## Phase 3 — Capstone

See [phase3-capstone/README.md](./phase3-capstone/README.md).

## Key reference repos

| Repo | Role |
|------|------|
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | Spine for Phase 2. ~8K lines, full LLM pipeline with deliberately-minimal RL stage |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | Distributed reference. FSDP2 + TP + PP + CP composition |
| [ChenmienTan/RL2](https://github.com/ChenmienTan/RL2) | Readable LLM-RL library, "Ray Less RL" |
| [verl-project/verl](https://github.com/verl-project/verl) | Production-grade RL infra reference |
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | Single-file deep RL (PPO, DQN, etc.) for Phase 0 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | Local accelerator for Phase 1; FP8 GRPO on consumer GPUs |

## Key reading

- [Spinning Up in Deep RL — Intro & Kinds of Algorithms](https://spinningup.openai.com/) (Parts 1 & 2)
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [TorchTitan paper](https://arxiv.org/abs/2410.06511)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)

## License

Personal learning repo. Code MIT, notes CC-BY-4.0.
