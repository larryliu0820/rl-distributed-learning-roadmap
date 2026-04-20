# Track B — Distributed Fundamentals

**Hardware note:** `torchrun --nproc_per_node=N` spawns N processes on a single machine and they share the GPU (or use CPU). That's fine for B1 and B2. For B3 and B4, rent a single 8xA100 or 8xH100 node for one evening (~$25–50). FSDP's memory savings only *feel* real when you have multiple devices.

## Exercises

- [B1 — NCCL collectives by hand](./b1-nccl-collectives/) (½ day)
- [B2 — DDP from scratch](./b2-ddp-from-scratch/) (1 day)
- [B3 — FSDP2 tutorial](./b3-fsdp2-tutorial/) (1 day, rented multi-GPU)
- [B4 — 2D parallelism: TP + FSDP](./b4-2d-parallelism/) (1 day, rented multi-GPU)

## B5 — Read torchtitan (optional, recommended)

Read [`torchtitan/models/llama3/infra/parallelize.py`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/infra/parallelize.py). Don't run it — just read. ~400 lines showing how a real codebase composes FSDP2 + TP + activation checkpointing + `torch.compile`. After B1–B4, this file should feel ~80% legible.

## What I'm deliberately not using

- Ray, Lightning, Accelerate, DeepSpeed. They're useful tools, but they hide the abstractions you're trying to learn. Stay in `torch.distributed` land for Phase 0.
