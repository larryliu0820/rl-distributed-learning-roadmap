# B1 — NCCL Collectives by Hand

**Goal:** Build intuition for the primitives every distributed framework is built on. ~½ day. ~50 lines.

## Spec

Write `collectives_demo.py` that uses `torch.distributed` with the `nccl` backend. Launch with `torchrun --nproc_per_node=4 collectives_demo.py` (works on one GPU if N processes share it; for real timing you want N GPUs).

Implement and time each of:
- `all_reduce` (sum across ranks)
- `all_gather` (concat across ranks)
- `reduce_scatter` (reduce, then scatter one chunk per rank)
- `broadcast` (one rank → all)
- `all_to_all` (rank-i sends chunk-j to rank-j)

For each, print before/after tensors on each rank and time vs tensor size (1KB → 1GB).

## The key insight

```
all_reduce = reduce_scatter + all_gather
```

This single equation is *why FSDP exists*. DDP does one all_reduce per gradient. FSDP decomposes that into reduce_scatter (gradients → sharded gradients) + all_gather (sharded params → full params before forward). The total bytes moved is ~1.5x, but peak memory is much lower.

## Output

A small writeup in `results.md` with your timing plots. Which collective is the bandwidth bottleneck? Does it match the theoretical ring-allreduce cost (2(N-1)/N × tensor_size)?
