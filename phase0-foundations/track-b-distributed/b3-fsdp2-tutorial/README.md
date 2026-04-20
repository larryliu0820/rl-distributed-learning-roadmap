# B3 — FSDP2 Tutorial

**Goal:** Understand per-parameter sharding via DTensor. ~1 day. Needs rented multi-GPU.

## Setup

Rent a single 8xA100 or 8xH100 node on Lambda or Runpod for one evening. ~$25–50.

## Walkthrough

Follow [Getting Started with FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

Core pattern:

```python
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy

model = Transformer()
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# Inspect sharding
from torch.distributed.tensor import DTensor
for param in model.parameters():
    assert isinstance(param, DTensor)
    # param.placements will be (Shard(0),)
    # param.to_local() shows the rank's shard
```

## Experiments

1. Baseline: single-GPU training of a small Transformer (say 100M params). Measure peak memory.
2. FSDP2 with shard_degree=2, 4, 8. Watch memory drop roughly linearly.
3. Add `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`. Memory drops again.
4. Add `CPUOffloadPolicy()`. Memory drops much further, throughput drops too.
5. Toggle `reshard_after_forward` between default / always / never. Understand the tradeoff.

## Why your ExecuTorch background helps

DTensor is a Tensor subclass with sharding metadata; the dispatcher handles the rest. `fully_shard` registers forward/backward hooks that all-gather params before compute and reshard after. This is familiar territory if you've spent time in PyTorch's dispatcher.

## Reading

- [FSDP2 RFC / API docs](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [TorchTitan paper](https://arxiv.org/abs/2410.06511) sections on FSDP2
