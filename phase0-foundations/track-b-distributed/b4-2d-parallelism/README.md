# B4 — 2D Parallelism: TP + FSDP

**Goal:** Understand why TP and FSDP compose — they shard along orthogonal dimensions. ~1 day. Needs rented multi-GPU.

## Spec

Build a small FeedForward block:

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

Shard it across an 8-GPU node as 2D: DP=4, TP=2.

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module,
)
from torch.distributed.fsdp import fully_shard

mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh["tp"]
dp_mesh = mesh["dp"]

parallelize_module(model, tp_mesh, {
    "w1": ColwiseParallel(),
    "w3": ColwiseParallel(),
    "w2": RowwiseParallel(),
})
fully_shard(model, mesh=dp_mesh)
```

## The insight

- **TP splits a matmul** *within* a layer. Colwise then Rowwise cancels the all-reduce nicely (output of Colwise is partial, Rowwise reduces it).
- **FSDP splits parameters** *across* the "storage" dimension: each rank owns a slice of every parameter.
- Because they shard different dimensions, they compose cleanly.

## Exercises

1. Run with (DP=8, TP=1), (DP=4, TP=2), (DP=2, TP=4), (DP=1, TP=8). Compare throughput and memory.
2. Profile: where does TP's extra all-reduce show up? Does it overlap with compute?
3. Read [TorchTitan's `parallelize.py` for Llama3](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/infra/parallelize.py) and match each call to what you just did.

## Why TP needs NVLink

TP all-reduces happen on the critical path of every forward/backward. If the interconnect is slow, compute idles waiting. This is why TP is usually confined to within a node (NVLink) while DP/FSDP can span nodes (IB/Ethernet).

## Reference

- [PyTorch Tensor Parallelism docs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Lightning 2D parallelism docs](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp_fsdp.html) (uses the same PyTorch APIs under the hood)
