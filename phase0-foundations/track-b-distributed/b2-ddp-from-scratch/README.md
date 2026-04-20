# B2 — DDP From Scratch

**Goal:** Understand DDP as "just gradient averaging," then see how the real DDP optimizes this. ~1 day.

## Part 1 — Manual DDP

Train a small model (MLP on MNIST or ResNet-9 on CIFAR-10) with this loop on N ranks:

```python
# pseudocode
for batch in dataloader:  # each rank gets a different slice
    loss = model(batch).loss
    loss.backward()
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
    optimizer.step()
    optimizer.zero_grad()
```

Verify: loss curves match single-GPU training (with adjusted effective batch size).

## Part 2 — Real DDP

Swap to `torch.nn.parallel.DistributedDataParallel`. Same training loop, just `model = DDP(model, device_ids=[local_rank])`.

Observe:
- Throughput is higher. Why? DDP buckets gradients and overlaps all_reduce with the backward pass.
- Use PyTorch profiler or NVIDIA Nsight to see the overlap.

## Takeaways

- Data parallel = each rank processes a different minibatch, then gradients are averaged.
- All the engineering is in overlapping communication with computation.
- Effective batch size = per-rank batch × world size. Learning rate usually scales linearly with it (up to a limit).

## Reference

- [PyTorch DDP docs](https://docs.pytorch.org/docs/stable/notes/ddp.html)
