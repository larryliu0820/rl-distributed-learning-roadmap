"""
B1 — NCCL Collectives by Hand
==============================

Goal: Build intuition for the primitives every distributed framework is built on.

Run with:
    torchrun --nproc_per_node=4 collectives_demo.py

Notes:
- `torchrun` spawns N processes. Each gets a unique RANK (0..N-1) and
  shares WORLD_SIZE = N. They communicate via a backend (we use NCCL on GPU).
- On a single GPU, all N processes share that GPU. Timing won't be
  representative, but the semantics are correct. For real timing, use a
  multi-GPU box.
- `dist.barrier()` is your friend for ordering print statements deterministically.
"""

import os
import time
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def setup():
    """
    TODO:
    - Call dist.init_process_group(backend="nccl")
      (Use "gloo" if you're testing on CPU.)
    - Read LOCAL_RANK from env, use it to set torch.cuda.set_device(local_rank).
      LOCAL_RANK is the rank within the node; RANK is the global rank.
      On a single node they're equal.
    - Return (rank, world_size, device).

    Reference:
      https://docs.pytorch.org/docs/stable/distributed.html#initialization
    """
    pass


def cleanup():
    """TODO: Call dist.destroy_process_group()."""
    pass


def rank0_print(msg):
    """Print only on rank 0 to avoid spam from N processes."""
    if dist.get_rank() == 0:
        print(msg, flush=True)


def ordered_print(msg):
    """Print from each rank in order. Useful for inspecting per-rank tensors."""
    world_size = dist.get_world_size()
    for r in range(world_size):
        if dist.get_rank() == r:
            print(f"[rank {r}] {msg}", flush=True)
        dist.barrier()


# ---------------------------------------------------------------------------
# Collective 1: all_reduce
# ---------------------------------------------------------------------------

def demo_all_reduce(rank, world_size, device):
    """
    all_reduce: every rank contributes a tensor; every rank ends up with the
    reduced result (sum, max, min, etc.).

    Mental model: gradient averaging in DDP is one big all_reduce per param.

    TODO:
    - Create a tensor of shape (4,) on `device` whose values depend on rank.
      e.g. torch.tensor([rank, rank, rank, rank], dtype=torch.float32, device=device)
    - Print "before" on each rank using ordered_print.
    - Call dist.all_reduce(tensor, op=dist.ReduceOp.SUM).
    - Print "after". Every rank should now have the sum: [0+1+2+3, ...] for ws=4.
    - Try also dist.ReduceOp.MAX.

    Reference:
      https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce
    """
    rank0_print("\n=== all_reduce ===")
    pass


# ---------------------------------------------------------------------------
# Collective 2: all_gather
# ---------------------------------------------------------------------------

def demo_all_gather(rank, world_size, device):
    """
    all_gather: every rank contributes a tensor; every rank ends up with a
    list of all tensors (no reduction, just concatenation across ranks).

    Mental model: collecting per-rank predictions for an aggregated metric.

    TODO:
    - Create a per-rank tensor, e.g. torch.tensor([rank * 10], device=device).
    - Allocate a list of `world_size` empty tensors of the same shape.
      gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
    - Call dist.all_gather(gathered, local_tensor).
    - Print the gathered list on each rank — they should all be identical.

    Reference:
      https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather
    """
    rank0_print("\n=== all_gather ===")
    pass


# ---------------------------------------------------------------------------
# Collective 3: reduce_scatter
# ---------------------------------------------------------------------------

def demo_reduce_scatter(rank, world_size, device):
    """
    reduce_scatter: every rank contributes a list of tensors (one per rank).
    The collective reduces element-wise across ranks, then scatters the result
    so that rank-i gets chunk-i of the reduced output.

    Mental model: this is what FSDP does to gradients. Each rank ends up
    holding only its shard of the reduced gradient — never the full thing.

    Identity to remember:
        all_reduce  ==  reduce_scatter  +  all_gather
    DDP does the left side. FSDP decomposes it into the right side, which is
    why FSDP's peak memory is lower (the full gradient never materializes
    on any single rank).

    TODO:
    - Build an input list on each rank: a list of `world_size` tensors,
      each of shape (2,). Make values depend on (rank, chunk_idx) so you can
      verify the math afterwards.
        input_list = [torch.tensor([rank, chunk_idx], device=device, dtype=torch.float32)
                      for chunk_idx in range(world_size)]
    - Allocate output of shape (2,) on each rank.
    - Call dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM).
    - Verify: rank-i's output should be the sum across ranks of input_list[i].

    Reference:
      https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter
    """
    rank0_print("\n=== reduce_scatter ===")
    pass


# ---------------------------------------------------------------------------
# Collective 4: broadcast
# ---------------------------------------------------------------------------

def demo_broadcast(rank, world_size, device):
    """
    broadcast: one rank (the source) sends a tensor; all other ranks receive
    it. Used for syncing initial weights, syncing RNG seeds, etc.

    TODO:
    - On rank 0, create a tensor with known values, e.g. torch.arange(4, device=device).
    - On all other ranks, create an empty tensor of the same shape.
    - Call dist.broadcast(tensor, src=0).
    - All ranks should now have rank-0's values.

    Reference:
      https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast
    """
    rank0_print("\n=== broadcast ===")
    pass


# ---------------------------------------------------------------------------
# Collective 5: all_to_all
# ---------------------------------------------------------------------------

def demo_all_to_all(rank, world_size, device):
    """
    all_to_all: each rank sends a different chunk to every other rank.
    Rank-i's j-th input chunk goes to rank-j's i-th output chunk.

    Mental model: this is the workhorse of expert parallelism (MoE) — tokens
    routed to different experts living on different ranks. Also used in some
    sequence-parallel implementations.

    TODO:
    - Build input list of `world_size` tensors. Encode (sender, receiver)
      in the values so you can verify routing.
        input_list = [torch.tensor([rank * 10 + dest], device=device, dtype=torch.float32)
                      for dest in range(world_size)]
      So rank-i's j-th tensor "addressed to rank-j" has value i*10 + j.
    - Allocate output_list of `world_size` empty tensors of the same shape.
    - Call dist.all_to_all(output_list, input_list).
    - On rank-r, output_list[s] should be the tensor sender-s addressed to r,
      i.e. value s*10 + r.

    Reference:
      https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all
    """
    rank0_print("\n=== all_to_all ===")
    pass


# ---------------------------------------------------------------------------
# Timing / bandwidth study
# ---------------------------------------------------------------------------

def benchmark_collectives(rank, world_size, device):
    """
    Measure how each collective scales with tensor size.

    TODO:
    - For sizes in [1KB, 16KB, 256KB, 4MB, 64MB, 1GB] (in float32 elements):
        - Create a tensor of that size on `device`.
        - Warm up: run the collective 5 times (NCCL needs warmup; cuDNN etc.).
        - torch.cuda.synchronize() before timing.
        - Time 20 iterations with time.perf_counter().
        - torch.cuda.synchronize() after.
        - Compute average latency and effective bandwidth (bytes / time).
    - Repeat for: all_reduce, all_gather, reduce_scatter, broadcast, all_to_all.
    - On rank 0, print a table or save to results.csv.

    Theoretical reference for ring all_reduce bandwidth cost:
        bytes_moved_per_rank ≈ 2 * (N - 1) / N * tensor_size
    So as N grows, per-rank bytes approaches 2 * tensor_size — that's the
    fundamental cost of all_reduce, regardless of how clever the implementation is.

    Reference:
      Sylvain Jeaugey, "Massively Scale Your Deep Learning Training with NCCL 2.4"
      https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
    """
    rank0_print("\n=== benchmark ===")
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rank, world_size, device = setup()
    rank0_print(f"World size: {world_size}, device: {device}")

    demo_all_reduce(rank, world_size, device)
    demo_all_gather(rank, world_size, device)
    demo_reduce_scatter(rank, world_size, device)
    demo_broadcast(rank, world_size, device)
    demo_all_to_all(rank, world_size, device)

    # benchmark_collectives(rank, world_size, device)  # Uncomment when ready

    cleanup()


if __name__ == "__main__":
    main()
