# Reading materials — Distributed basics

A short curated list. Don't try to read all of this before writing code; pick the first one or two, then come back to others as questions arise.

## Start here (concepts, ~1 hour)

- **[PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)** — Official entry point. Explains the relationship between `torch.distributed`, DDP, FSDP, RPC, and pipelining. Read the first half.

- **[Writing Distributed Applications with PyTorch](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)** — The canonical tutorial. Walks through `init_process_group`, point-to-point send/recv, then collectives. Code is dated in places but the concepts are exactly right. Skim, don't memorize — you'll write your own version in `collectives_demo.py`.

## Collectives, deeper (pick what's relevant)

- **[NVIDIA: Massively Scale Deep Learning Training with NCCL](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)** — Explains ring vs tree algorithms for all_reduce. The key insight: ring all_reduce moves `2(N-1)/N × tensor_size` bytes per rank, which is bandwidth-optimal. Tree algorithms have lower latency for small tensors.

- **[Horovod paper (Sergeev & Del Balso, 2018)](https://arxiv.org/abs/1802.05799)** — The paper that popularized ring all_reduce in deep learning. Section 3 ("Algorithm") is a clean explanation of why ring is bandwidth-optimal. Skip the rest.

- **[NCCL documentation: collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)** — Reference for what each collective actually does at the NCCL level. Useful when PyTorch's docs are too thin.

## DDP and FSDP context (will be used in B2 / B3)

- **[PyTorch DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html)** — Explains gradient bucketing and overlap with backward. The "Implementation" section shows exactly when each all_reduce fires.

- **[PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** — You'll do this in B3. Skim the "How FSDP works" diagram now so the `reduce_scatter + all_gather` decomposition has a visual to attach to.

- **[Hugging Face: Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many)** — Concise comparison of DDP, ZeRO/FSDP, TP, PP. Good for building the mental map.

## Going deeper (optional)

- **[The Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)** — Long-form interactive guide to scaling LLM training: data parallel, ZeRO, tensor parallel, sequence parallel, pipeline parallel, expert parallel. The single best modern resource if you want to go deeper after Phase 0.

- **[Megatron-LM paper (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)** — Original tensor parallelism paper. Section 3 (the math of column/row parallel) is the foundation for what you'll build in B4. Short and readable.

- **[GPipe paper (Huang et al., 2018)](https://arxiv.org/abs/1811.06965)** — Pipeline parallelism foundations. Read if you want to do Phase 3 Option 3 (PP for rollouts).

- **[ZeRO paper (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054)** — The DeepSpeed paper that motivated FSDP. Stages 1/2/3 correspond to sharding optimizer states, gradients, and parameters respectively. FSDP2 = ZeRO-3 done right in PyTorch.

## Mental models to internalize before B2

By the time you finish B1, the following should feel obvious, not memorized:

- `all_reduce = reduce_scatter + all_gather`. This is *the* identity. DDP does the left side per gradient. FSDP does the right side, with reduce_scatter on gradients (in backward) and all_gather on parameters (in forward). Total bytes moved is ~1.5x DDP, but peak memory is much lower because the full gradient never lives on one rank.

- **Why ring all_reduce is bandwidth-optimal**: each rank must send its data once and receive everyone else's once. Ring achieves exactly this with `2(N-1)/N × tensor_size` bytes per rank. You can't do better — that's the lower bound.

- **Latency vs bandwidth-bound regimes**: small tensors are dominated by the per-call latency (a few microseconds). Large tensors are dominated by bandwidth. NCCL switches algorithms (tree for small, ring for large) automatically.

- **Why TP wants NVLink**: TP all_reduces are on the critical path of every layer's forward and backward. Slow link = compute idles. DP/FSDP can tolerate slower links because their collectives are bucketed and overlapped with compute.
