# Phase 1 — GRPO on a small LLM (local)

**Goal:** Make GRPO real on your hardware. Start with Unsloth, then strip away abstraction until you've implemented GRPO yourself. 1–2 weeks.

**Hardware:** RTX 5080 (16GB VRAM, Blackwell).

**Prerequisites:** Phase 0 complete. You should already understand PPO's clipped surrogate objective and importance-sampling ratio.

## Step 1 — Run Unsloth GRPO on Qwen-1.5B / GSM8K

Unsloth has FP8 GRPO support for Blackwell. Qwen3-1.7B FP8 GRPO fits in ~5GB VRAM, so 16GB is very comfortable.

- Start from [Unsloth's GRPO notebook](https://huggingface.co/learn/llm-course/en/chapter12/6)
- Model: `Qwen/Qwen2.5-1.5B-Instruct` or `google/gemma-3-1b-it`
- Dataset: GSM8K
- Enable `load_in_fp8=True` if using Unsloth-latest on Blackwell
- Run for 250 steps
- Watch the reward plots: format-reward converges first, then correctness-reward lifts

## Step 2 — Read TRL's GRPOTrainer source

Source: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

Map each concept:
- [ ] Group sampling (n generations per prompt)
- [ ] Advantage = (reward - group_mean) / group_std
- [ ] KL penalty against the frozen reference model
- [ ] Importance-sampling ratio in the policy loss
- [ ] Per-token vs per-sequence loss masking

Compare mentally to CleanRL's `ppo.py`. What's the same? What's missing (hint: no value function)?

## Step 3 — Hand-rolled GRPO (~300 lines)

Reimplement against a tiny Qwen model, without TRL.

Components:
- Rollout engine: vLLM (fast KV-cache inference). For each prompt, sample N=8 completions.
- Reward function: format reward (XML tags) + correctness reward (parse numeric answer, match against GSM8K gold).
- Advantage: z-score normalize rewards within each group.
- Policy update: clipped PPO loss + KL penalty vs frozen reference, applied only on completion tokens.
- Reference model: frozen copy of initial policy, kept in memory.
- Optimizer: AdamW with lr~5e-6 (LLM RL needs very low LR).

Suggested file structure:

```
grpo_from_scratch/
├── rollout.py       # vLLM wrapper, generates N completions per prompt
├── reward.py        # format + correctness rewards
├── grpo.py          # main training loop
├── models.py        # policy + reference model loading
└── train.py         # CLI entry point
```

This is the moment it clicks. You'll also need this understanding for Phase 2.

## Step 4 (stretch) — Custom reward

Swap GSM8K for:
- Code-execution reward on a small Python eval (run the generated code in a subprocess, match stdout)
- Letter-counting (known toy task in the verl/RL2 ecosystem — "count the number of r's in 'strawberry'")

## Gotchas on 5080

- 16GB is tight for 7B+ models even with LoRA. Stick to 1.5B–3B for full fine-tune, or 7B with Unsloth FP8 + LoRA.
- vLLM + training model share VRAM. Use `gpu_memory_utilization=0.5` or less on vLLM side.
- Gradient accumulation is your friend.

## References

- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [DeepSeek-R1 paper (GRPO origin)](https://arxiv.org/abs/2501.12948)
