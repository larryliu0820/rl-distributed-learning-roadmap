# Track A — RL Fundamentals

All exercises run on your 5080 or even CPU. ~4–5 days.

## A1 — Read, don't code yet (½ day)

Read [Spinning Up "Key Concepts in RL"](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) and ["Kinds of RL Algorithms"](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html). Still the cleanest written intro to states/actions/returns/policies/value functions and the policy-gradient derivation.

Skip Part 3 unless you're curious. The repo itself is in maintenance mode — read the docs, ignore the code.

## [A2 — REINFORCE on CartPole, from scratch (½ day)](./a2-reinforce-cartpole/)

~80 lines. No libraries beyond `gymnasium` and `torch`.

## [A3 — CleanRL `ppo.py`, read and run (1–2 days)](./a3-cleanrl-ppo/)

The single best PyTorch RL learning artifact that exists. ~340 lines in one file.

## [A4 — Modify CleanRL PPO (1 day)](./a4-ppo-modifications/)

Pick one: swap CartPole for LunarLander-Continuous, or implement DQN on CartPole from the same single-file template. Breaking and fixing is the point.

## A5 — TorchRL PPO tutorial (optional)

[Official PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html). Useful for context on how a real library hides the same machinery you just wrote.

## Why this sequence

By the end, GRPO in Phase 1 should read as "PPO without the value function, with rewards normalized within a sampled group." About 30 of the 37 PPO implementation details carry over to GRPO/PPO-on-LLMs verbatim.
