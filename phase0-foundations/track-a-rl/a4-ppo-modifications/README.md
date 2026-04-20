# A4 — Modify CleanRL PPO

**Goal:** Break PPO and fix it. ~1 day.

## Pick one

### Option 1 — PPO on LunarLander-Continuous

- Swap the discrete-action head for a diagonal Gaussian
- Add `tanh` squashing if needed
- Retune: lr, num_envs, num_steps, entropy coefficient
- Target: solve LunarLanderContinuous-v2 (avg reward > 200)

### Option 2 — DQN on CartPole from scratch

Starting from `ppo.py`'s single-file template, write `dqn.py`:
- Replay buffer (10k transitions)
- Target network with hard or soft updates
- ε-greedy exploration schedule
- Bellman loss: `MSE(Q(s,a), r + γ·max_a' Q_target(s',a'))`
- Target: solve CartPole in < 500 episodes

## Why this matters

RL fluency is "I know which plot to look at when reward goes flat." You only get that by breaking things and debugging.

## Common bugs (both options)

- Reward curve is flat: check advantage normalization, check that rollouts are actually being collected
- NaN loss: gradient explosion, add clipping
- Learns then unlearns: KL too large, reduce `update_epochs` or clip range
- DQN Q-values explode: target network not being updated correctly, or γ too close to 1
