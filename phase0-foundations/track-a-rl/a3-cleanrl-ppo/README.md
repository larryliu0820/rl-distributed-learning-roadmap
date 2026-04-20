# A3 — CleanRL PPO

**Goal:** Internalize every trick that makes PPO actually work. 1–2 days.

## Setup

```bash
git clone https://github.com/vwxyzjn/cleanrl
cd cleanrl
uv sync  # or pip install -r requirements/requirements.txt
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 50000
```

Should solve CartPole in under a minute on your 5080.

## Read alongside

["The 37 Implementation Details of Proximal Policy Optimization"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — this paper is the difference between "I read the PPO paper" and "I can implement PPO."

## Checklist: concepts you should internalize

- [ ] Vectorized environments (`SyncVectorEnv`)
- [ ] Generalized Advantage Estimation (GAE), and what λ controls
- [ ] The clipped surrogate objective and why it's a lower bound
- [ ] Importance-sampling ratio `exp(log_prob_new - log_prob_old)`
- [ ] KL-divergence early stopping
- [ ] Value-loss clipping
- [ ] Advantage normalization per minibatch
- [ ] Orthogonal weight init + layer-wise learning-rate scaling
- [ ] Entropy bonus for exploration
- [ ] Gradient clipping by global norm

Roughly 30 of the 37 details transfer directly to GRPO/PPO-on-LLMs.

## Stretch

- Run on `LunarLander-v2` and `HalfCheetah-v4` (needs MuJoCo)
- Profile with PyTorch profiler — where's the time going?
- Compare `num_envs` values and plot sample efficiency vs wall-clock
