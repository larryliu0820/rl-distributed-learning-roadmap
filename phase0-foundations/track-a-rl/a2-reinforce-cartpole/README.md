# A2 — REINFORCE on CartPole

**Goal:** Feel the basic RL loop in your hands. ~½ day. ~80 lines.

## Spec

- Environment: `gymnasium.make("CartPole-v1")`
- Policy: 2-layer MLP (hidden 64–128), softmax over 2 actions
- Loop:
  1. Roll out one episode with the current policy
  2. Compute discounted returns G_t for each step (γ = 0.99)
  3. Loss: `-(log_prob * G_t).sum()` summed over the trajectory
  4. `.backward()`, optimizer step
- Optimizer: Adam, lr=1e-2

## Success criteria

- Reaches reward 200+ (near-optimal) within 500 episodes
- You can explain every line

## Stretch

Add a value-function baseline: a second head that predicts V(s), trained with MSE. Use `A_t = G_t - V(s_t)` as the weight instead of `G_t`. Watch the variance of loss drop noticeably — this is why actor-critic exists.

## Common bugs

- Forgot to detach returns from the graph → spurious gradients
- Used `prob` instead of `log_prob` → gradient has wrong scale
- Normalized returns before computing loss, then wondered why learning is slow (normalization is fine, but understand that it changes the optimal learning rate)
- Didn't zero grads between episodes
