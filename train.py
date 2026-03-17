"""
Autoresearch RL training script. Single-GPU, single-file PPO on Atari.
Usage: uv run train.py
"""

import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, ENV_ID, make_env, evaluate_return

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

NUM_ENVS = 24            # number of parallel environments (match CPU thread count)
NUM_STEPS = 128          # rollout length per update
LR = 2.5e-4             # learning rate
GAMMA = 0.99             # discount factor
GAE_LAMBDA = 0.95        # GAE lambda
CLIP_COEF = 0.1          # PPO clip coefficient
NUM_MINIBATCHES = 4      # number of minibatches per update
UPDATE_EPOCHS = 4        # number of PPO epochs per update
ENT_COEF = 0.01          # entropy bonus coefficient
VF_COEF = 0.5            # value function loss coefficient
MAX_GRAD_NORM = 0.5      # gradient clipping
HIDDEN_SIZE = 512        # hidden layer size

# ---------------------------------------------------------------------------
# Agent (NatureCNN + actor/critic heads)
# ---------------------------------------------------------------------------

class Agent(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flat size by doing a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_size = self.conv(dummy).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(flat_size, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.actor = nn.Linear(HIDDEN_SIZE, num_actions)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        """Returns action logits (for evaluation compatibility)."""
        features = self.linear(self.conv(x))
        return self.actor(features)

    def get_value(self, x):
        features = self.linear(self.conv(x))
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.linear(self.conv(x))
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create vectorized environments
    env = make_env(ENV_ID, num_envs=NUM_ENVS)
    obs_shape = env.single_observation_space.shape
    num_actions = env.single_action_space.n
    print(f"Env: {ENV_ID}, obs_shape: {obs_shape}, num_actions: {num_actions}")
    print(f"NUM_ENVS: {NUM_ENVS}, NUM_STEPS: {NUM_STEPS}")

    batch_size = NUM_ENVS * NUM_STEPS
    minibatch_size = batch_size // NUM_MINIBATCHES

    # Create agent and optimizer
    agent = Agent(obs_shape, num_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Time budget: {TIME_BUDGET}s")

    # -----------------------------------------------------------------------
    # Rollout storage (obs stored as uint8 on CPU for memory efficiency)
    # -----------------------------------------------------------------------

    obs_buf = torch.zeros((NUM_STEPS, NUM_ENVS, *obs_shape), dtype=torch.uint8)
    actions_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long)
    logprobs_buf = torch.zeros((NUM_STEPS, NUM_ENVS))
    rewards_buf = torch.zeros((NUM_STEPS, NUM_ENVS))
    dones_buf = torch.zeros((NUM_STEPS, NUM_ENVS))
    values_buf = torch.zeros((NUM_STEPS, NUM_ENVS))

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    t_start_training = time.time()
    total_training_time = 0.0
    total_frames = 0
    num_updates = 0

    obs_np, _ = env.reset()
    next_obs = torch.from_numpy(np.asarray(obs_np))
    next_done = torch.zeros(NUM_ENVS)

    while True:
        t0 = time.time()

        # LR annealing based on time progress
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lr_now = LR * (1.0 - progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        # --- Rollout phase ---
        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                obs_gpu = next_obs.to(device=device, dtype=torch.float32) / 255.0
                action, logprob, _, value = agent.get_action_and_value(obs_gpu)

            actions_buf[step] = action.cpu()
            logprobs_buf[step] = logprob.cpu()
            values_buf[step] = value.flatten().cpu()

            obs_np, reward, term, trunc, info = env.step(action.cpu().numpy())
            next_obs = torch.from_numpy(np.asarray(obs_np))
            next_done = torch.from_numpy(np.asarray(term | trunc)).float()
            rewards_buf[step] = torch.from_numpy(np.asarray(reward, dtype=np.float32))

        # --- GAE computation ---
        with torch.no_grad():
            next_obs_gpu = next_obs.to(device=device, dtype=torch.float32) / 255.0
            next_value = agent.get_value(next_obs_gpu).flatten().cpu()

        advantages = torch.zeros((NUM_STEPS, NUM_ENVS))
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + GAMMA * nextvalues * nextnonterminal - values_buf[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

        returns = advantages + values_buf

        # --- PPO update phase ---
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds].to(device=device, dtype=torch.float32) / 255.0
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                # Advantage normalization
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Clipped surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(newvalue - mb_values, -CLIP_COEF, CLIP_COEF)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                # Fast fail: abort if loss is NaN
                if math.isnan(loss.item()):
                    print("FAIL")
                    exit(1)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        total_frames += batch_size
        num_updates += 1

        if num_updates > 2:
            total_training_time += dt

        # Logging
        pct_done = 100 * progress
        fps = int(batch_size / dt)
        remaining = max(0, TIME_BUDGET - total_training_time)
        print(f"\rupdate {num_updates:04d} ({pct_done:.1f}%) | loss: {loss.item():.4f} | pg: {pg_loss.item():.4f} | vf: {v_loss.item():.4f} | ent: {entropy_loss.item():.4f} | lr: {lr_now:.2e} | fps: {fps:,} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # Time's up — but only stop after warmup updates
        if num_updates > 2 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r training log

    env.close()

    # -------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------

    agent.eval()
    avg_return = evaluate_return(agent, device)

    # Final summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

    print("---")
    print(f"avg_return:       {avg_return:.2f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_frames_M:   {total_frames / 1e6:.1f}")
    print(f"num_updates:      {num_updates}")
