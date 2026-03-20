"""
Autoresearch RL training script. PPO + Self-Imitation Learning (SIL) + Polyak Averaging.
Best config after 170+ experiments: mean avg_return ~4.4, peak 5.53.
Usage: uv run train.py
"""

import time
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, ENV_ID, make_env, evaluate_return

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

NUM_ENVS = 32
NUM_STEPS = 64
LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 6
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
HIDDEN_SIZE = 256

# SIL: Self-Imitation Learning
SIL_BUFFER_SIZE = 4096
SIL_COEF = 0.5
SIL_BATCH = 256
SIL_EPOCHS = 1
EMA_DECAY = 0.95
CHANNEL_COMPRESS = False

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer for learned exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # Factorized noise
        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out', torch.zeros(out_features))
        # Init
        mu_range = 1 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(out_features))
        self.reset_noise()

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.noise_in.copy_(eps_in)
        self.noise_out.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.noise_out.unsqueeze(1) * self.noise_in.unsqueeze(0)
            bias = self.bias_mu + self.bias_sigma * self.noise_out
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class Agent(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        c, h, w = obs_shape
        if CHANNEL_COMPRESS:
            self.channel_compress = nn.Sequential(
                layer_init(nn.Conv2d(c, 2, 1, stride=1)),
                nn.ReLU(),
            )
            cnn_channels = 2
        else:
            self.channel_compress = None
            cnn_channels = c
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(cnn_channels, 16, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            if self.channel_compress is not None:
                dummy = self.channel_compress(dummy)
            flat_size = self.conv(dummy).shape[1]
        self.linear = nn.Sequential(
            layer_init(nn.Linear(flat_size, HIDDEN_SIZE)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(HIDDEN_SIZE, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0)
        # Forward dynamics: predict next features from (features, action)
        self.dynamics = nn.Sequential(
            layer_init(nn.Linear(HIDDEN_SIZE + num_actions, HIDDEN_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), std=0.01),
        )
        self.num_actions = num_actions

    def _encode(self, x):
        if self.channel_compress is not None:
            x = self.channel_compress(x)
        return self.linear(self.conv(x))

    def forward(self, x):
        return self.actor(self._encode(x))

    def get_value(self, x):
        return self.critic(self._encode(x))

    def get_action_and_value(self, x, action=None):
        features = self._encode(x)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)

# ---------------------------------------------------------------------------
# SIL Buffer
# ---------------------------------------------------------------------------

class SILBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.returns = torch.zeros(capacity)
        self.pos = 0
        self.size = 0

    def add(self, obs, actions, returns, advantages):
        mask = advantages > 0
        if not mask.any():
            return
        good_obs = obs[mask]
        good_actions = actions[mask]
        good_returns = returns[mask]
        n = min(good_obs.shape[0], self.capacity)
        good_obs = good_obs[:n]
        good_actions = good_actions[:n]
        good_returns = good_returns[:n]
        if self.pos + n <= self.capacity:
            self.obs[self.pos:self.pos + n] = good_obs
            self.actions[self.pos:self.pos + n] = good_actions
            self.returns[self.pos:self.pos + n] = good_returns
        else:
            first = self.capacity - self.pos
            self.obs[self.pos:] = good_obs[:first]
            self.actions[self.pos:] = good_actions[:first]
            self.returns[self.pos:] = good_returns[:first]
            rest = n - first
            self.obs[:rest] = good_obs[first:]
            self.actions[:rest] = good_actions[first:]
            self.returns[:rest] = good_returns[first:]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return None
        idxs = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return self.obs[idxs], self.actions[idxs], self.returns[idxs]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(ENV_ID, num_envs=NUM_ENVS)
    _obs_space = getattr(env, 'single_observation_space', env.observation_space)
    _act_space = getattr(env, 'single_action_space', env.action_space)
    obs_shape = _obs_space.shape
    num_actions = _act_space.n
    print(f"Env: {ENV_ID}, obs_shape: {obs_shape}, num_actions: {num_actions}")
    print(f"PPO + Self-Imitation Learning + Polyak Averaging")

    batch_size = NUM_ENVS * NUM_STEPS
    minibatch_size = batch_size // NUM_MINIBATCHES

    agent = Agent(obs_shape, num_actions).to(device)
    ema_agent = copy.deepcopy(agent)
    for p in ema_agent.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    sil_buffer = SILBuffer(SIL_BUFFER_SIZE, obs_shape)

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Time budget: {TIME_BUDGET}s")

    obs_buf = torch.zeros((NUM_STEPS, NUM_ENVS, *obs_shape), dtype=torch.uint8)
    # Pre-allocated float buffer for rollout inference
    obs_float_buf = torch.zeros((NUM_ENVS, *obs_shape), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long)
    logprobs_buf = torch.zeros((NUM_STEPS, NUM_ENVS))
    rewards_buf = torch.zeros((NUM_STEPS, NUM_ENVS))
    dones_buf = torch.zeros((NUM_STEPS, NUM_ENVS))
    values_buf = torch.zeros((NUM_STEPS, NUM_ENVS))

    t_start_training = time.time()
    total_training_time = 0.0
    total_frames = 0
    num_updates = 0

    ep_returns = []
    current_ep_returns = np.zeros(NUM_ENVS, dtype=np.float64)

    obs_np, _ = env.reset()
    next_obs = torch.from_numpy(np.asarray(obs_np))
    next_done = torch.zeros(NUM_ENVS)

    while True:
        t0 = time.time()
        progress = min(total_training_time / TIME_BUDGET, 1.0)

        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.inference_mode():
                obs_float_buf.copy_(next_obs)
                obs_float_buf.div_(255.0)
                action, logprob, _, value = agent.get_action_and_value(obs_float_buf)

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            obs_np, reward, term, trunc, info = env.step(action.numpy())
            next_obs = torch.from_numpy(np.asarray(obs_np))
            dones = np.asarray(term | trunc)
            next_done = torch.from_numpy(dones).float()
            reward_np = np.asarray(reward, dtype=np.float32)
            rewards_buf[step] = torch.from_numpy(reward_np)

            current_ep_returns += reward_np
            for i in range(NUM_ENVS):
                if dones[i]:
                    ep_returns.append(current_ep_returns[i])
                    current_ep_returns[i] = 0.0

        with torch.inference_mode():
            next_obs_gpu = next_obs.to(device=device, dtype=torch.float32) / 255.0
            next_value = agent.get_value(next_obs_gpu).flatten()

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

        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        sil_buffer.add(b_obs, b_actions, b_returns, b_advantages)

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

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(newvalue - mb_values, -CLIP_COEF, CLIP_COEF)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                if math.isnan(loss.item()):
                    print("FAIL")
                    exit(1)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # Forward dynamics auxiliary loss (learn physics from consecutive pairs)
        dyn_steps = min(NUM_STEPS - 1, 16)  # use 16 consecutive pairs
        dyn_obs_t = obs_buf[:dyn_steps].reshape(-1, *obs_shape).to(device=device, dtype=torch.float32).div_(255.0)
        dyn_obs_tp1 = obs_buf[1:dyn_steps+1].reshape(-1, *obs_shape).to(device=device, dtype=torch.float32).div_(255.0)
        dyn_actions = actions_buf[:dyn_steps].reshape(-1).to(device)
        dyn_actions_oh = F.one_hot(dyn_actions, agent.num_actions).float()

        with torch.no_grad():
            target_feat = agent._encode(dyn_obs_tp1)
        current_feat = agent._encode(dyn_obs_t)
        pred_feat = agent.dynamics(torch.cat([current_feat, dyn_actions_oh], dim=-1))
        dyn_loss = 0.5 * F.mse_loss(pred_feat, target_feat.detach())

        optimizer.zero_grad(set_to_none=True)
        dyn_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        if sil_buffer.size >= SIL_BATCH:
            for _ in range(SIL_EPOCHS):
                sil_data = sil_buffer.sample(SIL_BATCH)
                if sil_data is not None:
                    sil_obs, sil_actions, sil_returns = sil_data
                    sil_obs = sil_obs.to(device=device, dtype=torch.float32) / 255.0
                    sil_actions = sil_actions.to(device)
                    sil_returns = sil_returns.to(device)

                    _, sil_logprob, _, sil_value = agent.get_action_and_value(sil_obs, sil_actions)
                    sil_value = sil_value.view(-1)

                    sil_advantage = (sil_returns - sil_value.detach()).clamp(min=0)
                    sil_policy_loss = -(sil_logprob * sil_advantage).mean()
                    sil_value_loss = 0.5 * (sil_advantage ** 2).mean()
                    sil_loss = SIL_COEF * (sil_policy_loss + sil_value_loss)

                    optimizer.zero_grad(set_to_none=True)
                    sil_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

        with torch.no_grad():
            for p_ema, p in zip(ema_agent.parameters(), agent.parameters()):
                p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        total_frames += batch_size
        num_updates += 1

        if num_updates > 2:
            total_training_time += dt

        pct_done = 100 * progress
        fps = int(batch_size / dt)
        remaining = max(0, TIME_BUDGET - total_training_time)
        avg_ep_ret = np.mean(ep_returns[-100:]) if ep_returns else 0.0
        print(f"\rupdate {num_updates:04d} ({pct_done:.1f}%) | ret: {avg_ep_ret:.1f} | sil: {sil_buffer.size} | fps: {fps:,} | remaining: {remaining:.0f}s    ", end="", flush=True)

        if num_updates > 2 and total_training_time >= TIME_BUDGET:
            break

    print()
    env.close()

    ema_agent.eval()
    avg_return = evaluate_return(ema_agent, device)

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

    print("---")
    print(f"avg_return:       {avg_return:.2f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_frames_M:   {total_frames / 1e6:.1f}")
    print(f"num_updates:      {num_updates}")
