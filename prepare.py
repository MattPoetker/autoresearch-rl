"""
Environment setup and evaluation for autoresearch RL experiments.
Uses PufferLib for fast vectorized environments when available,
falls back to Gymnasium otherwise.

Usage:
    python prepare.py   # verify environment setup
"""

import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Backend detection: PufferLib (fast) or Gymnasium (fallback)
# ---------------------------------------------------------------------------

try:
    import pufferlib
    import pufferlib.environments.atari
    BACKEND = "pufferlib"
except ImportError:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    BACKEND = "gymnasium"

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_EPISODES = 30       # number of episodes for evaluation

# Environment IDs differ by backend
_ENV_IDS = {
    "pufferlib": "breakout",
    "gymnasium": "ALE/Breakout-v5",
}
ENV_ID = _ENV_IDS[BACKEND]

# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def _gym_atari_preprocess(env):
    """Standard Atari preprocessing for Gymnasium backend."""
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def make_env(env_id=None, num_envs=1):
    """Create vectorized Atari environments.

    PufferLib backend: C++-optimized vectorization, obs shape (C, H, W).
    Gymnasium backend: sync vectorization with standard preprocessing,
                       obs shape (4, 84, 84).
    """
    if env_id is None:
        env_id = ENV_ID

    if BACKEND == "pufferlib":
        return pufferlib.environments.atari.make(env_id, num_envs=num_envs)
    else:
        return gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync",
                            frameskip=1, repeat_action_probability=0,
                            wrappers=[_gym_atari_preprocess])

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_return(agent, device, env_id=None, num_eval_envs=16):
    """
    Run EVAL_EPISODES greedy episodes and return the mean episodic return.
    Uses argmax (greedy) action selection for deterministic evaluation.
    Higher is better.
    """
    if env_id is None:
        env_id = ENV_ID

    envs = make_env(env_id, num_envs=num_eval_envs)
    obs, _ = envs.reset()

    episode_returns = []
    current_returns = np.zeros(num_eval_envs, dtype=np.float64)

    while len(episode_returns) < EVAL_EPISODES:
        obs_tensor = torch.from_numpy(np.asarray(obs)).to(device=device, dtype=torch.float32) / 255.0
        logits = agent(obs_tensor)
        actions = logits.argmax(dim=-1).cpu().numpy()

        obs, rewards, terms, truncs, infos = envs.step(actions)
        current_returns += rewards
        dones = terms | truncs

        for i in range(num_eval_envs):
            if dones[i]:
                episode_returns.append(current_returns[i])
                current_returns[i] = 0.0

    envs.close()
    return float(np.mean(episode_returns[:EVAL_EPISODES]))

# ---------------------------------------------------------------------------
# Main — verify environment setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Backend: {BACKEND}")
    print(f"Environment: {ENV_ID}")
    print(f"Eval episodes: {EVAL_EPISODES}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Test single env
    print("Testing single environment...")
    env = make_env(num_envs=1)
    obs, info = env.reset()
    obs = np.asarray(obs)
    print(f"  Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Single obs shape: {env.single_observation_space.shape}")
    print(f"  Action space: {env.single_action_space}")

    total_reward = 0
    steps = 0
    for _ in range(200):
        action = np.array([env.single_action_space.sample()])
        obs, reward, term, trunc, info = env.step(action)
        total_reward += np.sum(reward)
        steps += 1
    env.close()
    print(f"  Ran {steps} steps, total reward: {total_reward}")
    print()

    # Test vectorized env
    num_test_envs = 16
    print(f"Testing vectorized environment ({num_test_envs} envs)...")
    env = make_env(num_envs=num_test_envs)
    obs, info = env.reset()
    obs = np.asarray(obs)
    print(f"  Observation shape: {obs.shape}, dtype: {obs.dtype}")

    t0 = time.time()
    for _ in range(500):
        actions = np.array([env.single_action_space.sample() for _ in range(num_test_envs)])
        obs, reward, term, trunc, info = env.step(actions)
    dt = time.time() - t0
    fps = num_test_envs * 500 / dt
    env.close()
    print(f"  500 steps x {num_test_envs} envs in {dt:.2f}s ({fps:.0f} fps)")
    print()

    print("Done! Environment setup verified.")
