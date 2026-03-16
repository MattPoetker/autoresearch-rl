"""
Environment setup and evaluation for autoresearch RL experiments.
Provides vectorized Atari environments via Gymnasium and a fixed evaluation function.

Usage:
    python prepare.py   # verify environment setup
"""

import time
import functools

import numpy as np
import torch
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
ENV_ID = "ALE/Breakout-v5"  # Gymnasium Atari environment
EVAL_EPISODES = 30       # number of episodes for evaluation

# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def _atari_preprocess(env):
    """Standard Atari preprocessing: frame skip, grayscale, resize 84x84, frame stack 4."""
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def make_env(env_id=ENV_ID, num_envs=1):
    """Create Gymnasium vectorized Atari environments with standard preprocessing.

    Preprocessing (matching standard Atari PPO):
    - NoopReset (up to 30 no-ops, via AtariPreprocessing)
    - MaxAndSkip (frame skip=4, max over last 2 frames)
    - Resize to 84x84, Grayscale
    - FrameStack(4)

    Returns obs shape: (num_envs, 4, 84, 84) as uint8.
    """
    envs = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="async",
                        frameskip=1, repeat_action_probability=0,
                        wrappers=[_atari_preprocess])
    return envs

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_return(agent, device, env_id=ENV_ID, num_eval_envs=8):
    """
    Run EVAL_EPISODES greedy episodes and return the mean episodic return.
    Uses argmax (greedy) action selection for deterministic evaluation.
    Higher is better.
    """
    envs = make_env(env_id, num_envs=num_eval_envs)
    obs, _ = envs.reset()

    episode_returns = []
    current_returns = np.zeros(num_eval_envs, dtype=np.float64)

    while len(episode_returns) < EVAL_EPISODES:
        obs_tensor = torch.from_numpy(np.array(obs)).to(device=device, dtype=torch.float32) / 255.0
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
    print(f"Environment: {ENV_ID}")
    print(f"Eval episodes: {EVAL_EPISODES}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Test single env
    print("Testing single environment...")
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0)
    env = _atari_preprocess(env)
    obs, info = env.reset()
    print(f"  Observation shape: {np.array(obs).shape}, dtype: {np.array(obs).dtype}")
    print(f"  Action space: {env.action_space}")

    total_reward = 0
    steps = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if term or trunc:
            obs, info = env.reset()
    env.close()
    print(f"  Ran {steps} steps, total reward: {total_reward}")
    print()

    # Test vectorized env
    num_test_envs = 4
    print(f"Testing vectorized environment ({num_test_envs} envs)...")
    envs = make_env(ENV_ID, num_envs=num_test_envs)
    obs, info = envs.reset()
    print(f"  Observation shape: {np.array(obs).shape}, dtype: {np.array(obs).dtype}")

    t0 = time.time()
    for _ in range(100):
        actions = np.array([envs.single_action_space.sample() for _ in range(num_test_envs)])
        obs, reward, term, trunc, info = envs.step(actions)
    dt = time.time() - t0
    fps = num_test_envs * 100 / dt
    envs.close()
    print(f"  100 steps x {num_test_envs} envs in {dt:.2f}s ({fps:.0f} fps)")
    print()

    print("Done! Environment setup verified.")
