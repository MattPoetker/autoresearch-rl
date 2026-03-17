"""
Environment setup and evaluation for autoresearch RL experiments.
Uses envpool (fast C++ vectorization) when available,
falls back to Gymnasium SyncVectorEnv otherwise.

Usage:
    python prepare.py   # verify environment setup
"""

import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Backend detection: envpool (fast) or gymnasium (fallback)
# ---------------------------------------------------------------------------

try:
    import envpool
    BACKEND = "envpool"
except ImportError:
    import gymnasium as gym
    try:
        import ale_py
        gym.register_envs(ale_py)
    except Exception:
        pass
    BACKEND = "gymnasium"

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_EPISODES = 30       # number of episodes for evaluation

# Environment IDs differ by backend
_ENV_IDS = {
    "envpool": "Breakout-v5",
    "gymnasium": "ALE/Breakout-v5",
}
ENV_ID = _ENV_IDS[BACKEND]

# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(env_id=None, num_envs=1):
    """Create vectorized Atari environments.

    envpool backend: C++ thread pool, zero Python overhead.
        Obs shape: (num_envs, 4, 84, 84) uint8.
    gymnasium backend: SyncVectorEnv (no subprocess overhead).
        Obs shape: (num_envs, 4, 84, 84) uint8.
    """
    if env_id is None:
        env_id = ENV_ID

    if BACKEND == "envpool":
        return envpool.make(
            env_id,
            env_type="gymnasium",
            num_envs=num_envs,
            batch_size=num_envs,
            seed=42,
            episodic_life=True,
            repeat_action_probability=0,
            img_height=84,
            img_width=84,
            stack_num=4,
            gray_scale=True,
            frame_skip=4,
            noop_max=30,
        )
    else:
        def _make_single():
            # Re-register ALE in case we're in a subprocess
            try:
                import ale_py as _ale_py
                gym.register_envs(_ale_py)
            except Exception:
                pass
            env = gym.make(env_id, frameskip=1, repeat_action_probability=0)
            env = gym.wrappers.AtariPreprocessing(env)
            env = gym.wrappers.FrameStackObservation(env, stack_size=4)
            return env

        env_fns = [_make_single for _ in range(num_envs)]
        return gym.vector.SyncVectorEnv(env_fns)

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
    _obs_sp = getattr(env, 'single_observation_space', env.observation_space)
    _act_sp = getattr(env, 'single_action_space', env.action_space)
    print(f"  Single obs shape: {_obs_sp.shape}")
    print(f"  Action space: {_act_sp}")

    total_reward = 0
    steps = 0
    for _ in range(200):
        action = np.array([_act_sp.sample()])
        obs, reward, term, trunc, info = env.step(action)
        total_reward += np.sum(reward)
        steps += 1
    env.close()
    print(f"  Ran {steps} steps, total reward: {total_reward}")
    print()

    # Test vectorized env
    num_test_envs = 64
    print(f"Testing vectorized environment ({num_test_envs} envs)...")
    env = make_env(num_envs=num_test_envs)
    obs, info = env.reset()
    obs = np.asarray(obs)
    print(f"  Observation shape: {obs.shape}, dtype: {obs.dtype}")

    _act_sp2 = getattr(env, 'single_action_space', env.action_space)
    t0 = time.time()
    num_steps = 1000
    for _ in range(num_steps):
        actions = np.array([_act_sp2.sample() for _ in range(num_test_envs)])
        obs, reward, term, trunc, info = env.step(actions)
    dt = time.time() - t0
    fps = num_test_envs * num_steps / dt
    env.close()
    print(f"  {num_steps} steps x {num_test_envs} envs in {dt:.2f}s ({fps:,.0f} fps)")
    print()

    print("Done! Environment setup verified.")
