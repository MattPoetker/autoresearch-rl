# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real RL training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better agent. The training code here is a single-GPU PPO implementation (CleanRL-style) training on Atari Breakout via Gymnasium. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, environment setup, and evaluation. Creates Gymnasium vectorized Atari environments and provides the `evaluate_return` function. Not modified.
- **`train.py`** — the single file the agent edits. Contains the NatureCNN agent, PPO optimizer, and training loop. Everything is fair game: architecture, hyperparameters, number of envs, rollout length, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup), regardless of the details of your compute. The metric is **avg_return** (average episode return over 30 greedy eval episodes) — higher is better.

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Verify environment setup (one-time)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, env setup + evaluation (do not modify)
train.py        — agent, PPO optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, num envs, architecture, etc). Second, this means that autoresearch will find the most optimal agent for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch, Gymnasium, and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## RL tuning tips

If you're running on smaller hardware, here are some knobs to adjust:

1. **NUM_ENVS**: Reduce from 128 to 32 or 16 if you're memory constrained.
2. **NUM_STEPS**: Reduce rollout length from 128 to 64 or 32.
3. **HIDDEN_SIZE**: Reduce from 512 to 256 for a smaller network.
4. **NUM_MINIBATCHES**: Increase to reduce per-minibatch memory.

The agent can experiment with: architecture changes (deeper/wider networks, LSTM, attention), PPO hyperparameters (clip coef, entropy coef, GAE lambda), reward normalization, observation augmentation, auxiliary losses, frame stacking, and more.

## License

MIT
