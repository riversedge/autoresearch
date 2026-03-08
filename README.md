# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — entrypoint. On Apple Silicon it prefers MLX automatically; elsewhere it runs the PyTorch trainer.
- **`train_mlx.py`** — MLX-native trainer for Apple Silicon (AdamW-only, reduced eval budget for faster loop cadence).
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). CUDA GPUs are fastest, but MPS (Apple Silicon) and CPU now run via fallback paths.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

**Platforms support**. `train.py` now prefers MLX on Apple Silicon (set `AR_BACKEND=torch` to force PyTorch). CUDA still uses Flash Attention 3 kernels in the PyTorch path. MPS/CPU PyTorch fallback remains available. Results and speed are hardware-dependent.

**Profiles and pattern mode**.
- `AR_PROFILE=auto|cuda_fast|mps_fast|cpu_safe` selects runtime tuning presets.
- `AR_PATTERN_MODE=1` enables deterministic-oriented setup and prints a runtime fingerprint for reproducibility.
- `AR_LOOP_EVAL_TOKENS` controls fast loop metric budget (default `3*524288`).
- `AR_RUN_RESEARCH_EVAL=1` enables canonical full-budget research metric at run end.
- `AR_RESEARCH_EVAL_TOKENS` overrides canonical eval token budget (default `EVAL_TOKENS`).

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (torch + mlx loaders/eval)
train.py        — entrypoint (prefers MLX on Apple Silicon)
train_mlx.py    — MLX-native model/optimizer/training loop
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Port This Pattern

Use this checklist to apply the same setup in another repo:

1. Copy [prepare.py](/Users/wrightws/code/autoresearch/prepare.py), [train.py](/Users/wrightws/code/autoresearch/train.py), [train_mlx.py](/Users/wrightws/code/autoresearch/train_mlx.py), and [scripts/smoke_backend.py](/Users/wrightws/code/autoresearch/scripts/smoke_backend.py).
2. Add backend dependencies in `pyproject.toml`: torch + optional `mlx` on macOS arm64.
3. Preserve environment knobs: `AR_BACKEND`, `AR_PROFILE`, `AR_PATTERN_MODE`, `AR_LOOP_EVAL_TOKENS`, `AR_RUN_RESEARCH_EVAL`, `AR_RESEARCH_EVAL_TOKENS`.
4. Keep both metrics in logs:
   - `loop_bpb` for iteration speed decisions.
   - `research_bpb` for canonical comparisons.
5. Add CI smoke tests using [smoke.yml](/Users/wrightws/code/autoresearch/.github/workflows/smoke.yml) and run both torch + mlx where available.
6. Acceptance checks before handing to agents:
   - `uv run prepare.py --num-shards 1`
   - `uv run python scripts/smoke_backend.py --backend torch`
   - On Apple Silicon: `uv run python scripts/smoke_backend.py --backend mlx`
   - `uv run train.py` prints backend, profile, and runtime fingerprint.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

## License

MIT
