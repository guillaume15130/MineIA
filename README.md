# MineIA

Reinforcement learning agent that learns to complete Minecraft by killing the
Ender Dragon. Built around fine-tuning OpenAI's **VPT** foundation model with
**PPO** and a 3-phase **curriculum**.

## Stack

| Layer       | Choice                                                      |
| ----------- | ----------------------------------------------------------- |
| Language    | Python 3.10                                                 |
| Compute     | PyTorch 2.3 + CUDA 12.1, target GPU **RTX 4080 16 GB**      |
| Environment | [MineRL 1.0](https://github.com/minerllabs/minerl) (MC 1.16) |
| Backbone    | [VPT-2x](https://github.com/openai/Video-Pre-Training)      |
| Algorithm   | PPO (custom loop, recurrent-aware) + GAE                    |
| Logging     | Weights & Biases + TensorBoard                              |

## Curriculum

| Phase | Name      | Goal                                                                |
| ----- | --------- | ------------------------------------------------------------------- |
| 1     | Survival  | Stay alive, eat, manage health, basic motor control                 |
| 2     | Crafting  | Wood -> stone -> iron -> diamond tools, full tech tree              |
| 3     | Dragon    | Nether portal, blaze rods, eyes of ender, End, kill the dragon      |

Each phase has its own YAML in `configs/`, its own reward shaper in
`src/mineia/rewards/`, and resumes from the previous phase's final checkpoint.

## Setup

```bash
# Ubuntu 22.04, Python 3.10, NVIDIA driver >= 550, Java 8
bash scripts/setup_env.sh
source .venv/bin/activate

# Pull VPT-2x weights (~1 GB)
python scripts/download_vpt.py --size 2x
```

`setup_env.sh` also clones OpenAI's VPT repo into `vendor/vpt` so we can import
its policy / model definition modules.

## Train

Single phase:

```bash
python scripts/train.py --config configs/phase1_survival.yaml
```

Full curriculum (chains all 3 phases):

```bash
python scripts/train_curriculum.py
```

Resume a crashed run:

```bash
python scripts/train.py --config configs/phase3_dragon.yaml \
  --resume_from checkpoints/phase3_dragon/latest.pt
```

Checkpoints land in `checkpoints/<tag>/` and are rotated (`keep_last: 5` by
default). Every checkpoint stores model, optimizer, RNG state, and the config
snapshot, so a resumed run continues bit-exactly.

## Watch the agent

Two ways to see what the agent is doing.

**Passive (no extra cost during training)** - clips of env 0's POV are uploaded
to W&B every `logging.video_every_n_updates` updates (default 50). Open the
W&B run, scroll to the `rollout/env0_pov` panel.

**Active (separate process)** - load the latest checkpoint and render a live
OpenCV window:

```bash
python scripts/watch.py --config configs/phase1_survival.yaml \
  --checkpoint checkpoints/phase1_survival/latest.pt
```

Press `q` to close the window. Safe to run while training is in progress; it
spawns its own MineRL env and does not touch the trainer.

## Evaluate

```bash
python scripts/eval.py --config configs/phase3_dragon.yaml \
  --checkpoint checkpoints/phase3_dragon/latest.pt --episodes 5
```

## Layout

```
src/mineia/
  agents/      VPT policy + value head wrapper
  envs/        MineRL <-> Gymnasium adapter, curriculum builder
  rewards/     Per-phase reward shapers (survival / crafting / dragon)
  training/    PPO loop, checkpoint manager, W&B logger
  utils/       CUDA device setup, YAML config loader
configs/       base.yaml + 3 phase YAMLs (inherit via `defaults:`)
scripts/       setup_env.sh, download_vpt.py, train.py, train_curriculum.py, eval.py
tests/         pytest smoke tests (no MineRL/Java required)
```

## Tests

```bash
pytest -q
```

Smoke tests cover config inheritance, all three reward shapers, and checkpoint
save/load/rotation. They do not require MineRL, Java, or VPT weights.

## Notes for the RTX 4080

- VPT-2x with 4 parallel envs at 128x128 fits in ~12 GB, leaving headroom for
  the optimizer state. Drop to `num_envs=2` for VPT-3x.
- bf16 autocast is enabled by default (Ada Lovelace supports it natively).
- TF32 matmul + cuDNN benchmark are turned on in `utils/device.py`.
- If you hit OOM during PPO updates, lower `ppo.minibatch_size` first, then
  `ppo.rollout_steps`.

## What is and isn't here

This repo is a **scaffold**: the structure, configs, reward design, training
loop, checkpointing, and logging are all in place. Two integrations require
external code/data and will only run after `setup_env.sh` + `download_vpt.py`:

1. **VPT policy** - imported from `vendor/vpt` (cloned by `setup_env.sh`).
2. **MineRL env** - needs Java 8 and the MineRL pip package, which builds the
   underlying Minecraft client on first launch.

Phase 3 also assumes a custom MineRL task that surfaces `dragon_hp`,
`end_crystals_remaining`, and `dragon_killed` in `info`. Stock MineRL doesn't
ship that scenario; it lives on the roadmap (a Malmo XML mission file under
`envs/missions/dragon.xml` would slot in cleanly).
