"""Train MineIA on a single curriculum phase.

Examples:
    python scripts/train.py --config configs/phase1_survival.yaml
    python scripts/train.py --config configs/phase3_dragon.yaml \
        --resume_from checkpoints/phase3_dragon/latest.pt
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mineia.agents import load_vpt_policy
from mineia.envs import make_vec_env
from mineia.rewards import build_shaper
from mineia.training import PPOTrainer
from mineia.utils import get_device, load_config, log_device_info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="Path to a phase YAML config.")
    ap.add_argument("--resume_from", type=Path, default=None, help="Override checkpoint.resume_from.")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--num_envs", type=int, default=None, help="Override env.num_envs.")
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.resume_from is not None:
        cfg["checkpoint"]["resume_from"] = str(args.resume_from)
    if args.num_envs is not None:
        cfg["env"]["num_envs"] = args.num_envs

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg["device"]["prefer"])
    log_device_info(device)

    phase = cfg.get("phase", {}).get("name", "phase")
    run_name = args.run_name or f"{phase}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Build vectorized env with the phase's reward shaper.
    env_cfg = cfg["env"]
    shaper = build_shaper(cfg["reward"])
    envs = make_vec_env(
        env_id=env_cfg["id"],
        num_envs=int(env_cfg["num_envs"]),
        resolution=tuple(env_cfg["resolution"]),
        frame_skip=int(env_cfg["frame_skip"]),
        max_episode_steps=int(env_cfg["max_episode_steps"]),
        reward_shaper=shaper,
        seed=int(cfg.get("seed", 42)),
        asynchronous=True,
    )

    # Load VPT policy + value head.
    vpt_cfg = cfg["vpt"]
    weights_dir = Path(vpt_cfg["weights_dir"])
    policy = load_vpt_policy(
        model_def=weights_dir / vpt_cfg["model_def"],
        weights=weights_dir / vpt_cfg["policy_weights"],
        device=device,
        freeze_backbone=bool(vpt_cfg.get("freeze_backbone", False)),
    )

    trainer = PPOTrainer(policy=policy, envs=envs, cfg=cfg, device=device, run_name=run_name)
    trainer.train()
    envs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
