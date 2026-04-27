"""Evaluate a trained MineIA checkpoint over N episodes."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from mineia.agents import load_vpt_policy
from mineia.envs import make_env
from mineia.rewards import build_shaper
from mineia.training import CheckpointManager
from mineia.utils import get_device, load_config, log_device_info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    device = get_device(cfg["device"]["prefer"])
    log_device_info(device)

    env_cfg = cfg["env"]
    shaper = build_shaper(cfg["reward"])
    env = make_env(
        env_id=env_cfg["id"],
        resolution=tuple(env_cfg["resolution"]),
        frame_skip=int(env_cfg["frame_skip"]),
        max_episode_steps=int(env_cfg["max_episode_steps"]),
        reward_shaper=shaper,
    )()

    vpt_cfg = cfg["vpt"]
    weights_dir = Path(vpt_cfg["weights_dir"])
    policy = load_vpt_policy(
        model_def=weights_dir / vpt_cfg["model_def"],
        weights=weights_dir / vpt_cfg["policy_weights"],
        device=device,
        freeze_backbone=False,
    )
    payload = CheckpointManager.load(args.checkpoint, map_location=device)
    policy.load_state_dict(payload["policy_state"])
    policy.eval()

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        state = policy.initial_state(1)
        first = torch.ones(1, dtype=torch.bool, device=device)
        ep_ret, done, trunc = 0.0, False, False
        while not (done or trunc):
            obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in (obs.items() if isinstance(obs, dict) else [("pov", obs)])}
            with torch.no_grad():
                logits, _value, state = policy(obs_t, state, first)
            action = {k: int(torch.distributions.Categorical(logits=lg).sample().item()) for k, lg in logits.items()}
            obs, r, done, trunc, _info = env.step(action)
            ep_ret += float(r)
            first = torch.zeros_like(first)
            if args.render:
                env.render()
        print(f"Episode {ep + 1}: return={ep_ret:.2f}")
        returns.append(ep_ret)

    print(f"\nMean return over {args.episodes} eps: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
