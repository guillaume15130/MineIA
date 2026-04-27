"""Live-watch a trained checkpoint in a local OpenCV window.

Run this in a separate terminal while training runs in the background.
Loads the latest checkpoint, plays one episode at full speed, and renders the
POV in a window. Press `q` to quit.

    python scripts/watch.py --config configs/phase1_survival.yaml \\
        --checkpoint checkpoints/phase1_survival/latest.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import torch

from mineia.agents import load_vpt_policy
from mineia.envs import make_env
from mineia.rewards import build_shaper
from mineia.training import CheckpointManager
from mineia.utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--scale", type=int, default=4, help="Window upscale factor.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    device = get_device(cfg["device"]["prefer"])

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

    obs, _ = env.reset()
    state = policy.initial_state(1)
    first = torch.ones(1, dtype=torch.bool, device=device)

    h, w = env_cfg["resolution"]
    win = "MineIA POV (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w * args.scale, h * args.scale)

    ep_ret = 0.0
    while True:
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _v, state = policy({"pov": obs_t} if isinstance(obs, dict) else obs_t, state, first)
        action = {k: int(torch.distributions.Categorical(logits=lg).sample().item()) for k, lg in logits.items()}
        obs, r, done, trunc, _info = env.step(action)
        ep_ret += float(r)
        first = torch.zeros_like(first)

        frame = obs if not isinstance(obs, dict) else obs["pov"]
        # OpenCV expects BGR.
        cv2.imshow(win, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if done or trunc:
            print(f"Episode finished, return={ep_ret:.2f}. Resetting.")
            obs, _ = env.reset()
            state = policy.initial_state(1)
            first = torch.ones(1, dtype=torch.bool, device=device)
            ep_ret = 0.0

    env.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
