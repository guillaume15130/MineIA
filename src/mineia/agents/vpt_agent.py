"""VPT (Video PreTraining) policy wrapper for fine-tuning with PPO.

OpenAI's VPT (Baker et al. 2022) provides a Minecraft foundation model trained on
70k hours of YouTube gameplay. We load the public weights, expose a (pi, v) head
for PPO, and optionally freeze the backbone for parameter-efficient fine-tuning.

Reference repo: https://github.com/openai/Video-Pre-Training

The OpenAI repo isn't pip-installable, so the user is expected to:
  1. Run scripts/download_vpt.py to fetch .model + .weights into weights/vpt/
  2. Have the openai/Video-Pre-Training repo cloned at ./vendor/vpt (added to sys.path)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _ensure_vpt_on_path() -> None:
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "vpt"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


class VPTPolicy(nn.Module):
    """Wraps VPT's `MinecraftAgentPolicy` and adds a value head for PPO.

    The forward pass returns logits (a Dict matching the action space) and a value scalar.
    Hidden state is opaque - we treat VPT as recurrent and let the trainer carry it.
    """

    def __init__(self, vpt_agent: Any, freeze_backbone: bool = False):
        super().__init__()
        self.agent = vpt_agent  # `MineRLAgent` from OpenAI's repo
        self.policy = vpt_agent.policy

        hidden_size = getattr(self.policy.net, "hidsize", 1024)
        self.value_head = nn.Linear(hidden_size, 1)

        if freeze_backbone:
            for p in self.policy.net.parameters():
                p.requires_grad_(False)
            log.info("VPT backbone frozen; only heads will be trained.")

    @torch.no_grad()
    def initial_state(self, batch_size: int) -> Any:
        return self.policy.initial_state(batch_size)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        state: Any,
        first: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, Any]:
        """Run VPT and return (action_logits_dict, value, new_state)."""
        # VPT API: pi_logits, vpred, _, new_state = policy(obs, first, state)
        pi_logits, _vpred_unused, _, new_state = self.policy(obs, first, state)
        # Use our own value head off the last hidden activation for stable PPO.
        # OpenAI's repo exposes the latent via `policy.net(obs, ...)` -> use the recurrent
        # output. We rely on `_vpred_unused`'s precomputed latent stashed on the policy.
        latent = getattr(self.policy, "_last_latent", None)
        if latent is None:
            value = _vpred_unused.squeeze(-1)
        else:
            value = self.value_head(latent).squeeze(-1)
        return pi_logits, value, new_state


def load_vpt_policy(
    model_def: str | Path,
    weights: str | Path,
    *,
    device: torch.device,
    freeze_backbone: bool = False,
) -> VPTPolicy:
    """Load a VPT agent from .model + .weights files."""
    _ensure_vpt_on_path()
    try:
        from agent import MineRLAgent  # type: ignore
    except ImportError as e:
        raise ImportError(
            "VPT code not found. Clone https://github.com/openai/Video-Pre-Training "
            "into ./vendor/vpt before training."
        ) from e

    model_def = Path(model_def)
    weights = Path(weights)
    if not model_def.exists():
        raise FileNotFoundError(f"VPT model def missing: {model_def}")
    if not weights.exists():
        raise FileNotFoundError(f"VPT weights missing: {weights}")

    agent = MineRLAgent(env=None, device=device, policy_kwargs={}, pi_head_kwargs={})
    agent.load_weights(str(weights))
    agent.policy.to(device)

    return VPTPolicy(agent, freeze_backbone=freeze_backbone).to(device)
