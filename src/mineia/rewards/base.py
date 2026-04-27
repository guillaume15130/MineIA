"""Abstract reward shaper. Each phase implements `step` to translate raw MineRL
observations + env reward into a dense, phase-specific scalar reward.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RewardShaper(ABC):
    def __init__(self, config: dict[str, Any]):
        self.cfg = config

    def reset(self, obs: dict) -> None:  # noqa: B027 - intentional default
        return None

    @abstractmethod
    def step(self, obs: dict, env_reward: float, done: bool, info: dict) -> float: ...


def build_shaper(reward_cfg: dict[str, Any]) -> RewardShaper:
    name = reward_cfg.get("shaper")
    # Local imports to avoid circular dep with package __init__.
    if name == "survival":
        from .survival import SurvivalRewardShaper
        return SurvivalRewardShaper(reward_cfg)
    if name == "crafting":
        from .crafting import CraftingRewardShaper
        return CraftingRewardShaper(reward_cfg)
    if name == "dragon":
        from .dragon import DragonRewardShaper
        return DragonRewardShaper(reward_cfg)
    raise ValueError(f"Unknown reward shaper '{name}'")
