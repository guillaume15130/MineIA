from __future__ import annotations

from typing import Any

from .base import RewardShaper


class CraftingRewardShaper(RewardShaper):
    """Phase 2: tech-tree rewards. First time per episode the agent acquires an item."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.item_rewards: dict[str, float] = {
            k: float(v) for k, v in config.get("item_rewards", {}).items()
        }
        self.alive_bonus = float(config.get("alive_bonus", 0.0))
        self.death_penalty = float(config.get("death", 0.0))
        self._claimed: set[str] = set()
        self._prev_inv: dict[str, int] = {}

    def reset(self, obs: dict) -> None:
        self._claimed = set()
        self._prev_inv = self._inventory_counts(obs)

    @staticmethod
    def _inventory_counts(obs: dict) -> dict[str, int]:
        # MineRL observations expose `inventory` as either a flat dict of {item: qty}
        # or a Box per item. We coerce both into a {item: int} mapping.
        inv = obs.get("inventory", {}) or {}
        out: dict[str, int] = {}
        for k, v in inv.items():
            try:
                out[k] = int(v) if hasattr(v, "__int__") else int(getattr(v, "item", lambda: 0)())
            except (TypeError, ValueError):
                continue
        return out

    def step(self, obs: dict, env_reward: float, done: bool, info: dict) -> float:
        r = self.alive_bonus
        inv = self._inventory_counts(obs)

        for item, w in self.item_rewards.items():
            if item in self._claimed:
                continue
            if inv.get(item, 0) > self._prev_inv.get(item, 0):
                r += w
                self._claimed.add(item)

        self._prev_inv = inv

        stats = obs.get("life_stats", {})
        if done and float(stats.get("life", 1)) <= 0:
            r += self.death_penalty

        return r
