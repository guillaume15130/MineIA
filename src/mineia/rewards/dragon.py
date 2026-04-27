from __future__ import annotations

from typing import Any

from .base import RewardShaper


class DragonRewardShaper(RewardShaper):
    """Phase 3: Nether progression -> End -> Ender Dragon.

    Combines:
    - one-shot item acquisition rewards (obsidian, blaze rod, eye of ender, ...)
    - one-shot world events (portal lit, dimension entered, dragon killed)
    - dense per-step signals (dragon HP damage)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.item_rewards: dict[str, float] = {
            k: float(v) for k, v in config.get("item_rewards", {}).items()
        }
        self.event_rewards: dict[str, float] = {
            k: float(v) for k, v in config.get("event_rewards", {}).items()
        }

        self._claimed_items: set[str] = set()
        self._claimed_events: set[str] = set()
        self._prev_inv: dict[str, int] = {}
        self._prev_dragon_hp: float | None = None
        self._prev_crystals: int = 10

    def reset(self, obs: dict) -> None:
        self._claimed_items = set()
        self._claimed_events = set()
        self._prev_inv = CraftingInvHelper.counts(obs)
        self._prev_dragon_hp = None
        self._prev_crystals = 10

    def step(self, obs: dict, env_reward: float, done: bool, info: dict) -> float:
        r = 0.0
        inv = CraftingInvHelper.counts(obs)

        # Items (first-time acquisition).
        for item, w in self.item_rewards.items():
            if item in self._claimed_items:
                continue
            if inv.get(item, 0) > self._prev_inv.get(item, 0):
                r += w
                self._claimed_items.add(item)

        # World/event flags. These come from `info` populated by a custom MineRL
        # task or from observation booleans (e.g. dimension == "the_end").
        loc = obs.get("location_stats", {}) or {}
        dimension = loc.get("dimension") if isinstance(loc, dict) else None

        events = {
            "nether_portal_lit": info.get("nether_portal_lit", False),
            "end_portal_lit": info.get("end_portal_lit", False),
            "stronghold_entered": info.get("stronghold_entered", False),
            "end_dimension_entered": dimension == "the_end" or info.get("entered_end", False),
        }
        for name, flag in events.items():
            if flag and name not in self._claimed_events and name in self.event_rewards:
                r += self.event_rewards[name]
                self._claimed_events.add(name)

        # Dragon-specific dense signals.
        dragon_hp = info.get("dragon_hp")
        if dragon_hp is not None:
            if self._prev_dragon_hp is not None and dragon_hp < self._prev_dragon_hp:
                r += self.event_rewards.get("dragon_damage_dealt", 0.0) * (
                    self._prev_dragon_hp - dragon_hp
                )
            self._prev_dragon_hp = float(dragon_hp)

        crystals = info.get("end_crystals_remaining")
        if crystals is not None and crystals < self._prev_crystals:
            r += self.event_rewards.get("end_crystal_destroyed", 0.0) * (
                self._prev_crystals - crystals
            )
            self._prev_crystals = int(crystals)

        if info.get("dragon_killed"):
            if "dragon_killed" not in self._claimed_events:
                r += self.event_rewards.get("dragon_killed", 0.0)
                self._claimed_events.add("dragon_killed")

        stats = obs.get("life_stats", {}) or {}
        if done and float(stats.get("life", 1)) <= 0 and not info.get("dragon_killed"):
            r += self.event_rewards.get("agent_death", 0.0)

        self._prev_inv = inv
        return r


class CraftingInvHelper:
    """Local helper to share inventory parsing without an import cycle."""

    @staticmethod
    def counts(obs: dict) -> dict[str, int]:
        inv = obs.get("inventory", {}) or {}
        out: dict[str, int] = {}
        for k, v in inv.items():
            try:
                out[k] = int(v) if hasattr(v, "__int__") else int(getattr(v, "item", lambda: 0)())
            except (TypeError, ValueError):
                continue
        return out
