from __future__ import annotations

from typing import Any

from .base import RewardShaper


class SurvivalRewardShaper(RewardShaper):
    """Phase 1: stay alive, eat, avoid dying.

    MineRL observations expose `life_stats` with `food`, `health`, etc.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        w = config.get("weights", {})
        self.alive_bonus = float(w.get("alive_bonus", 0.0))
        self.food_gain_w = float(w.get("food_gain", 0.0))
        self.health_loss_w = float(w.get("health_loss", 0.0))
        self.death_penalty = float(w.get("death", 0.0))
        self.night_bonus = float(w.get("night_survived", 0.0))

        self._prev_food: float | None = None
        self._prev_health: float | None = None
        self._was_night: bool = False

    def reset(self, obs: dict) -> None:
        stats = obs.get("life_stats", {})
        self._prev_food = float(stats.get("food", 20))
        self._prev_health = float(stats.get("life", 20))
        self._was_night = False

    def step(self, obs: dict, env_reward: float, done: bool, info: dict) -> float:
        r = self.alive_bonus

        stats = obs.get("life_stats", {})
        food = float(stats.get("food", self._prev_food or 20))
        health = float(stats.get("life", self._prev_health or 20))

        if self._prev_food is not None:
            df = food - self._prev_food
            if df > 0:
                r += self.food_gain_w * df
        if self._prev_health is not None:
            dh = health - self._prev_health
            if dh < 0:
                r += self.health_loss_w * abs(dh)

        # Day/night: MineRL exposes world time in `location_stats` for some envs.
        loc = obs.get("location_stats", {})
        is_night = bool(loc.get("is_night", False))
        if self._was_night and not is_night:
            r += self.night_bonus
        self._was_night = is_night

        if done and health <= 0:
            r += self.death_penalty

        self._prev_food = food
        self._prev_health = health
        return r
