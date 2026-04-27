"""MineRL -> Gymnasium adapter with frame-skip, resize, and reward shaping hooks.

MineRL ships gym (legacy) envs; we wrap them so SB3 / Gymnasium can consume them.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from ..rewards.base import RewardShaper

log = logging.getLogger(__name__)


class MineRLGymnasiumAdapter(gym.Env):
    """Wraps a legacy MineRL `gym` env and exposes a Gymnasium API."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str,
        resolution: tuple[int, int] = (128, 128),
        frame_skip: int = 4,
        max_episode_steps: int = 6000,
        reward_shaper: RewardShaper | None = None,
    ):
        try:
            import minerl  # noqa: F401  - registers envs with classic gym
            import gym as legacy_gym
        except ImportError as e:
            raise ImportError(
                "minerl is required. Install Java 8 then `pip install minerl==1.0.2`."
            ) from e

        self._env = legacy_gym.make(env_id)
        self._resolution = resolution
        self._frame_skip = max(1, frame_skip)
        self._max_episode_steps = max_episode_steps
        self._shaper = reward_shaper
        self._step_count = 0

        h, w = resolution
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        self.action_space = self._translate_action_space(self._env.action_space)

    @staticmethod
    def _translate_action_space(legacy_space: Any) -> gym.spaces.Space:
        # MineRL exposes a Dict action space. We expose the same shape via Gymnasium.
        import gym as legacy_gym

        if isinstance(legacy_space, legacy_gym.spaces.Dict):
            return gym.spaces.Dict(
                {k: MineRLGymnasiumAdapter._translate_action_space(v) for k, v in legacy_space.spaces.items()}
            )
        if isinstance(legacy_space, legacy_gym.spaces.Box):
            return gym.spaces.Box(low=legacy_space.low, high=legacy_space.high, dtype=legacy_space.dtype)
        if isinstance(legacy_space, legacy_gym.spaces.Discrete):
            return gym.spaces.Discrete(legacy_space.n)
        if isinstance(legacy_space, legacy_gym.spaces.MultiDiscrete):
            return gym.spaces.MultiDiscrete(legacy_space.nvec)
        raise NotImplementedError(f"Unsupported MineRL space: {type(legacy_space)}")

    def _process_obs(self, obs: dict) -> np.ndarray:
        # MineRL obs is a dict with 'pov' (H, W, 3) uint8.
        pov = obs["pov"]
        h, w = self._resolution
        if pov.shape[:2] != (h, w):
            import cv2

            pov = cv2.resize(pov, (w, h), interpolation=cv2.INTER_AREA)
        return pov

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        self._step_count = 0
        if self._shaper is not None:
            self._shaper.reset(obs)
        return self._process_obs(obs), {"raw_obs": obs}

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        last_obs: dict | None = None

        for _ in range(self._frame_skip):
            obs, env_reward, done, info = self._env.step(action)
            last_obs = obs
            shaped = (
                self._shaper.step(obs, env_reward, done, info) if self._shaper else env_reward
            )
            total_reward += float(shaped)
            self._step_count += 1
            if done:
                terminated = True
                break
            if self._step_count >= self._max_episode_steps:
                truncated = True
                break

        assert last_obs is not None
        return self._process_obs(last_obs), total_reward, terminated, truncated, info

    def render(self):
        return self._env.render(mode="rgb_array")

    def close(self):
        self._env.close()


def make_env(
    env_id: str,
    *,
    resolution: tuple[int, int] = (128, 128),
    frame_skip: int = 4,
    max_episode_steps: int = 6000,
    reward_shaper: RewardShaper | None = None,
    seed: int | None = None,
) -> Callable[[], gym.Env]:
    """Returns a thunk so SB3 vector envs can lazily build the env in subprocs."""

    def _thunk() -> gym.Env:
        env = MineRLGymnasiumAdapter(
            env_id=env_id,
            resolution=resolution,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            reward_shaper=reward_shaper,
        )
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _thunk


def make_vec_env(
    env_id: str,
    num_envs: int,
    *,
    asynchronous: bool = True,
    **kwargs,
) -> gym.vector.VectorEnv:
    thunks = [make_env(env_id, seed=(kwargs.get("seed") or 0) + i, **kwargs) for i in range(num_envs)]
    cls = AsyncVectorEnv if asynchronous else SyncVectorEnv
    return cls(thunks)
