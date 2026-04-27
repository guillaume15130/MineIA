"""Smoke tests that don't require MineRL/Java/VPT to be installed.

These exercise config loading, reward shapers, and checkpoint roundtrips so we
get a fast `pytest` signal before launching expensive training.
"""
from __future__ import annotations

from pathlib import Path

import torch

from mineia.rewards import build_shaper
from mineia.training import CheckpointManager
from mineia.utils import load_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_load_all_phase_configs():
    for name in ["phase1_survival.yaml", "phase2_crafting.yaml", "phase3_dragon.yaml"]:
        cfg = load_config(CONFIGS / name)
        assert "ppo" in cfg and "env" in cfg and "vpt" in cfg
        assert cfg["device"]["prefer"] in {"cuda", "cpu"}


def test_curriculum_inheritance_overrides_base():
    base = load_config(CONFIGS / "base.yaml")
    p3 = load_config(CONFIGS / "phase3_dragon.yaml")
    assert p3["ppo"]["total_timesteps"] > base["ppo"]["total_timesteps"]
    assert p3["reward"]["shaper"] == "dragon"


def test_survival_shaper_rewards_food_gain():
    cfg = load_config(CONFIGS / "phase1_survival.yaml")["reward"]
    shaper = build_shaper(cfg)
    shaper.reset({"life_stats": {"food": 10, "life": 20}})
    r = shaper.step({"life_stats": {"food": 15, "life": 20}}, env_reward=0.0, done=False, info={})
    assert r > 0


def test_crafting_shaper_first_acquisition_only():
    cfg = load_config(CONFIGS / "phase2_crafting.yaml")["reward"]
    shaper = build_shaper(cfg)
    shaper.reset({"inventory": {"log": 0}, "life_stats": {"life": 20}})
    r1 = shaper.step({"inventory": {"log": 1}, "life_stats": {"life": 20}}, 0, False, {})
    r2 = shaper.step({"inventory": {"log": 2}, "life_stats": {"life": 20}}, 0, False, {})
    assert r1 > 0
    assert r2 == 0  # already claimed this episode


def test_dragon_shaper_dense_dragon_damage():
    cfg = load_config(CONFIGS / "phase3_dragon.yaml")["reward"]
    shaper = build_shaper(cfg)
    shaper.reset({"inventory": {}})
    shaper.step({"inventory": {}}, 0, False, {"dragon_hp": 200.0})
    r = shaper.step({"inventory": {}}, 0, False, {"dragon_hp": 180.0})
    assert r > 0


def test_checkpoint_roundtrip(tmp_path):
    mgr = CheckpointManager(out_dir=tmp_path, tag="test", every_n_updates=1, keep_last=2)
    state = {"policy_state": {"w": torch.zeros(2, 2)}, "global_step": 100}
    path = mgr.save(1, state)
    payload = CheckpointManager.load(path)
    assert payload["update_idx"] == 1
    assert payload["global_step"] == 100
    assert torch.equal(payload["policy_state"]["w"], torch.zeros(2, 2))


def test_checkpoint_rotation_keeps_only_last_n(tmp_path):
    mgr = CheckpointManager(out_dir=tmp_path, tag="rot", every_n_updates=1, keep_last=2)
    for i in range(5):
        mgr.save(i, {"policy_state": {}})
    files = sorted((tmp_path / "rot").glob("ckpt_u*.pt"))
    assert len(files) == 2
