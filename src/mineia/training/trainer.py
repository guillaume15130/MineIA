"""PPO trainer for VPT fine-tuning.

This is intentionally a from-scratch PPO loop (rather than SB3) because VPT's
recurrent transformer-XL backbone needs explicit hidden-state plumbing across
rollouts, which SB3's RecurrentPPO does not handle for arbitrary state shapes.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..agents.vpt_agent import VPTPolicy
from ..utils.device import autocast_dtype
from .checkpoint import CheckpointManager
from .logger import RunLogger

log = logging.getLogger(__name__)


class PPOTrainer:
    def __init__(
        self,
        policy: VPTPolicy,
        envs,                            # gymnasium VectorEnv
        cfg: dict[str, Any],
        device: torch.device,
        run_name: str,
    ):
        self.policy = policy
        self.envs = envs
        self.cfg = cfg
        self.device = device

        ppo = cfg["ppo"]
        self.rollout_steps = int(ppo["rollout_steps"])
        self.num_epochs = int(ppo["num_epochs"])
        self.minibatch_size = int(ppo["minibatch_size"])
        self.clip_range = float(ppo["clip_range"])
        self.vf_coef = float(ppo["vf_coef"])
        self.ent_coef = float(ppo["ent_coef"])
        self.gamma = float(ppo["gamma"])
        self.gae_lambda = float(ppo["gae_lambda"])
        self.max_grad_norm = float(ppo["max_grad_norm"])
        self.target_kl = float(ppo.get("target_kl", 0.0)) or None
        self.total_timesteps = int(ppo["total_timesteps"])

        trainable = [p for p in policy.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(trainable, lr=float(ppo["learning_rate"]))

        ckpt_cfg = cfg["checkpoint"]
        self.ckpt = CheckpointManager(
            out_dir=ckpt_cfg["dir"],
            tag=ckpt_cfg.get("tag", run_name),
            every_n_updates=int(ckpt_cfg["every_n_updates"]),
            keep_last=int(ckpt_cfg["keep_last"]),
        )
        self.logger = RunLogger(cfg, run_name=run_name)
        self._amp_dtype = autocast_dtype() if cfg["device"].get("amp_dtype") in {"bf16", "fp16"} else None

    def maybe_resume(self) -> int:
        path = self.cfg["checkpoint"].get("resume_from")
        if not path:
            return 0
        path = Path(path)
        if not path.exists():
            log.warning("resume_from=%s does not exist; starting fresh.", path)
            return 0
        payload = CheckpointManager.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["policy_state"])
        if "optim_state" in payload:
            self.optim.load_state_dict(payload["optim_state"])
        log.info("Resumed from %s @ update %d", path, payload.get("update_idx", 0))
        return int(payload.get("update_idx", 0))

    def train(self) -> None:
        num_envs = self.envs.num_envs
        batch_steps = self.rollout_steps * num_envs
        total_updates = self.total_timesteps // batch_steps

        start_update = self.maybe_resume()
        global_step = start_update * batch_steps
        ep_returns: deque[float] = deque(maxlen=100)
        ep_lengths: deque[int] = deque(maxlen=100)

        obs, _info = self.envs.reset(seed=self.cfg.get("seed", 0))
        state = self.policy.initial_state(num_envs)
        first = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        t0 = time.time()
        for update in range(start_update + 1, total_updates + 1):
            rollout = self._collect_rollout(obs, state, first, ep_returns, ep_lengths)
            obs, state, first = rollout["next_obs"], rollout["next_state"], rollout["next_first"]
            global_step += batch_steps

            metrics = self._update(rollout)
            metrics.update({
                "rollout/ep_return_mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
                "rollout/ep_length_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
                "rollout/sps": batch_steps / max(1e-6, time.time() - t0),
                "rollout/global_step": global_step,
            })
            self.logger.log_scalars(global_step, metrics)
            t0 = time.time()

            self.ckpt.maybe_save(update, {
                "policy_state": self.policy.state_dict(),
                "optim_state": self.optim.state_dict(),
                "global_step": global_step,
                "config": self.cfg,
            })

        # Final save.
        self.ckpt.save(total_updates, {
            "policy_state": self.policy.state_dict(),
            "optim_state": self.optim.state_dict(),
            "global_step": global_step,
            "config": self.cfg,
        })
        self.logger.close()

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_rollout(self, obs, state, first, ep_returns, ep_lengths):
        T = self.rollout_steps
        N = self.envs.num_envs

        obs_buf: list = []
        act_buf: list = []
        logp_buf = torch.zeros(T, N, device=self.device)
        val_buf = torch.zeros(T, N, device=self.device)
        rew_buf = torch.zeros(T, N, device=self.device)
        done_buf = torch.zeros(T, N, device=self.device)

        for t in range(T):
            obs_t = self._obs_to_tensor(obs)
            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype) if self._amp_dtype else _NullCtx():
                logits, value, state = self.policy(obs_t, state, first)
            action, logp = self._sample(logits)

            obs_buf.append(obs_t)
            act_buf.append(action)
            logp_buf[t] = logp
            val_buf[t] = value

            np_action = self._action_to_np(action)
            obs, reward, terminated, truncated, info = self.envs.step(np_action)
            done = np.logical_or(terminated, truncated)

            rew_buf[t] = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            done_buf[t] = torch.as_tensor(done, device=self.device, dtype=torch.float32)
            first = torch.as_tensor(done, device=self.device, dtype=torch.bool)

            self._collect_episode_stats(info, ep_returns, ep_lengths)

        # Bootstrap value for the final state.
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs)
            _, last_value, _ = self.policy(obs_t, state, first)

        adv, ret = self._gae(rew_buf, val_buf, done_buf, last_value)

        return {
            "obs": obs_buf,
            "actions": act_buf,
            "logp": logp_buf,
            "values": val_buf,
            "advantages": adv,
            "returns": ret,
            "next_obs": obs,
            "next_state": state,
            "next_first": first,
        }

    def _gae(self, rewards, values, dones, last_value):
        T, N = rewards.shape
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def _update(self, rollout) -> dict[str, float]:
        adv = rollout["advantages"].flatten()
        ret = rollout["returns"].flatten()
        old_logp = rollout["logp"].flatten()
        old_val = rollout["values"].flatten()

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = adv.shape[0]

        logs = {"train/policy_loss": 0.0, "train/value_loss": 0.0, "train/entropy": 0.0, "train/approx_kl": 0.0}
        n_updates = 0
        for _ in range(self.num_epochs):
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, self.minibatch_size):
                mb = indices[start : start + self.minibatch_size]
                losses = self._minibatch_loss(rollout, mb, old_logp[mb], adv[mb], ret[mb], old_val[mb])
                self.optim.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()

                logs["train/policy_loss"] += float(losses["pg"])
                logs["train/value_loss"] += float(losses["vf"])
                logs["train/entropy"] += float(losses["ent"])
                logs["train/approx_kl"] += float(losses["kl"])
                n_updates += 1

                if self.target_kl is not None and float(losses["kl"]) > 1.5 * self.target_kl:
                    log.info("Early-stopping epoch on KL=%.4f > %.4f", float(losses["kl"]), self.target_kl)
                    return {k: v / max(1, n_updates) for k, v in logs.items()}

        return {k: v / max(1, n_updates) for k, v in logs.items()}

    def _minibatch_loss(self, rollout, mb_idx, old_logp, adv, ret, old_val):
        # Reconstructing recurrent state for arbitrary minibatches breaks the
        # transformer-XL cache, so we recompute logits/value over the FULL chunk
        # and slice. This is what OpenAI's VPT fine-tuning code does.
        obs_seq = rollout["obs"]                # list[T] of tensors with shape (N, ...)
        act_seq = rollout["actions"]            # list[T] of action dicts
        T = len(obs_seq)
        N = obs_seq[0].shape[0] if torch.is_tensor(obs_seq[0]) else next(iter(obs_seq[0].values())).shape[0]

        first = torch.zeros(N, dtype=torch.bool, device=self.device)
        state = self.policy.initial_state(N)

        logp_all = []
        val_all = []
        ent_all = []
        for t in range(T):
            logits, value, state = self.policy(obs_seq[t], state, first)
            logp, ent = self._action_logprob_entropy(logits, act_seq[t])
            logp_all.append(logp)
            val_all.append(value)
            ent_all.append(ent)
            first = torch.zeros_like(first)

        new_logp = torch.stack(logp_all).flatten()[mb_idx]
        new_val = torch.stack(val_all).flatten()[mb_idx]
        new_ent = torch.stack(ent_all).flatten()[mb_idx]

        ratio = torch.exp(new_logp - old_logp)
        pg1 = ratio * adv
        pg2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
        pg_loss = -torch.min(pg1, pg2).mean()

        v_clipped = old_val + torch.clamp(new_val - old_val, -self.clip_range, self.clip_range)
        vf1 = (new_val - ret).pow(2)
        vf2 = (v_clipped - ret).pow(2)
        v_loss = 0.5 * torch.max(vf1, vf2).mean()

        ent_loss = new_ent.mean()
        total = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_loss
        approx_kl = (old_logp - new_logp).mean().detach()
        return {"total": total, "pg": pg_loss.detach(), "vf": v_loss.detach(), "ent": ent_loss.detach(), "kl": approx_kl}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _obs_to_tensor(self, obs):
        # gymnasium VectorEnv returns dict-of-arrays for Dict spaces, ndarray otherwise.
        if isinstance(obs, dict):
            return {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}
        return torch.as_tensor(obs, device=self.device)

    def _sample(self, logits):
        # VPT logits are a Dict: {action_name: logits}. We sample independently per head.
        actions: dict[str, torch.Tensor] = {}
        logp = 0.0
        for k, lg in logits.items():
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            actions[k] = a
            logp = logp + dist.log_prob(a)
        return actions, logp

    def _action_logprob_entropy(self, logits, action):
        logp = 0.0
        ent = 0.0
        for k, lg in logits.items():
            dist = torch.distributions.Categorical(logits=lg)
            logp = logp + dist.log_prob(action[k])
            ent = ent + dist.entropy()
        return logp, ent

    def _action_to_np(self, action):
        # Convert dict[str, Tensor(N,)] into list[dict] expected by gym vector env.
        N = next(iter(action.values())).shape[0]
        out = []
        for i in range(N):
            out.append({k: int(v[i].item()) for k, v in action.items()})
        return out

    @staticmethod
    def _collect_episode_stats(info, ep_returns, ep_lengths):
        # gymnasium >= 0.29 returns vectorized infos with `final_info` on episode end.
        if not isinstance(info, dict):
            return
        finals = info.get("final_info")
        if finals is None:
            return
        for f in finals:
            if not f or "episode" not in f:
                continue
            ep_returns.append(float(f["episode"]["r"]))
            ep_lengths.append(int(f["episode"]["l"]))


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
