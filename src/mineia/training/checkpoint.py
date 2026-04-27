"""Checkpoint manager: save full training state every N updates and rotate old files.

Stores model + optimizer + scheduler + global_step + RNG state + config snapshot
so a crashed run can be resumed bit-exactly via `--resume_from`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(
        self,
        out_dir: str | Path,
        tag: str = "run",
        every_n_updates: int = 25,
        keep_last: int = 5,
    ):
        self.dir = Path(out_dir) / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = every_n_updates
        self.keep_last = keep_last
        self.tag = tag

    def maybe_save(self, update_idx: int, state: dict[str, Any]) -> Path | None:
        if self.every <= 0 or update_idx % self.every != 0:
            return None
        return self.save(update_idx, state)

    def save(self, update_idx: int, state: dict[str, Any]) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = self.dir / f"ckpt_u{update_idx:07d}_{ts}.pt"
        payload = dict(state)
        payload["update_idx"] = update_idx
        payload["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)

        latest = self.dir / "latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(path.name)
        except OSError:
            torch.save(payload, latest)  # Windows / non-symlink FS fallback

        self._rotate()
        log.info("Saved checkpoint -> %s", path)
        return path

    def _rotate(self) -> None:
        files = sorted(
            (p for p in self.dir.glob("ckpt_u*.pt")),
            key=lambda p: p.stat().st_mtime,
        )
        for old in files[: -self.keep_last]:
            try:
                old.unlink()
            except OSError:
                pass

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = torch.load(str(path), map_location=map_location)
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "cuda_rng_state" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return payload
