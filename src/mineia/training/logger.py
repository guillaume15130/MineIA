"""Unified Weights & Biases + TensorBoard logger."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class RunLogger:
    def __init__(self, cfg: dict[str, Any], run_name: str, log_dir: str | Path = "logs"):
        self.cfg = cfg
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._wandb = None
        self._tb = None

        wb_cfg = cfg.get("logging", {}).get("wandb", {})
        if wb_cfg.get("enabled"):
            try:
                import wandb

                self._wandb = wandb.init(
                    project=wb_cfg.get("project", "mineia"),
                    entity=wb_cfg.get("entity"),
                    name=run_name,
                    config=cfg,
                    tags=wb_cfg.get("tags", []),
                    dir=str(self.log_dir),
                    resume="allow",
                )
                log.info("W&B run: %s", self._wandb.url)
            except ImportError:
                log.warning("wandb not installed; skipping W&B logging.")
            except Exception as e:  # noqa: BLE001
                log.warning("W&B init failed (%s); continuing without W&B.", e)

        if cfg.get("logging", {}).get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(self.log_dir)
            except ImportError:
                log.warning("tensorboard not available.")

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        if self._wandb is not None:
            self._wandb.log(scalars, step=step)
        if self._tb is not None:
            for k, v in scalars.items():
                self._tb.add_scalar(k, v, step)

    def log_video(self, step: int, tag: str, frames) -> None:
        # frames: numpy array (T, H, W, C) uint8
        if self._wandb is not None:
            try:
                import wandb

                self._wandb.log({tag: wandb.Video(frames.transpose(0, 3, 1, 2), fps=20)}, step=step)
            except Exception as e:  # noqa: BLE001
                log.warning("W&B video log failed: %s", e)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
