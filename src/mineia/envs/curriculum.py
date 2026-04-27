"""Curriculum learning: ordered phases with promotion criteria."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CurriculumStage:
    name: str
    index: int
    config_path: Path
    config: dict[str, Any]
    # Minimum mean episodic reward (over last 50 episodes) before promoting to the next stage.
    promotion_threshold: float | None = None


def build_curriculum(config_paths: list[str | Path]) -> list[CurriculumStage]:
    from ..utils.config import load_config

    stages: list[CurriculumStage] = []
    for p in config_paths:
        path = Path(p).resolve()
        cfg = load_config(path)
        phase = cfg.get("phase", {})
        stages.append(
            CurriculumStage(
                name=phase.get("name", path.stem),
                index=phase.get("index", len(stages) + 1),
                config_path=path,
                config=cfg,
                promotion_threshold=phase.get("promotion_threshold"),
            )
        )
    stages.sort(key=lambda s: s.index)
    return stages
