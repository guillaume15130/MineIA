"""Run the full 3-phase curriculum end-to-end.

Each phase trains until its `ppo.total_timesteps` budget is exhausted, saves a
final checkpoint, then the next phase is launched with `resume_from` pointing at
that checkpoint. This is the closest thing to "click run, leave it overnight".
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

DEFAULT_PHASES = [
    "configs/phase1_survival.yaml",
    "configs/phase2_crafting.yaml",
    "configs/phase3_dragon.yaml",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", nargs="+", default=DEFAULT_PHASES)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("curriculum")

    for cfg in args.phases:
        if not Path(cfg).exists():
            log.error("Config not found: %s", cfg)
            return 1
        log.info("=== Starting phase: %s ===", cfg)
        rc = subprocess.call([sys.executable, "scripts/train.py", "--config", cfg])
        if rc != 0:
            log.error("Phase %s failed with rc=%d - aborting curriculum.", cfg, rc)
            return rc
        log.info("=== Phase complete: %s ===", cfg)

    log.info("Curriculum complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
