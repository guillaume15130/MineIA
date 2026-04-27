"""YAML config loading with shallow merge + base inheritance via `defaults: <path>`."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def merge_configs(*configs: dict) -> dict:
    result: dict = {}
    for cfg in configs:
        result = _deep_merge(result, cfg)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. If it has a top-level `defaults: <path>` key, that base is loaded first."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("defaults", None)
    if base_path is None:
        return cfg

    base_path = (path.parent / base_path).resolve()
    base = load_config(base_path)
    return _deep_merge(base, cfg)
