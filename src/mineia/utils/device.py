"""CUDA device selection and diagnostics for RTX 4080 (16 GB, Ada Lovelace, sm_89)."""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available. Install the CUDA build of PyTorch "
            "(see requirements.txt) and verify your NVIDIA driver with `nvidia-smi`."
        )
    if prefer == "cuda":
        # Ada Lovelace (sm_89) gets best perf with TF32 + bf16 autocast.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda:0")
    return torch.device(prefer)


def log_device_info(device: torch.device) -> None:
    if device.type != "cuda":
        log.info("Using device: %s", device)
        return

    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    log.info(
        "CUDA device %d: %s | %.1f GB | sm_%d%d | CUDA %s | torch %s",
        idx,
        props.name,
        total_gb,
        props.major,
        props.minor,
        torch.version.cuda,
        torch.__version__,
    )
    if "4080" not in props.name and os.environ.get("MINEIA_STRICT_GPU"):
        log.warning("Expected RTX 4080, found '%s'. Set MINEIA_STRICT_GPU=0 to silence.", props.name)


def autocast_dtype() -> torch.dtype:
    """bf16 on Ada+, fp16 fallback elsewhere."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
