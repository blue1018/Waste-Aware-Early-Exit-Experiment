from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    name = requested.lower()
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        reason = "not built into PyTorch" if not built else "not available on this macOS session"
        raise RuntimeError(f"MPS was requested but is {reason}")
    if name not in {"cpu", "mps"}:
        raise ValueError("device must be 'auto', 'mps', or 'cpu'")
    return torch.device(name)


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }

