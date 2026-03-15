from __future__ import annotations

from typing import Optional

import torch


def is_apple_mps_device(device: Optional[str] = None) -> bool:
    return str(device or "").lower() == "mps"


def mps_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def preferred_torch_device(explicit_device: Optional[str] = None) -> str:
    if explicit_device:
        return str(explicit_device)
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def preferred_inference_dtype(
    device: Optional[str],
    accelerated_dtype: Optional[torch.dtype] = torch.float16,
) -> torch.dtype:
    resolved = preferred_torch_device(device)
    if resolved in {"cuda", "mps"} and accelerated_dtype is not None:
        return accelerated_dtype
    return torch.float32
