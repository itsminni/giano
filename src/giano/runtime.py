"""Shared execution helpers, independent of any model or trainer."""

import torch


def default_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, otherwise CPU for every neural family."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
