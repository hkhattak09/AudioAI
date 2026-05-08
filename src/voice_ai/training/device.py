"""Device selection utility."""

import torch


def get_device() -> torch.device:
    """Select the best available device.

    Priority:
      1. CUDA
      2. MPS (Apple Silicon)
      3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
