import platform

from .config import config


def get_optimal_device() -> str:
    """Determine optimal compute device (MPS, CUDA, or CPU)."""
    if config.device != "auto":
        return config.device

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"

    # We could check torch.cuda.is_available() but avoiding importing torch eagerly
    # by default. A more robust check could be added here.
    return "cpu"


def setup_cache():
    """Ensure cache directory exists."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
