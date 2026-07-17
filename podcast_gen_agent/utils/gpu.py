import logging

import torch

from ..config import settings

logger = logging.getLogger(__name__)


def clear_gpu_cache() -> None:
    """Release unused CUDA memory when a GPU is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_gpu_memory(min_free_mb: int | None = None) -> None:
    """Raise if free GPU memory is below the configured threshold."""
    if not torch.cuda.is_available():
        return

    threshold = min_free_mb if min_free_mb is not None else settings.min_gpu_free_mb
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mb = free_bytes / (1024 * 1024)
    total_mb = total_bytes / (1024 * 1024)

    if free_mb < threshold:
        raise RuntimeError(
            f"Insufficient GPU memory: {free_mb:.0f} MB free of {total_mb:.0f} MB "
            f"(need at least {threshold} MB free)."
        )

    logger.debug("GPU memory OK: %.0f MB free of %.0f MB", free_mb, total_mb)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible generation."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
