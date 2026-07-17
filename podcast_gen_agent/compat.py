"""Compatibility shims for dependency version mismatches."""


def ensure_tts_transformers_compat() -> None:
    """Patch missing transformers symbols before coqui-tts imports them."""
    try:
        from transformers.pytorch_utils import isin_mps_friendly  # noqa: F401
    except ImportError:
        import torch
        import transformers.pytorch_utils as pytorch_utils

        if not hasattr(pytorch_utils, "isin_mps_friendly"):
            pytorch_utils.isin_mps_friendly = torch.isin
