"""Compatibility shims for dependency version mismatches."""

from __future__ import annotations


def _torch_gte(version_str: str) -> bool:
    import torch
    from packaging import version

    current = version.parse(torch.__version__.split("+")[0])
    target = version.parse(version_str)
    return current >= target


def ensure_transformers_compat() -> None:
    """Patch missing transformers symbols before coqui-tts imports them."""
    import torch
    import transformers.pytorch_utils as pytorch_utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(pytorch_utils, "isin_mps_friendly"):
        pytorch_utils.isin_mps_friendly = torch.isin

    if not hasattr(import_utils, "is_torch_greater_or_equal"):
        import_utils.is_torch_greater_or_equal = _torch_gte


# Backward-compatible alias
ensure_tts_transformers_compat = ensure_transformers_compat
