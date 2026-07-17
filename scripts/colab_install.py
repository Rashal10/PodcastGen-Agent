"""Colab dependency installer and verifier."""

from __future__ import annotations

import subprocess
import sys

HF_STACK = ("transformers==4.46.3", "tokenizers==0.20.3")
NUMPY_STACK = ("numpy>=2.0,<2.3", "scipy>=1.12.0,<2.0.0")


def _pip(*packages: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if extra:
        cmd.extend(extra)
    cmd.extend(packages)
    subprocess.check_call(cmd)


def pin_numpy_stack() -> None:
    _pip(*NUMPY_STACK, extra=["--upgrade"])


def pin_hf_stack() -> None:
    """Force transformers + tokenizers versions required by XTTS and Qwen."""
    _pip(*HF_STACK, extra=["--force-reinstall"])


def ensure_hf_compat() -> None:
    import torch
    import transformers.pytorch_utils as pytorch_utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(pytorch_utils, "isin_mps_friendly"):
        pytorch_utils.isin_mps_friendly = torch.isin

    if not hasattr(import_utils, "is_torch_greater_or_equal"):
        def is_torch_greater_or_equal(version_str: str, /) -> bool:
            cur = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
            tgt = tuple(int(x) for x in version_str.split(".")[:3])
            while len(cur) < 3:
                cur += (0,)
            while len(tgt) < 3:
                tgt += (0,)
            return cur >= tgt

        import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal


def install_colab_dependencies() -> None:
    pin_numpy_stack()
    pin_hf_stack()

    _pip(
        "langgraph>=0.2.0,<0.3.0",
        "langgraph-checkpoint-sqlite>=2.0.0,<3.0.0",
        "tenacity",
        "bitsandbytes",
        "accelerate",
        "duckduckgo-search",
        "pydub",
        "tqdm",
    )

    _pip("coqui-tts>=0.24.0,<0.28.0")
    pin_hf_stack()

    _pip("av>=12.0.0")
    _pip("audiocraft==1.3.0", extra=["--no-deps"])
    _pip(
        "encodec",
        "flashy",
        "num2words",
        "hydra-core",
        "hydra-colorlog",
        "torchmetrics",
        "demucs",
        "einops",
    )

    pin_hf_stack()
    pin_numpy_stack()


def verify_colab_imports() -> None:
    import numpy as np
    import scipy
    import tokenizers
    import torch
    import transformers

    ensure_hf_compat()

    from TTS.api import TTS
    from audiocraft.models import MusicGen
    import langgraph

    print("numpy", np.__version__)
    print("scipy", scipy.__version__)
    print("tokenizers", tokenizers.__version__)
    print("transformers", transformers.__version__)
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    print("TTS OK | audiocraft OK | langgraph OK")


if __name__ == "__main__":
    install_colab_dependencies()
    verify_colab_imports()
    print("Colab dependencies ready")
