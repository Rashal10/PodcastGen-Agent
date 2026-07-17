"""Install and verify the Colab dependency set."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_colab_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "TTS",
            "audiocraft",
            "langchain",
            "langchain-community",
            "langchain-text-splitters",
        ],
        check=False,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(ROOT / "requirements-colab.txt"),
        ]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT), "--no-deps"]
    )


def verify_colab_imports() -> None:
    import tokenizers
    import torch
    import transformers

    from podcast_gen_agent.compat import ensure_langchain_compat, ensure_transformers_compat

    ensure_transformers_compat()
    ensure_langchain_compat()

    from langchain_core.globals import get_debug

    from transformers import MusicgenForConditionalGeneration  # noqa: F401
    from TTS.api import TTS  # noqa: F401

    from podcast_gen_agent.graph import get_graph

    assert torch.cuda.is_available(), "GPU runtime is not enabled"
    assert shutil.which("ffmpeg"), "ffmpeg is missing"
    assert shutil.which("espeak-ng"), "espeak-ng is missing"
    from packaging import version

    assert version.parse(transformers.__version__) >= version.parse("4.57.5")
    assert version.parse(tokenizers.__version__) >= version.parse("0.22.0")
    assert get_debug() in (True, False)
    get_graph()

    print("torch", torch.__version__)
    print("transformers", transformers.__version__)
    print("tokenizers", tokenizers.__version__)
    print("Colab dependencies and pipeline imports are ready")


if __name__ == "__main__":
    install_colab_dependencies()
    verify_colab_imports()
