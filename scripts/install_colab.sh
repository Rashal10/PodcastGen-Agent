#!/usr/bin/env bash
# Colab / Linux GPU install script for PodcastGen-Agent.
set -euo pipefail

pip install -q --upgrade pip

pip uninstall -y TTS audiocraft langchain langchain-community langchain-text-splitters || true
pip install -q -r requirements-colab.txt
pip install -q -e . --no-deps

apt-get update -qq
apt-get install -y -qq ffmpeg espeak-ng

python - <<'PY'
import shutil
import torch
import tokenizers
import transformers
from podcast_gen_agent.compat import ensure_langchain_compat, ensure_transformers_compat

ensure_transformers_compat()
ensure_langchain_compat()

from langchain_core.globals import get_debug
from TTS.api import TTS
from transformers import MusicgenForConditionalGeneration
from podcast_gen_agent.graph import get_graph

assert torch.cuda.is_available(), "GPU runtime is not enabled"
assert shutil.which("ffmpeg"), "ffmpeg is missing"
assert shutil.which("espeak-ng"), "espeak-ng is missing"
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.57.5")
assert version.parse(tokenizers.__version__) >= version.parse("0.22.0")
assert get_debug() in (True, False)
get_graph()
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("torch", torch.__version__)
print("XTTS, MusicGen, LangGraph, ffmpeg, and eSpeak are ready")
PY
