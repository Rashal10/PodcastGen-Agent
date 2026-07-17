#!/usr/bin/env bash
# Colab / Linux GPU install script for PodcastGen-Agent.
set -euo pipefail

pip install -q --upgrade pip

# Keep Colab numpy 2.x; downgrading numpy breaks scipy (numpy.char).
pip install -q --upgrade "numpy>=2.0,<2.3" "scipy>=1.12.0,<2.0.0"

pip install -q "transformers==4.43.3" "tokenizers>=0.19,<0.20"
pip install -q \
  "langgraph>=0.2.0,<0.3.0" \
  "langgraph-checkpoint-sqlite>=2.0.0,<3.0.0" \
  "tenacity>=8.2.0,<10.0.0" \
  "bitsandbytes>=0.43.0,<0.46.0" \
  "accelerate>=0.28.0,<1.0.0" \
  "duckduckgo-search>=6.0.0,<7.0.0" \
  "pydub>=0.25.0,<0.26.0" \
  "tqdm>=4.66.0,<5.0.0"

pip install -q "coqui-tts>=0.24.0,<0.28.0"
pip install -q --force-reinstall --no-deps "transformers==4.43.3"

pip install -q "av>=12.0.0"
pip install -q "audiocraft==1.3.0" --no-deps
pip install -q encodec flashy num2words hydra-core hydra-colorlog torchmetrics demucs einops

pip install -q --force-reinstall --no-deps "transformers==4.43.3"
pip install -q --upgrade "numpy>=2.0,<2.3" "scipy>=1.12.0,<2.0.0"

apt-get install -qq ffmpeg

python - <<'PY'
import numpy as np
import scipy
import torch
import transformers

try:
    from transformers.pytorch_utils import isin_mps_friendly
except ImportError:
    import transformers.pytorch_utils as pytorch_utils
    pytorch_utils.isin_mps_friendly = torch.isin

from TTS.api import TTS
from audiocraft.models import MusicGen
import langgraph

print("numpy", np.__version__)
print("scipy", scipy.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__)
print("TTS OK, audiocraft OK, langgraph OK")
PY
