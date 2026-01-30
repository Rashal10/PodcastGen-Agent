import torch
from pathlib import Path


# paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# model configs
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
MUSIC_MODEL = "facebook/musicgen-small"

# generation settings
DEFAULT_DURATION_MINS = 5
SAMPLE_RATE = 24000
MUSIC_DURATION_SECS = 12

# 4-bit quantization config
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": DTYPE,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

# voice settings (xtts built-in speakers)
HOST_VOICE = "Claribel Dervla"
GUEST_VOICE = "Daisy Studious"
