import os
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


class Settings:
    """Runtime configuration loaded from environment variables with defaults."""

    def __init__(self) -> None:
        self.output_dir = Path(os.getenv("PODCAST_OUTPUT_DIR", str(ROOT_DIR / "outputs")))
        self.checkpoint_db = Path(
            os.getenv("PODCAST_CHECKPOINT_DB", str(ROOT_DIR / "outputs" / "checkpoints.db"))
        )

        self.device = os.getenv("PODCAST_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.allow_cpu = _env_bool("PODCAST_ALLOW_CPU", default=False)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.llm_model = os.getenv("PODCAST_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.tts_model = os.getenv(
            "PODCAST_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        self.music_model = os.getenv("PODCAST_MUSIC_MODEL", "facebook/musicgen-small")

        self.default_duration_mins = _env_int("PODCAST_DEFAULT_DURATION_MINS", 5)
        self.tts_sample_rate = _env_int("PODCAST_TTS_SAMPLE_RATE", 24000)
        self.output_sample_rate = _env_int("PODCAST_OUTPUT_SAMPLE_RATE", 44100)
        self.music_duration_secs = _env_int("PODCAST_MUSIC_DURATION_SECS", 12)

        self.host_voice = os.getenv("PODCAST_HOST_VOICE", "Andrew Chipper")
        self.guest_voice = os.getenv("PODCAST_GUEST_VOICE", "Ana Florence")
        self.host_speaker_wav = os.getenv("PODCAST_HOST_SPEAKER_WAV", "")
        self.guest_speaker_wav = os.getenv("PODCAST_GUEST_SPEAKER_WAV", "")
        self.tts_language = os.getenv("PODCAST_TTS_LANGUAGE", "en")
        self.tts_split_sentences = _env_bool("PODCAST_TTS_SPLIT_SENTENCES", default=True)
        self.tts_expand_abbreviations = _env_bool(
            "PODCAST_TTS_EXPAND_ABBREVIATIONS",
            default=False,
        )

        self.research_max_results = _env_int("PODCAST_RESEARCH_MAX_RESULTS", 5)
        self.research_providers = self._parse_provider_list(
            os.getenv(
                "PODCAST_RESEARCH_PROVIDERS",
                "wikipedia,arxiv,tavily,brave,duckduckgo",
            )
        )
        self.tavily_api_key = os.getenv("PODCAST_TAVILY_API_KEY", "")
        self.brave_api_key = os.getenv("PODCAST_BRAVE_API_KEY", "")
        self.segment_silence_ms = _env_int("PODCAST_SEGMENT_SILENCE_MS", 300)
        self.music_duck_db = _env_int("PODCAST_MUSIC_DUCK_DB", 6)
        self.mp3_bitrate = os.getenv("PODCAST_MP3_BITRATE", "192k")

        self.llm_temperature = _env_float("PODCAST_LLM_TEMPERATURE", 0.7)
        self.llm_top_p = _env_float("PODCAST_LLM_TOP_P", 0.9)
        self.tts_chunk_max_chars = _env_int("PODCAST_TTS_CHUNK_MAX_CHARS", 250)

        self.min_gpu_free_mb = _env_int("PODCAST_MIN_GPU_FREE_MB", 256)
        self.max_script_lines = _env_int("PODCAST_MAX_SCRIPT_LINES", 15)
        self.node_retry_attempts = _env_int("PODCAST_NODE_RETRY_ATTEMPTS", 3)

        self.log_level = os.getenv("PODCAST_LOG_LEVEL", "INFO")
        self.log_json = _env_bool("PODCAST_LOG_JSON", default=False)

    @staticmethod
    def _parse_provider_list(raw: str) -> list[str]:
        providers = [part.strip().lower() for part in raw.split(",") if part.strip()]
        allowed = {"wikipedia", "arxiv", "tavily", "brave", "duckduckgo"}
        return [provider for provider in providers if provider in allowed]

    def speaker_wav_for(self, speaker: str) -> str:
        if speaker == "host":
            return self.host_speaker_wav
        if speaker == "guest":
            return self.guest_speaker_wav
        return ""

    def resolved_speaker_wav(self, speaker: str) -> Path | None:
        raw_path = self.speaker_wav_for(speaker).strip()
        if not raw_path:
            return None
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Speaker reference WAV not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Speaker reference WAV is empty: {path}")
        return path

    @property
    def quantization_config(self) -> dict:
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": self.dtype,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def run_output_dir(self, run_id: str) -> Path:
        path = self.output_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_runtime(self) -> None:
        import shutil

        if self.device == "cpu" and not self.allow_cpu:
            raise RuntimeError(
                "CUDA is not available. Set PODCAST_ALLOW_CPU=true to run on CPU "
                "(very slow for TTS and music generation)."
            )
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg was not found on PATH. Install ffmpeg before generating podcasts."
            )


settings = Settings()

# Backward-compatible module-level aliases
OUTPUT_DIR = settings.output_dir
DEVICE = settings.device
DTYPE = settings.dtype
LLM_MODEL = settings.llm_model
TTS_MODEL = settings.tts_model
MUSIC_MODEL = settings.music_model
DEFAULT_DURATION_MINS = settings.default_duration_mins
SAMPLE_RATE = settings.tts_sample_rate
OUTPUT_SAMPLE_RATE = settings.output_sample_rate
MUSIC_DURATION_SECS = settings.music_duration_secs
QUANTIZATION_CONFIG = settings.quantization_config
HOST_VOICE = settings.host_voice
GUEST_VOICE = settings.guest_voice
HOST_SPEAKER_WAV = settings.host_speaker_wav
GUEST_SPEAKER_WAV = settings.guest_speaker_wav
