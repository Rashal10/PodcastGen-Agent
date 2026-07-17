import logging
from typing import Any

import torch
from pydub.generators import Sine

from ..config import settings
from ..state import PodcastState
from ..utils.decorators import node_handler, with_retries
from ..utils.gpu import clear_gpu_cache, require_gpu_memory

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    logger.info("Loading MusicGen: %s", settings.music_model)
    require_gpu_memory(min_free_mb=128)
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    _processor = AutoProcessor.from_pretrained(settings.music_model)
    _model = MusicgenForConditionalGeneration.from_pretrained(
        settings.music_model,
        torch_dtype=settings.dtype,
    ).to(settings.device)
    return _model, _processor


def unload_model() -> None:
    """Release MusicGen from memory."""
    global _model, _processor
    _model = None
    _processor = None
    clear_gpu_cache()
    logger.info("Unloaded MusicGen")


@with_retries()
def _generate_music(model: Any, processor: Any, prompts: list[str]):
    inputs = processor(
        text=prompts,
        padding=True,
        return_tensors="pt",
    ).to(settings.device)
    max_new_tokens = max(1, settings.music_duration_secs * 50)
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens)


def _generate_fallback_music(intro_path, outro_path) -> None:
    """Create short musical stings when MusicGen is unavailable."""
    duration_ms = settings.music_duration_secs * 1000
    intro = (
        Sine(261.63).to_audio_segment(duration=duration_ms // 3).apply_gain(-16)
        + Sine(329.63).to_audio_segment(duration=duration_ms // 3).apply_gain(-16)
        + Sine(392.00).to_audio_segment(duration=duration_ms // 3).apply_gain(-16)
    ).fade_in(500).fade_out(1200)
    outro = (
        Sine(392.00).to_audio_segment(duration=duration_ms // 3).apply_gain(-18)
        + Sine(329.63).to_audio_segment(duration=duration_ms // 3).apply_gain(-18)
        + Sine(261.63).to_audio_segment(duration=duration_ms // 3).apply_gain(-18)
    ).fade_in(500).fade_out(1800)
    intro.export(str(intro_path), format="wav")
    outro.export(str(outro_path), format="wav")


@node_handler("music")
def music_generator_node(state: PodcastState) -> dict:
    """Generate intro and outro music in one batched call."""
    topic = state["topic"]
    run_dir = settings.run_output_dir(state["run_id"])
    intro_path = run_dir / "intro_music.wav"
    outro_path = run_dir / "outro_music.wav"

    intro_prompt = f"upbeat podcast intro music, professional, {topic} theme"
    outro_prompt = f"calm podcast outro music, {topic} theme, fade out, professional"

    try:
        model, processor = _load_model()
        logger.info("Generating intro and outro music")
        require_gpu_memory(min_free_mb=128)
        waves = _generate_music(model, processor, [intro_prompt, outro_prompt])
        sample_rate = model.config.audio_encoder.sampling_rate
        import torchaudio

        torchaudio.save(str(intro_path), waves[0].cpu(), sample_rate=sample_rate)
        torchaudio.save(str(outro_path), waves[1].cpu(), sample_rate=sample_rate)
    except Exception:
        logger.exception("MusicGen failed; creating fallback musical stings")
        _generate_fallback_music(intro_path, outro_path)
    finally:
        unload_model()

    return {
        "intro_music_path": str(intro_path),
        "outro_music_path": str(outro_path),
    }
