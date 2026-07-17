import logging

import torch
import torchaudio
from audiocraft.models import MusicGen

from ..config import settings
from ..state import PodcastState
from ..utils.decorators import node_handler, with_retries
from ..utils.gpu import clear_gpu_cache, require_gpu_memory

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    logger.info("Loading MusicGen: %s", settings.music_model)
    require_gpu_memory()
    _model = MusicGen.get_pretrained(settings.music_model)
    _model.set_generation_params(duration=settings.music_duration_secs)
    return _model


def unload_model() -> None:
    """Release MusicGen from memory."""
    global _model
    _model = None
    clear_gpu_cache()
    logger.info("Unloaded MusicGen")


@with_retries()
def _generate_music(model: MusicGen, prompts: list[str]):
    return model.generate(prompts)


@node_handler("music")
def music_generator_node(state: PodcastState) -> dict:
    """Generate intro and outro music in one batched call."""
    model = _load_model()
    topic = state["topic"]
    run_dir = settings.run_output_dir(state["run_id"])

    intro_prompt = f"upbeat podcast intro music, professional, {topic} theme"
    outro_prompt = f"calm podcast outro music, {topic} theme, fade out, professional"

    logger.info("Generating intro and outro music")
    require_gpu_memory()
    waves = _generate_music(model, [intro_prompt, outro_prompt])

    sample_rate = model.sample_rate
    intro_path = run_dir / "intro_music.wav"
    outro_path = run_dir / "outro_music.wav"

    torchaudio.save(str(intro_path), waves[0].cpu(), sample_rate=sample_rate)
    torchaudio.save(str(outro_path), waves[1].cpu(), sample_rate=sample_rate)

    clear_gpu_cache()

    return {
        "intro_music_path": str(intro_path),
        "outro_music_path": str(outro_path),
    }
