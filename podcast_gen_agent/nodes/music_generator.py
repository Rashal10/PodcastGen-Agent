import torch
import torchaudio
from audiocraft.models import MusicGen

from ..config import MUSIC_MODEL, OUTPUT_DIR, MUSIC_DURATION_SECS, DEVICE
from ..state import PodcastState


_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    
    print(f"[Music] Loading MusicGen...")
    _model = MusicGen.get_pretrained(MUSIC_MODEL)
    _model.set_generation_params(duration=MUSIC_DURATION_SECS)
    return _model


def music_generator_node(state: PodcastState) -> dict:
    """Generate intro and outro music."""
    model = _load_model()
    topic = state["topic"]
    
    # intro
    print("[Music] Generating intro...")
    intro_prompt = f"upbeat podcast intro music, professional, {topic} theme"
    intro_wav = model.generate([intro_prompt])
    
    intro_path = OUTPUT_DIR / "intro_music.wav"
    torchaudio.save(str(intro_path), intro_wav[0].cpu(), sample_rate=32000)
    
    # outro
    print("[Music] Generating outro...")
    outro_prompt = "calm podcast outro music, fade out, professional"
    outro_wav = model.generate([outro_prompt])
    
    outro_path = OUTPUT_DIR / "outro_music.wav"
    torchaudio.save(str(outro_path), outro_wav[0].cpu(), sample_rate=32000)
    
    torch.cuda.empty_cache()
    
    return {
        "intro_music_path": str(intro_path),
        "outro_music_path": str(outro_path),
    }
