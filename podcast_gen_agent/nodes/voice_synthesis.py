import torch
from pathlib import Path
from TTS.api import TTS

from ..config import TTS_MODEL, OUTPUT_DIR, HOST_VOICE, GUEST_VOICE, DEVICE
from ..state import PodcastState


_tts = None


def _load_tts():
    global _tts
    if _tts is not None:
        return _tts
    
    print(f"[Voice] Loading XTTS model...")
    _tts = TTS(TTS_MODEL).to(DEVICE)
    return _tts


def voice_synthesis_node(state: PodcastState) -> dict:
    """Convert one script line to audio."""
    tts = _load_tts()
    
    idx = state["current_line_idx"]
    script = state["script"]
    segments = list(state.get("audio_segments", []))
    
    if idx >= len(script):
        return {"current_line_idx": idx}
    
    line = script[idx]
    voice = HOST_VOICE if line.speaker == "host" else GUEST_VOICE
    
    output_path = OUTPUT_DIR / f"segment_{idx:03d}.wav"
    
    print(f"[Voice] ({idx+1}/{len(script)}) {line.speaker}: {line.text[:50]}...")
    
    tts.tts_to_file(
        text=line.text,
        speaker=voice,
        language="en",
        file_path=str(output_path),
    )
    
    segments.append(str(output_path))
    
    return {
        "audio_segments": segments,
        "current_line_idx": idx + 1,
    }


def should_continue_voice(state: PodcastState) -> str:
    """Router: check if more lines to synthesize."""
    idx = state["current_line_idx"]
    total = len(state["script"])
    
    if idx < total:
        return "continue"
    
    print(f"[Voice] All {total} segments complete")
    torch.cuda.empty_cache()
    return "done"
