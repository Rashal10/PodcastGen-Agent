from pydub import AudioSegment


def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize audio to target volume."""
    change = target_dbfs - audio.dBFS
    return audio.apply_gain(change)


def add_fade(audio: AudioSegment, fade_in_ms: int = 500, fade_out_ms: int = 500) -> AudioSegment:
    """Add fade in/out to audio."""
    return audio.fade_in(fade_in_ms).fade_out(fade_out_ms)


def concat_with_silence(segments: list[AudioSegment], silence_ms: int = 300) -> AudioSegment:
    """Join audio segments with silence between them."""
    if not segments:
        return AudioSegment.empty()
    
    silence = AudioSegment.silent(duration=silence_ms)
    result = segments[0]
    
    for seg in segments[1:]:
        result += silence + seg
    
    return result
