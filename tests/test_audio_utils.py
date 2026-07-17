from pydub import AudioSegment

from podcast_gen_agent.utils.audio import concat_with_silence, normalize_audio


def test_concat_with_silence_joins_segments():
    seg_a = AudioSegment.silent(duration=100)
    seg_b = AudioSegment.silent(duration=200)
    combined = concat_with_silence([seg_a, seg_b], silence_ms=50)
    assert len(combined) == 350


def test_normalize_audio_handles_silent_segment():
    silent = AudioSegment.silent(duration=100)
    result = normalize_audio(silent, target_dbfs=-20.0)
    assert len(result) == 100
