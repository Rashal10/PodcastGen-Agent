import shutil

import pytest
from pydub import AudioSegment

from podcast_gen_agent.utils.audio import (
    concat_with_silence,
    export_mp3,
    normalize_audio,
    write_silent_wav,
)


def test_concat_with_silence_joins_segments():
    seg_a = AudioSegment.silent(duration=100)
    seg_b = AudioSegment.silent(duration=200)
    combined = concat_with_silence([seg_a, seg_b], silence_ms=50)
    assert len(combined) == 350


def test_normalize_audio_handles_silent_segment():
    silent = AudioSegment.silent(duration=100)
    result = normalize_audio(silent, target_dbfs=-20.0)
    assert len(result) == 100


def test_write_silent_wav_creates_file(tmp_path):
    path = tmp_path / "silent.wav"
    write_silent_wav(path, duration_ms=250)
    assert path.exists()
    assert path.stat().st_size > 0


def test_export_mp3_writes_file(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    path = tmp_path / "clip.mp3"
    export_mp3(AudioSegment.silent(duration=500), path)
    assert path.exists()
    assert path.stat().st_size > 0
