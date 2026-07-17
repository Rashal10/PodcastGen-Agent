from pathlib import Path
from unittest.mock import MagicMock, patch

from podcast_gen_agent import config
from podcast_gen_agent.nodes.voice_synthesis import _synthesize_line


def test_synthesize_line_uses_speaker_wav_when_provided(tmp_path: Path):
    output_path = tmp_path / "segment.wav"
    speaker_wav = tmp_path / "host_ref.wav"
    speaker_wav.write_bytes(b"RIFF")

    tts = MagicMock()
    _synthesize_line(
        tts,
        "Hello listeners.",
        "Andrew Chipper",
        output_path,
        speaker_wav=speaker_wav,
    )

    tts.tts_to_file.assert_called_once()
    kwargs = tts.tts_to_file.call_args.kwargs
    assert kwargs["speaker_wav"] == str(speaker_wav)
    assert kwargs["split_sentences"] is True
    assert "speaker" not in kwargs


def test_synthesize_line_uses_preset_voice_without_wav(tmp_path: Path):
    output_path = tmp_path / "segment.wav"
    tts = MagicMock()

    _synthesize_line(tts, "Hello listeners.", "Ana Florence", output_path)

    kwargs = tts.tts_to_file.call_args.kwargs
    assert kwargs["speaker"] == "Ana Florence"
    assert "speaker_wav" not in kwargs


def test_resolved_speaker_wav_requires_existing_file(tmp_path: Path, monkeypatch):
    missing = tmp_path / "missing.wav"
    monkeypatch.setattr(config.settings, "host_speaker_wav", str(missing))

    try:
        config.settings.resolved_speaker_wav("host")
        raised = False
    except FileNotFoundError:
        raised = True

    assert raised
