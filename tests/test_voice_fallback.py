from pathlib import Path
from unittest.mock import patch

from podcast_gen_agent.nodes.voice_synthesis import _synthesize_segment
from podcast_gen_agent.state import DialogueLine, make_initial_state


def test_synthesize_segment_falls_back_to_silent_audio(tmp_path: Path):
    output_path = tmp_path / "segment.wav"

    with (
        patch("podcast_gen_agent.nodes.voice_synthesis._load_tts", side_effect=RuntimeError("XTTS down")),
        patch(
            "podcast_gen_agent.nodes.voice_synthesis._synthesize_with_espeak",
            side_effect=FileNotFoundError("espeak missing"),
        ),
    ):
        _synthesize_segment("hello world", "Claribel Dervla", "host", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_voice_node_never_raises_when_all_tts_backends_fail(tmp_path: Path, monkeypatch):
    from podcast_gen_agent import config
    from podcast_gen_agent.nodes import voice_synthesis as voice

    monkeypatch.setattr(config.settings, "output_dir", tmp_path)
    state = make_initial_state("topic", 1)
    state["script"] = [DialogueLine("host", "Hello listeners")]

    with (
        patch.object(voice, "_load_tts", side_effect=RuntimeError("XTTS down")),
        patch.object(voice, "_synthesize_with_espeak", side_effect=FileNotFoundError("espeak missing")),
    ):
        result = voice.voice_synthesis_node(state)

    assert "error" not in result or not result.get("error")
    assert result["audio_segments"]
    assert Path(result["audio_segments"][0]).exists()
