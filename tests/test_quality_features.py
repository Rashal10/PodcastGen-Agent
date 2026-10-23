import json
from pathlib import Path

from pydub import AudioSegment

from podcast_gen_agent.nodes.audio_assembler import write_show_notes, write_timestamped_transcript
from podcast_gen_agent.nodes.script_critic import parse_critic_response, should_rewrite_script
from podcast_gen_agent.nodes.script_generator import _parse_script
from podcast_gen_agent.state import DialogueLine, make_initial_state
from podcast_gen_agent.utils.audio import loop_to_length, overlay_ducked_bed
from podcast_gen_agent.utils.output import list_run_summaries
from podcast_gen_agent.utils.quality import QUALITY_DEGRADED, finalize_quality
from podcast_gen_agent.utils.scripting import (
    estimate_speech_seconds,
    format_timestamp,
    refine_script,
    takeaway_lines,
)


def test_refine_script_drops_short_and_duplicate_lines():
    lines = [
        DialogueLine("host", "Hi"),
        DialogueLine("host", "Welcome to a longer opening line about policy."),
        DialogueLine("guest", "Welcome to a longer opening line about policy."),
        DialogueLine("guest", "The constraint that matters is the remaining training budget."),
    ]
    cleaned = refine_script(lines, max_lines=10)
    assert len(cleaned) == 2
    assert cleaned[0].speaker == "host"
    assert cleaned[1].speaker == "guest"


def test_parse_script_keeps_natural_lines():
    raw = """
[Host]: We are starting with the constraint that most explainers skip.
[Guest]: That is the right place to start, because the popular version is incomplete.
"""
    lines = _parse_script(raw)
    assert len(lines) == 2


def test_estimate_speech_seconds_scales_with_words(monkeypatch):
    from podcast_gen_agent import config

    monkeypatch.setattr(config.settings, "speech_words_per_minute", 150)
    script = [DialogueLine("host", " ".join(["word"] * 150))]
    assert estimate_speech_seconds(script) == 60.0


def test_format_timestamp_mm_ss():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65000) == "01:05"


def test_parse_critic_response_reads_score_and_notes():
    raw = "SCORE: 8.5\nNOTES: Add a concrete limitation in segment two."
    score, notes = parse_critic_response(raw)
    assert score == 8.5
    assert "limitation" in notes


def test_should_rewrite_script_routes(monkeypatch):
    from podcast_gen_agent import config

    monkeypatch.setattr(config.settings, "critic_min_score", 7.0)
    monkeypatch.setattr(config.settings, "critic_max_retries", 2)

    passing = make_initial_state("topic", 2)
    passing["critic_score"] = 8.0
    passing["critic_attempts"] = 1
    assert should_rewrite_script(passing) == "voice"

    retry = make_initial_state("topic", 2)
    retry["critic_score"] = 5.0
    retry["critic_attempts"] = 1
    assert should_rewrite_script(retry) == "rewrite"

    exhausted = make_initial_state("topic", 2)
    exhausted["critic_score"] = 5.0
    exhausted["critic_attempts"] = 2
    assert should_rewrite_script(exhausted) == "voice"

    failed = make_initial_state("topic", 2)
    failed["error"] = "boom"
    assert should_rewrite_script(failed) == "fail"


def test_finalize_quality_from_notes():
    assert finalize_quality([]) == "full"
    assert finalize_quality(["tts:espeak"]) == QUALITY_DEGRADED
    assert finalize_quality([], failed=True) == "failed"


def test_overlay_ducked_bed_keeps_voice_length():
    voice = AudioSegment.silent(duration=1000)
    bed = AudioSegment.silent(duration=200)
    mixed = overlay_ducked_bed(voice, bed, duck_db=12)
    assert len(mixed) == 1000
    assert len(loop_to_length(bed, 1000)) == 1000


def test_timestamped_transcript_and_show_notes(tmp_path, monkeypatch):
    from podcast_gen_agent import config

    monkeypatch.setattr(config.settings, "output_dir", tmp_path)
    state = make_initial_state("Climate policy", 2)
    state["script"] = [
        DialogueLine("host", "We open with the carbon budget that most headlines skip."),
        DialogueLine(
            "guest",
            "The hard limit is remaining gigatons, not annual percentage slogans.",
        ),
    ]
    state["sources"] = [
        {
            "title": "Wikipedia: Climate",
            "body": "A summary.",
            "url": "https://en.wikipedia.org/wiki/Climate",
        }
    ]
    state["critic_score"] = 8.0
    state["critic_attempts"] = 1
    segments = [
        AudioSegment.silent(duration=1000),
        AudioSegment.silent(duration=1500),
    ]
    transcript = Path(
        write_timestamped_transcript(state, tmp_path, segments, intro_ms=2000, silence_ms=300)
    )
    text = transcript.read_text(encoding="utf-8")
    assert "[00:02] Host:" in text
    assert "[00:03] Guest:" in text or "[00:03]" in text

    notes_path = Path(write_show_notes(state, tmp_path, audio_duration_secs=12.5))
    notes = notes_path.read_text(encoding="utf-8")
    assert "Climate policy" in notes
    assert "Takeaways" in notes
    assert "https://en.wikipedia.org/wiki/Climate" in notes
    assert takeaway_lines(state["script"])


def test_list_run_summaries_reads_manifests(tmp_path):
    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "abc",
                "topic": "Robots",
                "status": "completed",
                "quality": "full",
                "finished_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    runs = list_run_summaries(tmp_path)
    assert len(runs) == 1
    assert runs[0]["topic"] == "Robots"
