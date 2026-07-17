from podcast_gen_agent.nodes.script_generator import (
    _parse_script,
    compute_max_new_tokens,
    compute_target_lines,
)
from podcast_gen_agent.state import make_initial_state
from podcast_gen_agent.utils.slug import sanitize_topic_slug
from podcast_gen_agent.utils.text import chunk_text, clean_text_for_tts


def test_clean_text_for_tts_expands_abbreviations():
    result = clean_text_for_tts("AI and GPU APIs are core to ML.")
    assert "A.I." in result
    assert "G.P.U." in result
    assert "A.P.I" in result.replace(" ", "")
    assert "M.L." in result


def test_chunk_text_splits_long_sentences():
    long_sentence = "word " * 80
    chunks = chunk_text(long_sentence.strip(), max_chars=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_parse_script_line_by_line():
    raw = """
[Host]: Welcome to the show.
[Guest]: Thanks for having me.
"""
    lines = _parse_script(raw)
    assert len(lines) == 2
    assert lines[0].speaker == "host"
    assert lines[1].speaker == "guest"


def test_compute_max_new_tokens_scales_with_duration():
    short = compute_max_new_tokens(1)
    long = compute_max_new_tokens(10)
    assert long > short
    assert short >= 512


def test_compute_target_lines():
    assert compute_target_lines(5) == 75


def test_sanitize_topic_slug():
    assert sanitize_topic_slug("AI/ML: The Future!") == "ai_ml_the_future"
    assert sanitize_topic_slug("!!!") == "podcast"


def test_make_initial_state_has_run_id():
    state = make_initial_state("Test Topic", 5)
    assert state["run_id"]
    assert state["topic"] == "Test Topic"
    assert state["duration_mins"] == 5
    assert state["error"] is None
