from dataclasses import dataclass
from operator import add
from typing import Annotated, TypedDict

from .config import settings
from .utils.slug import new_run_id, sanitize_topic_slug


@dataclass
class DialogueLine:
    """Single line of dialogue."""

    speaker: str  # "host" or "guest"
    text: str
    audio_path: str | None = None


def coerce_dialogue_line(line: DialogueLine | dict) -> DialogueLine:
    """Normalize checkpoint or JSON state back into DialogueLine objects."""
    if isinstance(line, DialogueLine):
        return line
    return DialogueLine(
        speaker=str(line["speaker"]).lower(),
        text=str(line["text"]),
        audio_path=line.get("audio_path"),
    )


def coerce_script(script: list[DialogueLine | dict]) -> list[DialogueLine]:
    """Normalize a script list after LangGraph checkpoint round-trips."""
    return [coerce_dialogue_line(line) for line in script]


class SourceInfo(TypedDict):
    title: str
    body: str
    url: str


class PodcastState(TypedDict):
    """State passed through the LangGraph pipeline."""

    run_id: str
    topic: str
    duration_mins: int
    seed: int | None

    research_data: str
    sources: Annotated[list[SourceInfo], add]
    sources_path: str
    facts_to_hit: list[str]

    script: list[DialogueLine]
    current_line_idx: int
    estimated_duration_secs: float
    actual_duration_secs: float

    audio_segments: Annotated[list[str], add]
    tts_backends: Annotated[list[str], add]
    intro_music_path: str
    outro_music_path: str
    music_backend: str

    final_audio_path: str
    transcript_path: str
    show_notes_path: str
    manifest_path: str

    quality: str
    quality_notes: Annotated[list[str], add]
    critic_score: float | None
    critic_notes: str
    critic_attempts: int

    node_timings: dict
    error: str | None


def make_initial_state(
    topic: str,
    duration_mins: int,
    run_id: str | None = None,
    seed: int | None = None,
) -> PodcastState:
    """Build the initial graph state for a new podcast run."""
    resolved_run_id = run_id or new_run_id()
    settings.run_output_dir(resolved_run_id)

    return PodcastState(
        run_id=resolved_run_id,
        topic=topic,
        duration_mins=duration_mins,
        seed=seed,
        research_data="",
        sources=[],
        sources_path="",
        facts_to_hit=[],
        script=[],
        current_line_idx=0,
        estimated_duration_secs=0.0,
        actual_duration_secs=0.0,
        audio_segments=[],
        tts_backends=[],
        intro_music_path="",
        outro_music_path="",
        music_backend="",
        final_audio_path="",
        transcript_path="",
        show_notes_path="",
        manifest_path="",
        quality="full",
        quality_notes=[],
        critic_score=None,
        critic_notes="",
        critic_attempts=0,
        node_timings={},
        error=None,
    )


def topic_slug(topic: str) -> str:
    """Public helper for topic slug generation."""
    return sanitize_topic_slug(topic)
