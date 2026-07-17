from dataclasses import dataclass
from operator import add
from typing import Annotated, List, Optional, TypedDict

from .config import settings
from .utils.slug import new_run_id, sanitize_topic_slug


@dataclass
class DialogueLine:
    """Single line of dialogue."""

    speaker: str  # "host" or "guest"
    text: str
    audio_path: Optional[str] = None


class SourceInfo(TypedDict):
    title: str
    body: str
    url: str


class PodcastState(TypedDict):
    """State passed through the LangGraph pipeline."""

    run_id: str
    topic: str
    duration_mins: int
    seed: Optional[int]

    research_data: str
    sources: Annotated[List[SourceInfo], add]

    script: List[DialogueLine]
    current_line_idx: int

    audio_segments: Annotated[List[str], add]
    intro_music_path: str
    outro_music_path: str

    final_audio_path: str
    transcript_path: str
    manifest_path: str

    node_timings: dict
    error: Optional[str]


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
        script=[],
        current_line_idx=0,
        audio_segments=[],
        intro_music_path="",
        outro_music_path="",
        final_audio_path="",
        transcript_path="",
        manifest_path="",
        node_timings={},
        error=None,
    )


def topic_slug(topic: str) -> str:
    """Public helper for topic slug generation."""
    return sanitize_topic_slug(topic)
