import logging
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from pydub import AudioSegment
from pydub.effects import normalize

from ..config import settings
from ..nodes.fail import write_success_manifest
from ..state import PodcastState, coerce_script, topic_slug
from ..utils.audio import (
    add_fade,
    concat_with_silence,
    export_mp3,
    normalize_segment,
    overlay_ducked_bed,
)
from ..utils.decorators import node_handler
from ..utils.quality import finalize_quality, unique_notes
from ..utils.scripting import estimate_speech_seconds, format_timestamp, takeaway_lines

logger = logging.getLogger(__name__)


def write_timestamped_transcript(
    state: PodcastState,
    run_dir: Path,
    voice_segments: list[AudioSegment],
    intro_ms: int,
    silence_ms: int,
) -> str:
    """Write a transcript with MM:SS timestamps from assembled segment lengths."""
    lines = [
        "# Podcast Transcript",
        f"Topic: {state['topic']}",
        "",
    ]
    offset_ms = intro_ms
    script = coerce_script(state["script"])
    for index, line in enumerate(script):
        lines.append(
            f"[{format_timestamp(offset_ms)}] {line.speaker.title()}: {line.text}"
        )
        if index < len(voice_segments):
            offset_ms += len(voice_segments[index])
        if index < len(script) - 1:
            offset_ms += silence_ms

    transcript_path = run_dir / "transcript.txt"
    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(transcript_path)


def write_show_notes(state: PodcastState, run_dir: Path, audio_duration_secs: float) -> str:
    """Write title, takeaways, sources, and model metadata."""
    takeaways = takeaway_lines(state.get("script") or [])
    notes = unique_notes(list(state.get("quality_notes") or []))
    quality = finalize_quality(notes)
    lines = [
        f"# Show notes: {state['topic']}",
        "",
        f"- Requested length: {state['duration_mins']} minutes",
        f"- Estimated speech: {estimate_speech_seconds(state.get('script') or []):.1f} seconds",
        f"- Final audio: {audio_duration_secs:.1f} seconds",
        f"- Quality: {quality}",
        "",
        "## Takeaways",
    ]
    if takeaways:
        for item in takeaways:
            lines.append(f"- {item}")
    else:
        lines.append("- No takeaways were extracted from the script.")

    lines.extend(["", "## Sources"])
    sources = list(state.get("sources") or [])
    if sources:
        for source in sources:
            title = source.get("title") or "Source"
            url = source.get("url") or ""
            if url:
                lines.append(f"- {title}: {url}")
            else:
                lines.append(f"- {title}")
    else:
        lines.append("- No research sources were stored for this run.")

    backends = ", ".join(unique_notes(list(state.get("tts_backends") or []))) or "unknown"
    lines.extend(
        [
            "",
            "## Models",
            f"- LLM: {settings.llm_model}",
            f"- TTS: {settings.tts_model}",
            f"- Music: {settings.music_model}",
            f"- TTS backends: {backends}",
            f"- Music backend: {state.get('music_backend') or 'unknown'}",
        ]
    )
    if notes:
        lines.extend(["", "## Quality notes"])
        for note in notes:
            lines.append(f"- {note}")
    if state.get("critic_score") is not None:
        lines.extend(
            [
                "",
                "## Critic",
                f"- Score: {state.get('critic_score')}",
                f"- Attempts: {state.get('critic_attempts') or 0}",
            ]
        )
        if state.get("critic_notes"):
            lines.append(f"- Notes: {state['critic_notes']}")

    path = run_dir / "show_notes.md"
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path)


def _write_rss(state: PodcastState, run_dir: Path, mp3_path: Path) -> str:
    title = escape(state["topic"])
    description = escape(f"Generated podcast episode about {state['topic']}")
    rss_path = run_dir / "feed.xml"
    rss_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{title}</title>
    <description>{description}</description>
    <item>
      <title>{title}</title>
      <description>{description}</description>
      <enclosure url="{escape(mp3_path.name)}" type="audio/mpeg"/>
    </item>
  </channel>
</rss>
""",
        encoding="utf-8",
    )
    return str(rss_path)


@node_handler("assemble")
def audio_assembler_node(state: PodcastState) -> dict:
    """Mix all audio into final podcast with a ducked music bed under speech."""
    logger.info("Starting final mix")

    run_dir = settings.run_output_dir(state["run_id"])
    frame_rate = settings.output_sample_rate

    intro_path = state["intro_music_path"]
    outro_path = state["outro_music_path"]
    if not intro_path or not outro_path:
        raise FileNotFoundError("Intro or outro music path is missing from state")

    intro = normalize_segment(AudioSegment.from_wav(intro_path), frame_rate)
    intro = add_fade(intro, fade_in_ms=500, fade_out_ms=1000)
    intro_bumper = intro - settings.music_duck_db

    voice_segments = [
        normalize_segment(AudioSegment.from_wav(seg_path), frame_rate)
        for seg_path in state["audio_segments"]
    ]
    if not voice_segments:
        raise ValueError("No voice segments available to assemble")

    voice_track = concat_with_silence(voice_segments, silence_ms=settings.segment_silence_ms)
    voice_track = overlay_ducked_bed(
        voice_track,
        intro,
        duck_db=settings.bed_music_duck_db,
    )

    outro = normalize_segment(AudioSegment.from_wav(outro_path), frame_rate)
    outro = add_fade(outro, fade_in_ms=1000, fade_out_ms=2000)
    outro = outro - settings.music_duck_db

    final = intro_bumper + voice_track + outro
    final = normalize(final, headroom=1.0)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = topic_slug(state["topic"])
    output_path = run_dir / f"podcast_{slug}_{timestamp}.mp3"
    export_mp3(final, output_path, bitrate=settings.mp3_bitrate)

    actual_secs = round(len(final) / 1000, 1)
    transcript_path = write_timestamped_transcript(
        state,
        run_dir,
        voice_segments,
        intro_ms=len(intro_bumper),
        silence_ms=settings.segment_silence_ms,
    )
    show_notes_path = write_show_notes(state, run_dir, actual_secs)
    _write_rss(state, run_dir, output_path)

    notes = unique_notes(list(state.get("quality_notes") or []))
    quality = finalize_quality(notes)
    manifest_path = write_success_manifest(
        {**state, "transcript_path": transcript_path, "show_notes_path": show_notes_path},
        output_path,
        actual_duration_secs=actual_secs,
        quality=quality,
        quality_notes=notes,
    )

    duration_mins = actual_secs / 60
    logger.info(
        "Assembly complete: %.1f min actual vs %d min requested -> %s",
        duration_mins,
        state["duration_mins"],
        output_path,
    )
    if notes:
        logger.warning("Episode quality is %s (%s)", quality, ", ".join(notes))

    return {
        "final_audio_path": str(output_path),
        "transcript_path": transcript_path,
        "show_notes_path": show_notes_path,
        "manifest_path": manifest_path,
        "actual_duration_secs": actual_secs,
        "quality": quality,
    }
