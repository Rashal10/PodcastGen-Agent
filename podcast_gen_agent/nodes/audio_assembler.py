import logging
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from pydub import AudioSegment
from pydub.effects import normalize

from ..config import settings
from ..nodes.fail import write_success_manifest
from ..state import PodcastState, topic_slug
from ..utils.audio import add_fade, concat_with_silence, normalize_segment
from ..utils.decorators import node_handler

logger = logging.getLogger(__name__)


def _write_transcript(state: PodcastState, run_dir: Path) -> str:
    lines = ["# Podcast Transcript", f"Topic: {state['topic']}", ""]
    for line in state["script"]:
        lines.append(f"[{line.speaker.title()}]: {line.text}")
    transcript_path = run_dir / "transcript.txt"
    transcript_path.write_text("\n".join(lines), encoding="utf-8")
    return str(transcript_path)


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
    """Mix all audio into final podcast."""
    logger.info("Starting final mix")

    run_dir = settings.run_output_dir(state["run_id"])
    frame_rate = settings.output_sample_rate

    intro_path = state["intro_music_path"]
    outro_path = state["outro_music_path"]
    if not intro_path or not outro_path:
        raise FileNotFoundError("Intro or outro music path is missing from state")

    intro = normalize_segment(AudioSegment.from_wav(intro_path), frame_rate)
    intro = add_fade(intro, fade_in_ms=500, fade_out_ms=1000)
    intro = intro - settings.music_duck_db

    voice_segments = [
        normalize_segment(AudioSegment.from_wav(seg_path), frame_rate)
        for seg_path in state["audio_segments"]
    ]
    if not voice_segments:
        raise ValueError("No voice segments available to assemble")

    voice_track = concat_with_silence(voice_segments, silence_ms=settings.segment_silence_ms)

    outro = normalize_segment(AudioSegment.from_wav(outro_path), frame_rate)
    outro = add_fade(outro, fade_in_ms=1000, fade_out_ms=2000)
    outro = outro - settings.music_duck_db

    final = intro + voice_track + outro
    final = normalize(final, headroom=1.0)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = topic_slug(state["topic"])
    output_path = run_dir / f"podcast_{slug}_{timestamp}.mp3"
    final.export(str(output_path), format="mp3", bitrate=settings.mp3_bitrate)

    transcript_path = _write_transcript(state, run_dir)
    _write_rss(state, run_dir, output_path)
    manifest_path = write_success_manifest(state, output_path)

    duration_mins = len(final) / 1000 / 60
    logger.info("Assembly complete: %.1f min -> %s", duration_mins, output_path)

    return {
        "final_audio_path": str(output_path),
        "transcript_path": transcript_path,
        "manifest_path": manifest_path,
    }
