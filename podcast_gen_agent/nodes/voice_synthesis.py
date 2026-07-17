import logging
import subprocess
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from ..compat import ensure_transformers_compat
from ..config import settings
from ..state import PodcastState, coerce_dialogue_line, coerce_script
from ..utils.audio import normalize_segment
from ..utils.decorators import node_handler, with_retries
from ..utils.gpu import clear_gpu_cache, require_gpu_memory
from ..utils.text import chunk_text, clean_text_for_tts

logger = logging.getLogger(__name__)

_tts = None
_xtts_unavailable = False


def _load_tts():
    global _tts, _xtts_unavailable
    if _tts is not None:
        return _tts
    if _xtts_unavailable:
        raise RuntimeError("XTTS was disabled after an earlier initialization failure")

    try:
        ensure_transformers_compat()
        from TTS.api import TTS

        logger.info("Loading XTTS model")
        require_gpu_memory()
        _tts = TTS(settings.tts_model).to(settings.device)
    except Exception:
        _xtts_unavailable = True
        raise

    available = getattr(_tts, "speakers", None) or []
    for voice in (settings.host_voice, settings.guest_voice):
        if available and voice not in available:
            raise ValueError(
                f"Voice '{voice}' is not available. "
                f"Choose from: {', '.join(available[:10])}..."
            )

    return _tts


def unload_tts() -> None:
    """Release the TTS model from memory."""
    global _tts
    _tts = None
    clear_gpu_cache()
    logger.info("Unloaded TTS model")


@with_retries()
def _synthesize_line(tts: Any, text: str, voice: str, output_path: Path) -> None:
    chunks = chunk_text(text, max_chars=settings.tts_chunk_max_chars)
    if len(chunks) == 1:
        tts.tts_to_file(
            text=chunks[0],
            speaker=voice,
            language=settings.tts_language,
            file_path=str(output_path),
        )
        return

    chunk_paths: list[Path] = []
    for chunk_idx, chunk in enumerate(chunks):
        chunk_path = output_path.with_name(f"{output_path.stem}_chunk_{chunk_idx}.wav")
        tts.tts_to_file(
            text=chunk,
            speaker=voice,
            language=settings.tts_language,
            file_path=str(chunk_path),
        )
        chunk_paths.append(chunk_path)

    segments = [
        normalize_segment(AudioSegment.from_wav(str(path)), settings.output_sample_rate)
        for path in chunk_paths
    ]
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg
    combined.export(str(output_path), format="wav")

    for path in chunk_paths:
        path.unlink(missing_ok=True)


def _synthesize_with_espeak(text: str, output_path: Path, speaker: str) -> None:
    """Use eSpeak as a reliable CPU fallback when XTTS cannot run."""
    voice = "en-us+f3" if speaker == "host" else "en-us+m3"
    subprocess.run(
        [
            "espeak-ng",
            "-v",
            voice,
            "-s",
            "155",
            "-w",
            str(output_path),
            text,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@node_handler("voice")
def voice_synthesis_node(state: PodcastState) -> dict:
    """Convert one script line to audio."""
    idx = state["current_line_idx"]
    script = coerce_script(state["script"])

    if idx >= len(script):
        return {"current_line_idx": idx}

    line = coerce_dialogue_line(script[idx])
    voice = settings.host_voice if line.speaker == "host" else settings.guest_voice
    cleaned_text = clean_text_for_tts(line.text)

    run_dir = settings.run_output_dir(state["run_id"])
    output_path = run_dir / f"segment_{idx:03d}.wav"

    logger.info(
        "Synthesizing (%d/%d) %s: %s...",
        idx + 1,
        len(script),
        line.speaker,
        cleaned_text[:50],
    )

    try:
        tts = _load_tts()
        require_gpu_memory()
        _synthesize_line(tts, cleaned_text, voice, output_path)
    except Exception:
        logger.exception("XTTS failed; using eSpeak fallback for segment %d", idx)
        _synthesize_with_espeak(cleaned_text, output_path, line.speaker)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"TTS produced empty audio file: {output_path}")

    line.audio_path = str(output_path)
    script[idx] = line

    return {
        "script": script,
        "audio_segments": [str(output_path)],
        "current_line_idx": idx + 1,
    }


def should_continue_voice(state: PodcastState) -> str:
    """Router: check if more lines to synthesize."""
    if state.get("error"):
        return "fail"

    idx = state["current_line_idx"]
    total = len(state["script"])

    if idx < total:
        return "continue"

    logger.info("All %d voice segments complete", total)
    unload_tts()
    return "done"
