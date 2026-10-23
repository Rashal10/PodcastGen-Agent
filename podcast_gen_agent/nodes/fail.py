import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from ..state import PodcastState
from ..utils.llm import unload_llm
from ..utils.quality import QUALITY_FAILED, unique_notes
from ..utils.scripting import estimate_speech_seconds

logger = logging.getLogger(__name__)


def _base_manifest_fields(state: PodcastState) -> dict:
    notes = unique_notes(list(state.get("quality_notes") or []))
    return {
        "run_id": state["run_id"],
        "topic": state["topic"],
        "requested_duration_mins": state.get("duration_mins"),
        "estimated_duration_secs": state.get("estimated_duration_secs")
        or estimate_speech_seconds(state.get("script") or []),
        "critic_score": state.get("critic_score"),
        "critic_attempts": state.get("critic_attempts") or 0,
        "tts_backends": unique_notes(list(state.get("tts_backends") or [])),
        "music_backend": state.get("music_backend") or "",
        "quality_notes": notes,
        "sources_path": state.get("sources_path") or "",
        "show_notes_path": state.get("show_notes_path") or "",
        "node_timings": state.get("node_timings", {}),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def fail_node(state: PodcastState) -> dict:
    """Terminal node that records pipeline failure details."""
    unload_llm()
    error = state.get("error") or "Unknown pipeline error"
    run_dir = settings.run_output_dir(state["run_id"])
    manifest_path = run_dir / "run_manifest.json"

    manifest = {
        **_base_manifest_fields(state),
        "status": "failed",
        "quality": QUALITY_FAILED,
        "error": error,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.error("Pipeline failed: %s", error)
    return {"manifest_path": str(manifest_path), "quality": QUALITY_FAILED}


def write_success_manifest(
    state: PodcastState,
    output_path: Path,
    *,
    actual_duration_secs: float = 0.0,
    quality: str = "full",
    quality_notes: list[str] | None = None,
) -> str:
    """Write a success manifest with timing, duration, and quality metadata."""
    run_dir = settings.run_output_dir(state["run_id"])
    manifest_path = run_dir / "run_manifest.json"
    total_seconds = sum(state.get("node_timings", {}).values())
    fields = _base_manifest_fields(state)
    if quality_notes is not None:
        fields["quality_notes"] = unique_notes(quality_notes)

    manifest = {
        **fields,
        "status": "completed",
        "quality": quality,
        "final_audio_path": str(output_path),
        "transcript_path": state.get("transcript_path", ""),
        "actual_duration_secs": round(actual_duration_secs, 1),
        "total_wall_seconds": round(total_seconds, 3),
        "estimated_gpu_cost_usd": round(total_seconds * 0.0006, 4),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)
