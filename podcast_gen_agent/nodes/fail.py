import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from ..state import PodcastState
from ..utils.decorators import node_handler

logger = logging.getLogger(__name__)


@node_handler("fail")
def fail_node(state: PodcastState) -> dict:
    """Terminal node that records pipeline failure details."""
    error = state.get("error") or "Unknown pipeline error"
    run_dir = settings.run_output_dir(state["run_id"])
    manifest_path = run_dir / "run_manifest.json"

    manifest = {
        "run_id": state["run_id"],
        "topic": state["topic"],
        "status": "failed",
        "error": error,
        "node_timings": state.get("node_timings", {}),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.error("Pipeline failed: %s", error)
    return {"manifest_path": str(manifest_path)}


def write_success_manifest(state: PodcastState, output_path: Path) -> str:
    """Write a success manifest with timing and cost metadata."""
    run_dir = settings.run_output_dir(state["run_id"])
    manifest_path = run_dir / "run_manifest.json"
    total_seconds = sum(state.get("node_timings", {}).values())

    manifest = {
        "run_id": state["run_id"],
        "topic": state["topic"],
        "status": "completed",
        "final_audio_path": str(output_path),
        "transcript_path": state.get("transcript_path", ""),
        "node_timings": state.get("node_timings", {}),
        "total_wall_seconds": round(total_seconds, 3),
        "estimated_gpu_cost_usd": round(total_seconds * 0.0006, 4),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)
