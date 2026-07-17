import argparse
import logging
import sys
import traceback
from pathlib import Path

from .config import settings
from .graph import get_graph
from .state import make_initial_state
from .utils.gpu import set_seed
from .utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a podcast from a topic")
    parser.add_argument("topic", nargs="?", help="Topic for the podcast")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help=f"Target duration in minutes (default: {settings.default_duration_mins})",
    )
    parser.add_argument("--output-dir", type=str, help="Directory for generated files")
    parser.add_argument("--host-voice", type=str, help="XTTS speaker name for the host")
    parser.add_argument("--guest-voice", type=str, help="XTTS speaker name for the guest")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    parser.add_argument("--run-id", type=str, help="Custom run identifier")
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume a previous run using its thread/run id",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        default=settings.log_json,
        help="Emit structured JSON logs",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level, json_logs=args.json_logs)

    if not args.topic and not args.resume:
        parser.error("topic is required unless --resume is provided")

    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
    if args.host_voice:
        settings.host_voice = args.host_voice
    if args.guest_voice:
        settings.guest_voice = args.guest_voice

    settings.ensure_output_dir()

    try:
        settings.validate_runtime()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    duration = args.duration or settings.default_duration_mins
    topic = args.topic or "Resumed podcast run"

    if args.seed is not None:
        set_seed(args.seed)

    logger.info("=" * 50)
    logger.info("Podcast-Gen-Agent")
    logger.info("Topic: %s", topic)
    logger.info("Duration: ~%d minutes", duration)
    logger.info("Output: %s", settings.output_dir)
    logger.info("=" * 50)

    graph = get_graph()

    if args.resume:
        thread_id = args.resume
        logger.info("Resuming run %s", thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(None, config=config)
    else:
        initial_state = make_initial_state(
            topic,
            duration,
            run_id=args.run_id,
            seed=args.seed,
        )
        thread_id = initial_state["run_id"]
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(initial_state, config=config)

    if result.get("error"):
        logger.error("Pipeline failed: %s", result["error"])
        if result.get("manifest_path"):
            logger.error("Manifest: %s", result["manifest_path"])
        sys.exit(1)

    final_path = result.get("final_audio_path")
    logger.info("=" * 50)
    logger.info("COMPLETE")
    if final_path:
        logger.info("Audio: %s", final_path)
    if result.get("transcript_path"):
        logger.info("Transcript: %s", result["transcript_path"])
    if result.get("manifest_path"):
        logger.info("Manifest: %s", result["manifest_path"])
    logger.info("Run id: %s", thread_id)
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Unhandled error:\n%s", traceback.format_exc())
        sys.exit(1)
