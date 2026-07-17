import logging
import time
from functools import wraps
from typing import Any, Callable

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings

logger = logging.getLogger(__name__)


def with_retries(
    attempts: int | None = None,
    min_wait: int = 2,
    max_wait: int = 30,
) -> Callable:
    """Retry decorator for flaky network or GPU operations."""

    def decorator(func: Callable) -> Callable:
        max_attempts = attempts if attempts is not None else settings.node_retry_attempts
        wrapped = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            reraise=True,
        )(func)
        return wrapped

    return decorator


def node_handler(node_name: str) -> Callable:
    """Wrap LangGraph nodes with timing, tracing logs, and error capture."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict) -> dict:
            if state.get("error"):
                logger.warning("Skipping %s because pipeline already failed", node_name)
                return {}

            logger.info("Node start: %s", node_name)
            start = time.perf_counter()
            try:
                result = func(state) or {}
            except Exception as exc:
                logger.exception("Node failed: %s", node_name)
                return {"error": f"{node_name}: {exc}"}

            elapsed = time.perf_counter() - start
            timings = dict(state.get("node_timings", {}))
            timings[node_name] = round(timings.get(node_name, 0.0) + elapsed, 3)
            result["node_timings"] = timings
            logger.info("Node complete: %s (%.2fs)", node_name, elapsed)
            return result

        return wrapper

    return decorator
