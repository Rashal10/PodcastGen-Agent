"""Minimal API surface for health checks and future deployment."""

from fastapi import FastAPI

from .config import settings

app = FastAPI(title="PodcastGen-Agent", version="0.2.0")


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness probe with basic runtime info."""
    return {
        "status": "ready",
        "device": settings.device,
        "output_dir": str(settings.output_dir),
        "allow_cpu": settings.allow_cpu,
    }
