"""Episode quality labels used in manifests and CLI output."""

QUALITY_FULL = "full"
QUALITY_DEGRADED = "degraded"
QUALITY_FAILED = "failed"


def finalize_quality(notes: list[str] | None, *, failed: bool = False) -> str:
    """Collapse accumulated quality notes into a single status."""
    if failed:
        return QUALITY_FAILED
    if notes:
        return QUALITY_DEGRADED
    return QUALITY_FULL


def unique_notes(notes: list[str] | None) -> list[str]:
    """Preserve note order while removing duplicates."""
    seen: set[str] = set()
    ordered: list[str] = []
    for note in notes or []:
        if note in seen:
            continue
        seen.add(note)
        ordered.append(note)
    return ordered
