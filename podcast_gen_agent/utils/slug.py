import re
import uuid


def sanitize_topic_slug(topic: str, max_length: int = 30) -> str:
    """Create a filesystem-safe slug from a topic string."""
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    if not slug:
        slug = "podcast"
    return slug[:max_length]


def new_run_id() -> str:
    """Generate a unique run identifier."""
    return uuid.uuid4().hex[:12]
