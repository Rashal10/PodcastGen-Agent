import logging

from ..config import settings
from ..state import PodcastState
from ..utils.decorators import node_handler
from ..utils.research_sources import gather_research

logger = logging.getLogger(__name__)


@node_handler("research")
def research_node(state: PodcastState) -> dict:
    """Gather info about the topic from configured research providers."""
    topic = state["topic"]
    logger.info("Searching for topic: %s", topic)

    research_text, sources = gather_research(
        topic,
        max_results=settings.research_max_results,
        providers=settings.research_providers,
        tavily_api_key=settings.tavily_api_key,
        brave_api_key=settings.brave_api_key,
    )

    if sources:
        logger.info("Found %d unique source(s)", len(sources))
    else:
        logger.warning("No research sources returned for topic: %s", topic)

    return {"research_data": research_text, "sources": sources}
