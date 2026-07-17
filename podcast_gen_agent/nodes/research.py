import logging

from duckduckgo_search import DDGS

from ..config import settings
from ..state import PodcastState, SourceInfo
from ..utils.decorators import node_handler, with_retries

logger = logging.getLogger(__name__)


@with_retries()
def _search_topic(topic: str, max_results: int) -> list[dict]:
    ddgs = DDGS()
    return list(ddgs.text(topic, max_results=max_results))


@node_handler("research")
def research_node(state: PodcastState) -> dict:
    """Gather info about the topic from the web."""
    topic = state["topic"]
    logger.info("Searching for topic: %s", topic)

    try:
        results = _search_topic(topic, settings.research_max_results)
    except Exception as exc:
        logger.warning("Search failed after retries: %s", exc)
        return {
            "research_data": f"Topic: {topic}\nNo additional research available.",
            "sources": [],
        }

    sources: list[SourceInfo] = []
    research_text = f"Topic: {topic}\n\nKey Information:\n"
    for i, result in enumerate(results, 1):
        title = result.get("title", "")
        body = result.get("body", "")
        url = result.get("href") or result.get("link") or ""
        sources.append(SourceInfo(title=title, body=body, url=url))
        research_text += f"\n{i}. {title}\n{body}\n"

    logger.info("Found %d sources", len(sources))
    return {"research_data": research_text, "sources": sources}
