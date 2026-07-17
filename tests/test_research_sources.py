from unittest.mock import patch

from podcast_gen_agent.utils.research_sources import (
    format_research_text,
    gather_research,
    search_wikipedia,
)

def test_format_research_text_without_sources():
    text = format_research_text("LoRA", [])
    assert "No additional research available" in text


def test_gather_research_merges_providers():
    with (
        patch(
            "podcast_gen_agent.utils.research_sources.search_wikipedia",
            return_value=[
                {"title": "Wiki", "body": "Body A", "url": "https://example.com/a"}
            ],
        ),
        patch(
            "podcast_gen_agent.utils.research_sources.search_arxiv",
            return_value=[
                {"title": "Paper", "body": "Body B", "url": "https://example.com/b"}
            ],
        ),
        patch("podcast_gen_agent.utils.research_sources.search_duckduckgo", return_value=[]),
    ):
        text, sources = gather_research(
            "LoRA",
            max_results=5,
            providers=["wikipedia", "arxiv", "duckduckgo"],
        )

    assert "Wiki" in text
    assert "Paper" in text
    assert len(sources) == 2


def test_search_wikipedia_parses_api_response():
    search_payload = {
        "query": {
            "search": [{"title": "Low-rank adaptation"}],
        }
    }
    extract_payload = {
        "query": {
            "pages": {
                "42": {
                    "title": "Low-rank adaptation",
                    "extract": "LoRA is a fine-tuning method.",
                }
            }
        }
    }

    with patch(
        "podcast_gen_agent.utils.research_sources._http_get_json",
        side_effect=[search_payload, extract_payload],
    ):
        sources = search_wikipedia("LoRA", max_results=1)

    assert len(sources) == 1
    assert sources[0]["title"].startswith("Wikipedia:")
    assert "fine-tuning" in sources[0]["body"]


def test_gather_research_skips_tavily_without_key():
    with (
        patch(
            "podcast_gen_agent.utils.research_sources.search_tavily",
            side_effect=AssertionError("should not be called"),
        ),
        patch("podcast_gen_agent.utils.research_sources.search_wikipedia", return_value=[]),
    ):
        text, sources = gather_research(
            "Robotics",
            max_results=3,
            providers=["tavily", "wikipedia"],
            tavily_api_key="",
        )

    assert isinstance(text, str)
    assert sources == []


def test_gather_research_uses_tavily_when_key_present():
    with (
        patch(
            "podcast_gen_agent.utils.research_sources.search_tavily",
            return_value=[
                {
                    "title": "Tavily result",
                    "body": "Useful summary",
                    "url": "https://example.com/tavily",
                }
            ],
        ),
        patch("podcast_gen_agent.utils.research_sources.search_wikipedia", return_value=[]),
    ):
        text, sources = gather_research(
            "LoRA",
            max_results=3,
            providers=["tavily", "wikipedia"],
            tavily_api_key="tvly-test",
        )

    assert "Tavily result" in text
    assert len(sources) == 1
