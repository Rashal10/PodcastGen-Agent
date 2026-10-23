from unittest.mock import patch

from podcast_gen_agent.utils.research_sources import (
    extract_facts,
    format_research_text,
    gather_research,
    rank_sources,
    search_wikipedia,
    write_sources_markdown,
)


def test_format_research_text_without_sources():
    text = format_research_text("LoRA", [])
    assert "No additional research available" in text


def test_rank_sources_prefers_wikipedia_and_arxiv():
    ranked = rank_sources(
        [
            {"title": "Blog", "body": "web", "url": "https://example.com/b"},
            {"title": "arXiv: Paper", "body": "paper", "url": "https://arxiv.org/abs/1"},
            {"title": "Wikipedia: LoRA", "body": "wiki", "url": "https://en.wikipedia.org/wiki/LoRA"},
        ]
    )
    assert ranked[0]["title"].startswith("Wikipedia:")
    assert ranked[1]["title"].startswith("arXiv:")


def test_extract_facts_uses_first_sentence():
    facts = extract_facts(
        [
            {
                "title": "Wikipedia: LoRA",
                "body": "LoRA is a fine-tuning method. It reduces trainable parameters.",
                "url": "https://example.com",
            }
        ]
    )
    assert facts[0].startswith("LoRA is a fine-tuning method.")


def test_write_sources_markdown(tmp_path):
    path = write_sources_markdown(
        tmp_path / "sources.md",
        "LoRA",
        [{"title": "Wikipedia: LoRA", "body": "A method.", "url": "https://example.com"}],
    )
    text = (tmp_path / "sources.md").read_text(encoding="utf-8")
    assert "Wikipedia: LoRA" in text
    assert "https://example.com" in text
    assert path.endswith("sources.md")


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
        text, sources, facts = gather_research(
            "LoRA",
            max_results=5,
            providers=["wikipedia", "arxiv", "duckduckgo"],
        )

    assert "Wiki" in text
    assert "Paper" in text
    assert len(sources) == 2
    assert facts


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
        text, sources, facts = gather_research(
            "Robotics",
            max_results=3,
            providers=["tavily", "wikipedia"],
            tavily_api_key="",
        )

    assert isinstance(text, str)
    assert sources == []
    assert facts == []


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
        text, sources, facts = gather_research(
            "LoRA",
            max_results=3,
            providers=["tavily", "wikipedia"],
            tavily_api_key="tvly-test",
        )

    assert "Tavily result" in text
    assert len(sources) == 1
    assert facts
