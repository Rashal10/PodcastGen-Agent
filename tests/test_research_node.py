from podcast_gen_agent.nodes.research import research_node
from podcast_gen_agent.state import make_initial_state


def test_research_node_returns_sources(mocker):
    mocker.patch(
        "podcast_gen_agent.nodes.research._search_topic",
        return_value=[
            {"title": "Title One", "body": "Body one", "href": "https://example.com/1"},
            {"title": "Title Two", "body": "Body two"},
        ],
    )

    state = make_initial_state("Quantum Computing", 5)
    result = research_node(state)

    assert "research_data" in result
    assert "Quantum Computing" in result["research_data"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["url"] == "https://example.com/1"


def test_research_node_fallback_on_failure(mocker):
    mocker.patch(
        "podcast_gen_agent.nodes.research._search_topic",
        side_effect=RuntimeError("network down"),
    )

    state = make_initial_state("Robotics", 5)
    result = research_node(state)

    assert "No additional research available" in result["research_data"]
    assert result["sources"] == []
