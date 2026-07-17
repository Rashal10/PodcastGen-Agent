from podcast_gen_agent.nodes.research import research_node
from podcast_gen_agent.state import make_initial_state


def test_research_node_returns_sources(mocker):
    mocker.patch(
        "podcast_gen_agent.nodes.research.gather_research",
        return_value=(
            "Topic: Quantum Computing\n\nKey Information:\n\n1. Wikipedia: Quantum",
            [
                {
                    "title": "Wikipedia: Quantum computing",
                    "body": "Quantum computing uses qubits.",
                    "url": "https://en.wikipedia.org/wiki/Quantum_computing",
                }
            ],
        ),
    )

    state = make_initial_state("Quantum Computing", 5)
    result = research_node(state)

    assert "research_data" in result
    assert "Quantum Computing" in result["research_data"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["url"].startswith("https://")


def test_research_node_handles_empty_sources(mocker):
    mocker.patch(
        "podcast_gen_agent.nodes.research.gather_research",
        return_value=("Topic: Robotics\nNo additional research available.", []),
    )

    state = make_initial_state("Robotics", 5)
    result = research_node(state)

    assert "No additional research available" in result["research_data"]
    assert result["sources"] == []
