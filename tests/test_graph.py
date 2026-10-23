from langgraph.checkpoint.memory import MemorySaver

from podcast_gen_agent.graph import build_graph


def test_graph_compiles_without_state_key_collisions():
    graph = build_graph(checkpointer=MemorySaver())
    assert graph is not None
    assert "critic" in graph.nodes
    assert "generate_script" in graph.nodes
