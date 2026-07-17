import sqlite3
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import settings
from .nodes.audio_assembler import audio_assembler_node
from .nodes.fail import fail_node
from .nodes.music_generator import music_generator_node
from .nodes.research import research_node
from .nodes.script_generator import script_generator_node
from .nodes.voice_synthesis import should_continue_voice, voice_synthesis_node
from .state import PodcastState


def _route_on_error(state: PodcastState, ok: str) -> Literal["fail", "ok"]:
    return "fail" if state.get("error") else ok


def build_graph(checkpointer: SqliteSaver | None = None) -> CompiledStateGraph:
    """Build the podcast generation workflow with optional checkpointing."""
    graph = StateGraph(PodcastState)

    graph.add_node("research", research_node)
    graph.add_node("generate_script", script_generator_node)
    graph.add_node("voice", voice_synthesis_node)
    graph.add_node("music", music_generator_node)
    graph.add_node("assemble", audio_assembler_node)
    graph.add_node("fail", fail_node)

    graph.set_entry_point("research")

    graph.add_conditional_edges(
        "research",
        lambda state: _route_on_error(state, "generate_script"),
        {"fail": "fail", "generate_script": "generate_script"},
    )
    graph.add_conditional_edges(
        "generate_script",
        lambda state: _route_on_error(state, "voice"),
        {"fail": "fail", "voice": "voice"},
    )
    graph.add_conditional_edges(
        "voice",
        should_continue_voice,
        {
            "continue": "voice",
            "done": "music",
            "fail": "fail",
        },
    )
    graph.add_conditional_edges(
        "music",
        lambda state: _route_on_error(state, "assemble"),
        {"fail": "fail", "assemble": "assemble"},
    )
    graph.add_conditional_edges(
        "assemble",
        lambda state: _route_on_error(state, "end"),
        {"fail": "fail", "end": END},
    )
    graph.add_edge("fail", END)

    if checkpointer is None:
        settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


def get_graph() -> CompiledStateGraph:
    """Return a compiled graph with the default checkpointer."""
    return build_graph()
