from langgraph.graph import StateGraph, END

from .state import PodcastState
from .nodes.research import research_node
from .nodes.script_generator import script_generator_node
from .nodes.voice_synthesis import voice_synthesis_node, should_continue_voice
from .nodes.music_generator import music_generator_node
from .nodes.audio_assembler import audio_assembler_node


def build_graph() -> StateGraph:
    """Build the podcast generation workflow."""
    
    graph = StateGraph(PodcastState)
    
    graph.add_node("research", research_node)
    graph.add_node("script", script_generator_node)
    graph.add_node("voice", voice_synthesis_node)
    graph.add_node("music", music_generator_node)
    graph.add_node("assemble", audio_assembler_node)
    
    graph.set_entry_point("research")
    
    graph.add_edge("research", "script")
    graph.add_edge("script", "voice")
    
    graph.add_conditional_edges(
        "voice",
        should_continue_voice,
        {
            "continue": "voice",
            "done": "music",
        }
    )
    
    graph.add_edge("music", "assemble")
    graph.add_edge("assemble", END)
    
    return graph.compile()


podcast_graph = build_graph()