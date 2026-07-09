from typing import TypedDict, List, Optional
from dataclasses import dataclass


@dataclass
class DialogueLine:
    """Single line of dialogue."""
    speaker: str  # "host" or "guest"
    text: str
    audio_path: Optional[str] = None


class PodcastState(TypedDict):
    """State passed through the LangGraph pipeline."""
    topic: str
    duration_mins: int
    
    research_data: str
    
    script: List[DialogueLine]
    current_line_idx: int
    
    audio_segments: List[str]
    intro_music_path: str
    outro_music_path: str
    
    final_audio_path: str
    
    error: Optional[str]
