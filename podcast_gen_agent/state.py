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
    
    # research phase
    research_data: str
    
    # script phase
    script: List[DialogueLine]
    current_line_idx: int
    
    # audio phase
    audio_segments: List[str]  # paths to wav files
    intro_music_path: str
    outro_music_path: str
    
    # final output
    final_audio_path: str
    
    # status tracking
    error: Optional[str]
