from datetime import datetime
from pydub import AudioSegment

from ..config import OUTPUT_DIR
from ..state import PodcastState


def audio_assembler_node(state: PodcastState) -> dict:
    """Mix all audio into final podcast."""
    print("[Assembler] Starting final mix...")
    
    # load intro
    intro = AudioSegment.from_wav(state["intro_music_path"])
    intro = intro.fade_in(500).fade_out(1000)
    intro = intro - 6  
    
    # load and concat all voice segments
    voice_segments = []
    for seg_path in state["audio_segments"]:
        seg = AudioSegment.from_wav(seg_path)
        voice_segments.append(seg)
        voice_segments.append(AudioSegment.silent(duration=300))  # pause between lines
    
    voice_track = sum(voice_segments, AudioSegment.empty())
    
    # load outro
    outro = AudioSegment.from_wav(state["outro_music_path"])
    outro = outro.fade_in(1000).fade_out(2000)
    outro = outro - 6
    
    # assemble: intro -> voice -> outro
    final = intro + voice_track + outro
    
    # normalize
    final = final.normalize()
    
    # export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = state["topic"].lower().replace(" ", "_")[:30]
    output_path = OUTPUT_DIR / f"podcast_{topic_slug}_{timestamp}.mp3"
    
    final.export(str(output_path), format="mp3", bitrate="192k")
    
    duration_mins = len(final) / 1000 / 60
    print(f"[Assembler] Done! {duration_mins:.1f} min -> {output_path}")
    
    return {"final_audio_path": str(output_path)}
