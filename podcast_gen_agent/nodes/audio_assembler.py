from datetime import datetime
from pydub import AudioSegment

from ..config import OUTPUT_DIR
from ..state import PodcastState


def _normalize_segment(seg: AudioSegment) -> AudioSegment:
    return seg.set_frame_rate(44100).set_channels(2).set_sample_width(2)


def audio_assembler_node(state: PodcastState) -> dict:
    """Mix all audio into final podcast."""
    print("[Assembler] Starting final mix...")
    
    intro = AudioSegment.from_wav(state["intro_music_path"])
    intro = _normalize_segment(intro)
    intro = intro.fade_in(500).fade_out(1000)
    intro = intro - 6  
    
    voice_segments = []
    for seg_path in state["audio_segments"]:
        seg = AudioSegment.from_wav(seg_path)
        seg = _normalize_segment(seg)
        voice_segments.append(seg)
        voice_segments.append(_normalize_segment(AudioSegment.silent(duration=300)))  
    
    voice_track = sum(voice_segments, _normalize_segment(AudioSegment.empty()))
    
    outro = AudioSegment.from_wav(state["outro_music_path"])
    outro = _normalize_segment(outro)
    outro = outro.fade_in(1000).fade_out(2000)
    outro = outro - 6
    
    final = intro + voice_track + outro
    final = final.normalize()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = state["topic"].lower().replace(" ", "_")[:30]
    output_path = OUTPUT_DIR / f"podcast_{topic_slug}_{timestamp}.mp3"
    
    final.export(str(output_path), format="mp3", bitrate="192k")
    
    duration_mins = len(final) / 1000 / 60
    print(f"[Assembler] Done! {duration_mins:.1f} min -> {output_path}")
    
    return {"final_audio_path": str(output_path)}
