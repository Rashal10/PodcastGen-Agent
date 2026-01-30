import argparse
import sys

from .graph import podcast_graph
from .config import DEFAULT_DURATION_MINS, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate a podcast from a topic")
    parser.add_argument("topic", nargs="?", help="Topic for the podcast")
    parser.add_argument("--url", help="URL to base podcast on (coming soon)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_MINS,
                        help=f"Target duration in minutes (default: {DEFAULT_DURATION_MINS})")
    
    args = parser.parse_args()
    
    if not args.topic and not args.url:
        parser.print_help()
        sys.exit(1)
    
    topic = args.topic or "Content from URL"  
    
    print("=" * 50)
    print(f"Podcast-Gen-Agent")
    print(f"Topic: {topic}")
    print(f"Duration: ~{args.duration} minutes")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 50)
    
    initial_state = {
        "topic": topic,
        "duration_mins": args.duration,
        "research_data": "",
        "script": [],
        "current_line_idx": 0,
        "audio_segments": [],
        "intro_music_path": "",
        "outro_music_path": "",
        "final_audio_path": "",
        "error": None,
    }
    
    try:
        result = podcast_graph.invoke(initial_state)
        
        print("\n" + "=" * 50)
        print("COMPLETE!")
        print(f"Output: {result['final_audio_path']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
