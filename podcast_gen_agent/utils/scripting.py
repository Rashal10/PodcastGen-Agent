"""Script length, cleanup, and timestamp helpers."""

from ..config import settings
from ..state import DialogueLine, coerce_script

MIN_LINE_WORDS = 4


def compute_target_words(duration_mins: int) -> int:
    """Target spoken-word count from requested episode length."""
    return max(1, duration_mins) * settings.speech_words_per_minute


def compute_target_lines(duration_mins: int) -> int:
    """Estimate dialogue line count from target episode duration."""
    words = compute_target_words(duration_mins)
    estimated = max(round(words / settings.words_per_line), 6)
    return min(estimated, settings.max_script_lines)


def compute_recursion_limit(duration_mins: int) -> int:
    """LangGraph step budget for research, critic retries, voice loop, music, and assembly."""
    critic_budget = settings.critic_max_retries * 4
    return compute_target_lines(duration_mins) + 25 + critic_budget


def script_word_count(script: list[DialogueLine] | list[dict]) -> int:
    """Count words across dialogue lines."""
    return sum(len(line.text.split()) for line in coerce_script(script))


def estimate_speech_seconds(script: list[DialogueLine] | list[dict]) -> float:
    """Estimate spoken duration from word count."""
    words = script_word_count(script)
    wpm = max(settings.speech_words_per_minute, 1)
    return round((words / wpm) * 60.0, 1)


def refine_script(
    lines: list[DialogueLine],
    max_lines: int | None = None,
) -> list[DialogueLine]:
    """Drop tiny or duplicate lines and cap the script length."""
    limit = max_lines if max_lines is not None else settings.max_script_lines
    seen: set[str] = set()
    cleaned: list[DialogueLine] = []
    for line in lines:
        text = " ".join(line.text.split())
        if len(text.split()) < MIN_LINE_WORDS:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(DialogueLine(speaker=line.speaker, text=text, audio_path=line.audio_path))
        if len(cleaned) >= limit:
            break
    return cleaned


def format_timestamp(ms: int) -> str:
    """Format milliseconds as MM:SS for transcripts."""
    total_seconds = max(0, int(ms / 1000))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def takeaway_lines(script: list[DialogueLine] | list[dict], limit: int = 3) -> list[str]:
    """Pick the most substantial guest (then host) lines as show-note takeaways."""
    lines = coerce_script(script)
    guests = sorted(
        (line for line in lines if line.speaker == "guest"),
        key=lambda line: len(line.text),
        reverse=True,
    )
    hosts = sorted(
        (line for line in lines if line.speaker == "host"),
        key=lambda line: len(line.text),
        reverse=True,
    )
    selected: list[str] = []
    for line in guests + hosts:
        if line.text not in selected:
            selected.append(line.text)
        if len(selected) >= limit:
            break
    return selected
