import logging
import shutil
import subprocess
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def configure_pydub() -> str | None:
    """Point pydub at the system ffmpeg binaries when they are available."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg:
        AudioSegment.converter = ffmpeg
    if ffprobe:
        AudioSegment.ffprobe = ffprobe
    return ffmpeg


def write_silent_wav(path: Path, duration_ms: int = 1500, frame_rate: int = 44100) -> None:
    """Write a short silent wav file as a last-resort voice placeholder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate).export(
        str(path),
        format="wav",
    )


def export_mp3(segment: AudioSegment, path: Path, bitrate: str = "192k") -> None:
    """Export mp3 via pydub, falling back to a direct ffmpeg conversion if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        segment.export(str(path), format="mp3", bitrate=bitrate)
        if path.exists() and path.stat().st_size > 0:
            return
    except Exception:
        logger.exception("pydub mp3 export failed for %s; trying ffmpeg CLI", path)

    wav_path = path.with_suffix(".wav")
    segment.export(str(wav_path), format="wav")
    ffmpeg = shutil.which("ffmpeg") or getattr(AudioSegment, "converter", None)
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create mp3 output")

    subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-b:a", bitrate, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    wav_path.unlink(missing_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed to create mp3 output at {path}")


def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize audio to target volume without positive gain clipping."""
    if audio.dBFS == float("-inf"):
        return audio
    change = target_dbfs - audio.dBFS
    if change > 0:
        change = min(change, 6.0)
    return audio.apply_gain(change)


def add_fade(
    audio: AudioSegment,
    fade_in_ms: int = 500,
    fade_out_ms: int = 500,
) -> AudioSegment:
    """Add fade in/out to audio."""
    return audio.fade_in(fade_in_ms).fade_out(fade_out_ms)


def concat_with_silence(segments: list[AudioSegment], silence_ms: int = 300) -> AudioSegment:
    """Join audio segments with silence between them."""
    if not segments:
        return AudioSegment.empty()

    silence = AudioSegment.silent(duration=silence_ms)
    result = segments[0]

    for seg in segments[1:]:
        result += silence + seg

    return result


def normalize_segment(
    seg: AudioSegment,
    frame_rate: int = 44100,
    channels: int = 2,
) -> AudioSegment:
    """Normalize pydub segment format for mixing."""
    return seg.set_frame_rate(frame_rate).set_channels(channels).set_sample_width(2)
