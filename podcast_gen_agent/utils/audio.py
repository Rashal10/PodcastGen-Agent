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

    if silence_ms <= 0:
        result = segments[0]
        for seg in segments[1:]:
            result += seg
        return result

    silence = AudioSegment.silent(duration=silence_ms)
    result = segments[0]

    for seg in segments[1:]:
        result += silence + seg

    return result


def loop_to_length(audio: AudioSegment, length_ms: int) -> AudioSegment:
    """Repeat an audio clip until it covers length_ms, then trim."""
    if length_ms <= 0:
        return AudioSegment.empty()
    if len(audio) <= 0:
        return AudioSegment.silent(duration=length_ms)
    looped = AudioSegment.empty()
    while len(looped) < length_ms:
        looped += audio
    return looped[:length_ms]


def overlay_ducked_bed(
    voice: AudioSegment,
    bed_source: AudioSegment,
    duck_db: int = 14,
    fade_in_ms: int = 1000,
    fade_out_ms: int = 1500,
) -> AudioSegment:
    """Loop bed music under speech and attenuate it so dialogue stays in front."""
    if len(voice) <= 0 or len(bed_source) <= 0:
        return voice

    bed = loop_to_length(bed_source, len(voice)) - abs(duck_db)
    fade_in = min(fade_in_ms, len(bed))
    fade_out = min(fade_out_ms, len(bed))
    if fade_in:
        bed = bed.fade_in(fade_in)
    if fade_out:
        bed = bed.fade_out(fade_out)
    return voice.overlay(bed)


def normalize_segment(
    seg: AudioSegment,
    frame_rate: int = 44100,
    channels: int = 2,
) -> AudioSegment:
    """Normalize pydub segment format for mixing."""
    return seg.set_frame_rate(frame_rate).set_channels(channels).set_sample_width(2)
