from pathlib import Path

from ..config import settings


def candidate_output_dirs(extra_dirs: list[Path | str] | None = None) -> list[Path]:
    """Return likely output directories, newest search roots first."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | str) -> None:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    if extra_dirs:
        for path in extra_dirs:
            add(path)

    add(Path.cwd() / "outputs")
    add(settings.output_dir)
    add(Path("/content/PodcastGen-Agent/outputs"))
    add(Path("/kaggle/working/PodcastGen-Agent/outputs"))

    for content_root in (Path("/content"), Path("/kaggle/working")):
        if not content_root.exists():
            continue
        for manifest in content_root.rglob("run_manifest.json"):
            add(manifest.parent)

    return candidates


def find_latest_podcast(
    output_dir: Path | str | None = None,
    *,
    extra_dirs: list[Path | str] | None = None,
) -> Path | None:
    """Return the newest final podcast mp3 under outputs/, including run subfolders."""
    roots = [Path(output_dir).resolve()] if output_dir else candidate_output_dirs(extra_dirs)

    best: Path | None = None
    best_mtime = -1.0
    for root in roots:
        if not root.exists():
            continue

        mp3_files = [path for path in root.rglob("podcast_*.mp3") if path.stat().st_size > 0]
        if not mp3_files:
            mp3_files = [path for path in root.rglob("*.mp3") if path.stat().st_size > 0]

        for path in mp3_files:
            mtime = path.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best = path

    return best


def diagnose_outputs(extra_dirs: list[Path | str] | None = None) -> list[dict[str, object]]:
    """Summarize where the notebook searched for generated podcasts."""
    report: list[dict[str, object]] = []
    for root in candidate_output_dirs(extra_dirs):
        exists = root.exists()
        mp3_files: list[Path] = []
        if exists:
            mp3_files = sorted(
                [path for path in root.rglob("*.mp3") if path.stat().st_size > 0],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        report.append(
            {
                "path": root,
                "exists": exists,
                "mp3_count": len(mp3_files),
                "latest": mp3_files[0] if mp3_files else None,
            }
        )
    return report
