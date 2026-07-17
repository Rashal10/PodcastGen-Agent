from pathlib import Path


def find_latest_podcast(output_dir: Path | str | None = None) -> Path | None:
    """Return the newest podcast mp3 under outputs/, including run subfolders."""
    root = Path(output_dir) if output_dir else Path("outputs")
    if not root.exists():
        return None

    mp3_files = list(root.rglob("*.mp3"))
    if not mp3_files:
        return None

    return max(mp3_files, key=lambda path: path.stat().st_mtime)
