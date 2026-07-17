"""Notebook kernel bootstrap for Colab and Kaggle."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

COLAB_REPO_DIR = Path("/content/PodcastGen-Agent")
KAGGLE_REPO_DIR = Path("/kaggle/working/PodcastGen-Agent")


def detect_notebook_env() -> str:
    """Return 'kaggle', 'colab', or 'local'."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        return "colab"
    return "local"


def default_repo_dir(env_name: str | None = None) -> Path:
    env = env_name or detect_notebook_env()
    if env == "kaggle":
        return KAGGLE_REPO_DIR
    if env == "colab":
        return COLAB_REPO_DIR
    return Path.cwd() / "PodcastGen-Agent"


def default_work_root(env_name: str | None = None) -> Path:
    env = env_name or detect_notebook_env()
    if env == "kaggle":
        return Path("/kaggle/working")
    if env == "colab":
        return Path("/content")
    return Path.cwd()


def ensure_colab_repo(repo_dir: str | Path | None = None) -> Path:
    """Add the repository root to sys.path for the active Jupyter kernel."""
    root = Path(
        repo_dir
        or os.environ.get("PODCAST_GEN_REPO")
        or default_repo_dir()
    ).resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"Repository not found at {root}. Run the clone/setup cell first."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    os.environ.setdefault("PODCAST_GEN_REPO", root_str)
    return root


def _new_root_collector() -> tuple[list[Path], Callable[[Path | str], None]]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | str) -> None:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append(resolved)

    return roots, add


def candidate_output_dirs(output_dir: Path | str | None = None) -> list[Path]:
    """Return likely output directories for Colab and Kaggle notebooks."""
    roots, add = _new_root_collector()

    if output_dir is not None:
        add(output_dir)
    else:
        try:
            add(ensure_colab_repo() / "outputs")
        except RuntimeError:
            pass

    add(Path.cwd() / "outputs")
    add(COLAB_REPO_DIR / "outputs")
    add(KAGGLE_REPO_DIR / "outputs")

    for content_root in (Path("/content"), Path("/kaggle/working")):
        if not content_root.exists():
            continue
        for manifest in content_root.rglob("run_manifest.json"):
            add(manifest.parent)

    return roots


def find_latest_podcast_mp3(output_dir: Path | str | None = None) -> Path | None:
    """Locate the newest final podcast MP3 without importing podcast_gen_agent."""
    best: Path | None = None
    best_mtime = -1.0
    for root in candidate_output_dirs(output_dir):
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


def diagnose_outputs() -> list[dict[str, object]]:
    """Summarize likely output folders for notebook troubleshooting."""
    report: list[dict[str, object]] = []
    for root in candidate_output_dirs():
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


def download_or_link_file(path: Path) -> None:
    """Trigger a browser download on Colab, or expose a Kaggle FileLink / copy."""
    resolved = path.resolve()
    env = detect_notebook_env()

    if env == "colab":
        from google.colab import files

        files.download(str(resolved))
        return

    if env == "kaggle":
        from IPython.display import FileLink, display

        dest = Path("/kaggle/working") / resolved.name
        if resolved != dest.resolve():
            dest.write_bytes(resolved.read_bytes())
            print(f"Copied to {dest}")
        try:
            link_target = str(dest.relative_to(Path.cwd()))
        except ValueError:
            link_target = dest.name
        display(FileLink(link_target))
        print("Use the link above, or download from the Kaggle Output panel.")
        return

    print(f"File ready at: {resolved}")
