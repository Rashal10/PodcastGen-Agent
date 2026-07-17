from pathlib import Path

from podcast_gen_agent.compat import ensure_transformers_compat
from podcast_gen_agent.utils.output import find_latest_podcast


def test_ensure_transformers_compat_patches_missing_symbols(monkeypatch):
    import transformers.pytorch_utils as pytorch_utils
    import transformers.utils.import_utils as import_utils

    if hasattr(pytorch_utils, "isin_mps_friendly"):
        monkeypatch.delattr(pytorch_utils, "isin_mps_friendly", raising=False)
    if hasattr(import_utils, "is_torch_greater_or_equal"):
        monkeypatch.delattr(import_utils, "is_torch_greater_or_equal", raising=False)

    ensure_transformers_compat()

    assert hasattr(pytorch_utils, "isin_mps_friendly")
    assert hasattr(import_utils, "is_torch_greater_or_equal")
    assert import_utils.is_torch_greater_or_equal("0.0.0") is True
    assert import_utils.is_torch_greater_or_equal("99.0.0") is False


def test_find_latest_podcast_nested(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    older = run_a / "podcast_old.mp3"
    newer = run_b / "podcast_new.mp3"
    older.write_bytes(b"old")
    newer.write_bytes(b"newer-content")

    import os
    import time

    os.utime(older, (time.time() - 10, time.time() - 10))
    os.utime(newer, (time.time(), time.time()))

    latest = find_latest_podcast(tmp_path)
    assert latest == newer
