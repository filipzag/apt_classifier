"""Small shared helpers: the memory guard every script calls first."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/home/fzagar/apt_classifier/data")
MIN_FREE_MB = 900


def mem_available_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def require_memory(min_mb: int = MIN_FREE_MB) -> None:
    """Stop before the laptop runs short of memory (it froze on a 1.1 GB job once)."""
    free = mem_available_mb()
    if free is not None and free < min_mb:
        raise SystemExit(f"only {free} MB available, need {min_mb} MB: stopping")
