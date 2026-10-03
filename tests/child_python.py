"""Run a snippet in a child interpreter under a chosen PYTHONHASHSEED (for hash-order tests).

For ``python -c`` the working directory precedes PYTHONPATH on sys.path, so the child runs from
the repository root holding this file, with only that root on PYTHONPATH, and checks it
imported the same ``apt_classifier`` as the test process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_child(script: str, *args: str, seed: str) -> str:
    """The child's stdout after the ``apt_classifier`` location line it prints first."""
    import apt_classifier

    completed = subprocess.run(
        [sys.executable, "-c", "import apt_classifier\nprint(apt_classifier.__file__)\n" + script]
        + list(args),
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO_ROOT)},
    )
    loaded, _, output = completed.stdout.partition("\n")
    assert loaded == apt_classifier.__file__, (loaded, apt_classifier.__file__)
    return output
