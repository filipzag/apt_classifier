"""Build the lean core's masked corpus for one store pin and level (cached under data/core/).

scripts/core_watch.sh scripts/core_py scripts/core_corpus.py --pin before_2025 --level behavior
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vestige_core.util import require_memory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", choices=["before_2025", "before_2026_03", "whole"], required=True)
    parser.add_argument("--level", choices=["strict", "behavior"], required=True)
    args = parser.parse_args()
    require_memory()
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.knowledge_store.members import StoreMembers, members_path
    from vestige_core.corpus import load_rows

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    pin = None if args.pin == "whole" else args.pin
    with gzip.open(members_path(settings), "rt", encoding="utf-8") as handle:
        names = tuple(json.load(handle).get("masking_names") or ())
    members = StoreMembers.load(members_path(settings, pin), kb)
    start = time.time()
    rows = load_rows(members, kb, pin, args.level, names if args.level == "behavior" else ())
    print(
        json.dumps(
            {
                "pin": args.pin,
                "level": args.level,
                "rows": len(rows),
                "store_rows": len(members.text_rows),
                "seconds": round(time.time() - start),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
