"""Build the free-text group model (``apt_classifier/text_group.py``) from ATT&CK.

    PYTHONPATH=. .venv/bin/python scripts/fit_text_group.py [--output PATH]

Uses only the knowledge base and the raw enterprise STIX bundle (software and campaign
procedure examples live only there). Writes ``TEXT_GROUP_PATH`` (gzip JSON: the anonymised
training records, parameters and a fingerprint, pinned to the ATT&CK version); the TF-IDF is
refitted at load. The model votes only with ``TEXT_GROUP_WEIGHT`` > 0, which also needs a
combiner fitted with it (``scripts/fit_attribution_combiner.py`` under the same setting).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.text_group import fit_text_group  # noqa: E402

RAW = REPO_ROOT / "data" / "raw" / "enterprise-attack.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="default: TEXT_GROUP_PATH")
    parser.add_argument("--bundle", type=Path, default=RAW, help="raw enterprise STIX bundle")
    args = parser.parse_args(argv)
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    if not args.bundle.is_file():
        raise SystemExit(f"raw STIX bundle not found: {args.bundle} (run make ingest)")
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    started = time.time()
    model = fit_text_group(kb, bundle)
    path = model.save(args.output or settings.text_group_path)
    print(
        f"text group model -> {path} ({len(model.records)} records "
        f"{model.metadata['records_by_source']}, {len(model.group_ids)} groups, "
        f"ATT&CK {model.metadata['attack_version']}, fingerprint {model.fingerprint}, "
        f"{time.time() - started:.1f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
