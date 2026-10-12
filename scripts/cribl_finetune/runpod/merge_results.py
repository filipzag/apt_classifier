"""Merge a RunPod results archive (``remote_run.sh``) into this checkout (README.md here).

New B2 checkpoints go to ``data/models/cribl_ft/B2/``; the B2 log, its dev predictions and the
remote scoring outputs go to ``data/audits/cribl_ft/``. Checkpoint paths in the B2 log are rewritten
to this machine. Existing files are kept unless identical or the B2 log (whose pre-remote copy is
saved as ``B2_train_log_before_remote.json``). Standard library only.

Usage: python3 scripts/cribl_finetune/runpod/merge_results.py <cribl_results.tar.gz>
"""

from __future__ import annotations

import filecmp
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDITS = ROOT / "data" / "audits" / "cribl_ft"
MODELS = ROOT / "data" / "models" / "cribl_ft"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.rsplit("Usage: ", 1)[1])
    archive = Path(sys.argv[1]).expanduser()
    with tempfile.TemporaryDirectory() as scratch:
        with tarfile.open(archive) as bundle:
            bundle.extractall(scratch, filter="data")
        source = Path(scratch)
        remote_audits = source / "data" / "audits" / "cribl_ft"
        remote_models = source / "data" / "models" / "cribl_ft" / "B2"
        added, kept = [], []
        for checkpoint in sorted(remote_models.glob("step_*")):
            target = MODELS / "B2" / checkpoint.name
            if target.exists():
                same = all(
                    filecmp.cmp(item, target / item.name, shallow=False)
                    for item in checkpoint.iterdir()
                    if item.is_file()
                )
                (kept if same else added).append(f"{target} ({'identical' if same else 'DIFFERS'})")
                if not same:
                    raise SystemExit(f"{target} exists and differs from the archive; not merging")
                continue
            shutil.copytree(checkpoint, target)
            added.append(str(target))
        log_path = AUDITS / "B2_train_log.json"
        if log_path.exists():
            shutil.copy2(log_path, AUDITS / "B2_train_log_before_remote.json")
        for item in sorted(remote_audits.iterdir()):
            target = AUDITS / item.name
            if item.name == "B2_train_log.json":
                log = json.loads(item.read_text(encoding="utf-8"))
                for entry in log["checkpoints"]:
                    entry["dir"] = str(MODELS / "B2" / Path(entry["dir"]).name)
                log["selected"] = str(MODELS / "B2" / Path(log["selected"]).name)
                log["remote"] = True
                target.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
                added.append(str(target))
            elif target.exists() and not filecmp.cmp(item, target, shallow=False):
                if item.name.startswith("B2_step") and target.stat().st_size <= item.stat().st_size:
                    shutil.copy2(item, target)  # a dev-prediction file the remote run completed
                    added.append(f"{target} (completed)")
                else:
                    raise SystemExit(f"{target} exists and differs from the archive; not merging")
            elif target.exists():
                kept.append(f"{target} (identical)")
            else:
                shutil.copy2(item, target)
                added.append(str(target))
    print("added or updated:\n  " + "\n  ".join(added))
    if kept:
        print("already present:\n  " + "\n  ".join(kept))


if __name__ == "__main__":
    main()
