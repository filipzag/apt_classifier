"""Serve a technique head checkpoint to the product (``.venv-cribl``, GPU; docs 21.3.18).

  python scripts/cribl_finetune/serve.py --checkpoint data/models/cribl_ft/E2/step_52706
  (make technique-head-serve)

Loads the checkpoint as ``family_head.py evaluate`` does (4-bit base + LoRA + family head +
technique head) and answers ``/health`` and ``/score`` on localhost (``head_service.py``). The
product's client is :mod:`apt_classifier.technique_head`; it is used only with
``TECHNIQUE_HEAD_ENABLED=true``. The model stays loaded for the life of the process (about 3.5 GB
of GPU memory for the 4B base); requests are served one at a time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import family_head
import torch
from decision_model import with_oom_retry
from head_service import DEFAULT_PORT, HeadService, Scorer, checkpoint_fingerprint, serve

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "models" / "cribl_ft" / "E2" / "step_52706"


def build_scorer(checkpoint: Path, *, batch: int, top_kept: int) -> tuple[Scorer, dict[str, Any]]:
    """``(scorer, manifest)`` for :class:`head_service.HeadService`."""
    family_head.TOP_KEPT = int(top_kept)
    model, encoder, head, info = family_head.load_checkpoint(checkpoint)
    families = info["families"]
    techniques = family_head.load_techniques(checkpoint, encoder.device)
    if techniques is None:
        raise SystemExit(f"{checkpoint} carries no technique head; the product needs one")
    if str(info.get("task", "technique")) != "technique":
        raise SystemExit(f"{checkpoint} is a {info.get('task')} head, not the technique head")

    def scorer(texts: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for start in range(0, len(texts), batch):
            chunk = [
                {"case_id": str(start + i), "dataset": "service", "gold_ids": [], "text": text}
                for i, text in enumerate(texts[start : start + batch])
            ]
            with torch.inference_mode():
                predicted = with_oom_retry(
                    lambda cases=chunk: family_head.predict_batch(
                        encoder, head, families, cases, techniques
                    ),
                    note="serve",
                )
            for row in predicted:
                if row.get("status") != "ok":
                    rows.append({"status": "error", "error": str(row.get("error") or "failed")})
                    continue
                rows.append(
                    {
                        "status": "ok",
                        "families": row["top"],
                        "techniques": row.get("technique_top", []),
                        "pick": row["pick"],
                        "p_pick": row["p_pick"],
                        "technique_pick": row.get("technique_pick"),
                        "technique_p": row.get("technique_p"),
                    }
                )
        return rows

    manifest = {
        "fingerprint": checkpoint_fingerprint(checkpoint),
        "checkpoint": str(
            checkpoint.relative_to(REPO_ROOT)
            if checkpoint.is_relative_to(REPO_ROOT)
            else checkpoint
        ),
        "base": info.get("base"),
        "adapter": info.get("adapter_dir"),
        "lora": info.get("lora"),
        "attack_version": str(info.get("attack_version", "")),
        "task": str(info.get("task", "technique")),
        "families": len(families),
        "techniques": len(techniques.ids),
        # the ids the head can answer with: a newer knowledge base accepts the head only when
        # every one of them still exists (apt_classifier.technique_head.newer_release_problem)
        "technique_ids": [str(t) for t in techniques.ids],
        "families_considered": techniques.considered,
        "top_kept": int(top_kept),
        "batch": int(batch),
    }
    return scorer, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--batch", type=int, default=16, help="sentences per forward pass")
    parser.add_argument(
        "--top-kept", type=int, default=family_head.TOP_KEPT, help="families/techniques per row"
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory not found: {checkpoint}")
    started = time.monotonic()
    scorer, manifest = build_scorer(checkpoint, batch=args.batch, top_kept=args.top_kept)
    warm = scorer(["The adversary used PowerShell to download a second-stage payload."])
    print(
        f"technique head ready in {time.monotonic() - started:.0f}s: {manifest['checkpoint']} "
        f"fingerprint {manifest['fingerprint'][:12]}... ATT&CK {manifest['attack_version']} "
        f"{manifest['families']} families, {manifest['techniques']} techniques; warm-up pick "
        f"{warm[0].get('technique_pick')} (P {warm[0].get('technique_p', 0):.2f}); "
        f"serving on http://{args.host}:{args.port}",
        flush=True,
    )
    service = HeadService(scorer, manifest)
    try:
        serve(service, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print(f"stopped after {service.requests} requests, {service.scored} texts", file=sys.stderr)


if __name__ == "__main__":
    main()
