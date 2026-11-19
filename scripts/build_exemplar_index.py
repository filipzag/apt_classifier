"""Build the exemplar index: labelled real-report sentences for technique mapping.

Reads the public TRAIN splits (tram, expert, derived_procedures) prepared by
``scripts/prepare_public_benchmarks.py`` (tumeteor/mitre-ttp-mapping, CC-BY-4.0). Never the
leaky ``procedures`` split, never dev or test (the acceptance model is fitted on dev and the
public gate runs on test), never CTI-HAL. Rows whose text equals an indexed ATT&CK procedure
sentence are dropped by ``load_cases``. The index is embedded with the configured technique
embedder and written to ``TECHNIQUE_EXEMPLAR_INDEX_PATH``.

    PYTHONPATH=. .venv/bin/python scripts/build_exemplar_index.py

Re-embedding gives a new fingerprint, which disables the acceptance model fitted on the old
index, so an existing output is replaced only with ``--overwrite``, and a path that resolves
outside the repository (a worktree's ``data/models`` symlink) only with ``--allow-shared-output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.embeddings import SentenceTransformerEmbedder  # noqa: E402
from apt_classifier.exemplars import ExemplarIndex, npz_path  # noqa: E402
from apt_classifier.vectorstore import procedure_sentence_texts  # noqa: E402
from scripts.benchmark_mapping import DEFAULT_ROOT, load_cases  # noqa: E402
from scripts.run_tooling import add_output_guard_arguments, enforce_output_guard  # noqa: E402

DATASETS = ("tram", "expert", "derived_procedures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--output", type=Path, help="default: TECHNIQUE_EXEMPLAR_INDEX_PATH")
    add_output_guard_arguments(parser)
    args = parser.parse_args(argv)
    if "procedures" in args.datasets:
        parser.error("the procedures split duplicates indexed ATT&CK text and is not allowed")
    settings = Settings()
    # the file numpy actually writes (it appends .npz), so the guard checks the real target
    output = npz_path(args.output or settings.technique_exemplar_index_path)
    enforce_output_guard(parser, args, [output], repo_root=REPO_ROOT)
    kb = AttackKnowledgeBase.load(settings.kb_path)
    normalized = Path(args.root)
    missing = [d for d in args.datasets if not (normalized / f"{d}_train.jsonl").is_file()]
    if missing:
        parser.error(
            f"public train files missing for {', '.join(missing)} under {normalized}; "
            "run scripts/prepare_public_benchmarks.py first"
        )
    cases, diagnostics = load_cases(
        args.root,
        datasets=set(args.datasets),
        split="train",
        indexed_texts=procedure_sentence_texts(kb),
    )
    manifest_path = next(
        (
            p
            for p in (normalized / "manifest.json", normalized.parent / "manifest.json")
            if p.is_file()
        ),
        None,
    )
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path else {}
    files = [
        {k: item.get(k) for k in ("dataset", "split", "git_blob_sha", "case_count")}
        for item in manifest.get("files", [])
        if item.get("split") == "train" and item.get("dataset") in args.datasets
    ]
    digest = hashlib.sha256(
        json.dumps([[c.case_id, c.text, list(c.expected_technique_ids)] for c in cases]).encode()
    ).hexdigest()
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    index = ExemplarIndex.build(
        ((c.text, c.expected_technique_ids, (c.source or "").split(":")[-1]) for c in cases),
        embedder,
        embedding_model=settings.embedding_model,
        embedding_revision=settings.embedding_model_revision,
        attack_version=str(kb.version),
        extra_metadata={
            "datasets": sorted(args.datasets),
            "split": "train",
            "source": "tumeteor/mitre-ttp-mapping",
            "license": manifest.get("license", "CC-BY-4.0"),
            "source_files": files,
            "cases_hash": digest,
            "case_diagnostics": diagnostics,
        },
    )
    path = index.save(output)
    print(
        f"exemplar index: {len(index)} sentences, {index.metadata['labels']} techniques, "
        f"fingerprint {index.fingerprint} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
