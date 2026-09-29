from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.evaluation import AttackIdMap, TechniqueEvaluationCase, pipeline_fingerprint

REPOSITORY = "tumeteor/mitre-ttp-mapping"
REVISION = "a16856a6438ca2b7888c5cadfba6d7c854f04a55"
LICENSE = "CC-BY-4.0"
BASE_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}"


@dataclass(frozen=True, slots=True)
class PublicSource:
    dataset: str
    split: str
    path: str
    git_blob_sha: str


SOURCES = (
    PublicSource(
        "tram", "train", "datasets/tram/tram_train.tsv", "e3fa562ee542870aa0470045108c47c5b7214f36"
    ),
    PublicSource(
        "tram", "dev", "datasets/tram/tram_dev.tsv", "79fe946d9d286ef5904b7c6c21c3c75268259a86"
    ),
    PublicSource(
        "tram", "test", "datasets/tram/tram_test.tsv", "0a8441cf66e80cbbc5f72e4af959846059f180d4"
    ),
    PublicSource(
        "expert",
        "train",
        "datasets/expert/expert_train.tsv",
        "a60b11dcdca8f5b05b1cebeb789d05bede8a7da9",
    ),
    PublicSource(
        "expert",
        "dev",
        "datasets/expert/expert_dev.tsv",
        "810ccb963906da8583c619039305262daeb24593",
    ),
    PublicSource(
        "expert",
        "test",
        "datasets/expert/expert_test.tsv",
        "ef89f0c3df66ac3c466964b4dd52ebc27d97e753",
    ),
    PublicSource(
        "procedures",
        "train",
        "datasets/procedures/procedure_train.tsv",
        "d71890f095bd217dfc28d5c47fa57596284ebbfc",
    ),
    PublicSource(
        "procedures",
        "dev",
        "datasets/procedures/procedure_dev.tsv",
        "c9d99c9e2e83ef95545df783593a49776a7d836c",
    ),
    PublicSource(
        "procedures",
        "test",
        "datasets/procedures/procedure_test.tsv",
        "70f0e4d7ef3344018494b96a7a518286d8e8dc6b",
    ),
    PublicSource(
        "derived_procedures",
        "train",
        "datasets/derived_procedures/derived_procedure_train.tsv",
        "c92580b51953b65233ec27b456f84236aa293bbf",
    ),
    PublicSource(
        "derived_procedures",
        "dev",
        "datasets/derived_procedures/derived_procedure_dev.tsv",
        "ee46500ba6a6b529f0cf893078d3222aa0b0fa0a",
    ),
    PublicSource(
        "derived_procedures",
        "test",
        "datasets/derived_procedures/derived_procedure_test.tsv",
        "9222bf09b0e6f4e88bdd5b1ddfaae41baab8f7e6",
    ),
)


def prepare(
    destination: Path,
    *,
    kb_path: Path | None = None,
    bundle_path: Path | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir = destination / "raw"
    normalized_dir = destination / "normalized"
    raw_dir.mkdir(exist_ok=True)
    normalized_dir.mkdir(exist_ok=True)
    kb = AttackKnowledgeBase.load(kb_path) if kb_path else None
    id_map = AttackIdMap.from_bundle(bundle_path) if bundle_path else None
    files = []
    seen_text: dict[str, tuple[str, str]] = {}
    cross_split_duplicates: list[dict[str, str]] = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for source in SOURCES:
            raw_path = raw_dir / f"{source.dataset}_{source.split}.tsv"
            content = _download(client, source)
            raw_path.write_bytes(content)
            cases, remap_count, unscorable_count = _normalize(content, source, kb, id_map)
            normalized_path = normalized_dir / f"{source.dataset}_{source.split}.jsonl"
            normalized_path.write_text(
                "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases),
                encoding="utf-8",
            )
            for case in cases:
                digest = hashlib.sha256(_normalized_text(case["text"]).encode()).hexdigest()
                previous = seen_text.get(digest)
                if previous and previous[1] != source.split:
                    cross_split_duplicates.append(
                        {
                            "case_id": case["case_id"],
                            "split": source.split,
                            "previous_case_id": previous[0],
                            "previous_split": previous[1],
                        }
                    )
                else:
                    seen_text[digest] = (case["case_id"], source.split)
            files.append(
                {
                    **asdict(source),
                    "raw_path": str(raw_path),
                    "normalized_path": str(normalized_path),
                    "case_count": len(cases),
                    "remapped_label_count": remap_count,
                    "unscorable_label_count": unscorable_count,
                }
            )
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "document_grouping": "publisher-provided splits; processed TSV files do not expose source document IDs",
        "files": files,
        "cross_split_exact_duplicates": cross_split_duplicates,
        "attack_version": kb.version if kb else None,
    }
    manifest["fingerprint"] = pipeline_fingerprint(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _download(client: httpx.Client, source: PublicSource) -> bytes:
    response = client.get(f"{BASE_URL}/{source.path}")
    response.raise_for_status()
    content = response.content
    if len(content) > 5_000_000:
        raise ValueError(f"public benchmark file too large: {source.path}")
    observed = _git_blob_sha(content)
    if observed != source.git_blob_sha:
        raise ValueError(
            f"checksum mismatch for {source.path}: expected {source.git_blob_sha}, got {observed}"
        )
    return content


def _normalize(
    content: bytes,
    source: PublicSource,
    kb: AttackKnowledgeBase | None,
    id_map: AttackIdMap | None,
) -> tuple[list[dict[str, Any]], int, int]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), delimiter="\t")
    output: list[dict[str, Any]] = []
    remap_count = 0
    unscorable_count = 0
    for index, row in enumerate(reader, 1):
        case = TechniqueEvaluationCase.from_dict(
            {
                "case_id": f"{source.dataset}-{source.split}-{index:06d}",
                "document_id": f"{source.dataset}-{source.split}-{index:06d}",
                "text": row.get("text1") or row.get("text") or "",
                "labels": row.get("labels") or "[]",
                "source": f"{REPOSITORY}:{source.dataset}",
                "label_quality": "public_dataset",
                "split": source.split,
            }
        )
        if not case.text:
            continue
        labels = case.expected_technique_ids
        replacements: dict[str, str] = {}
        unscorable: tuple[str, ...] = ()
        if kb is not None and id_map is not None:
            remap = id_map.remap(labels, kb.techniques)
            labels = remap.mapped
            replacements = remap.replacements
            unscorable = remap.unscorable
            remap_count += len(replacements)
            unscorable_count += len(unscorable)
        output.append(
            {
                **asdict(case),
                "expected_technique_ids": list(labels),
                "original_technique_ids": list(case.expected_technique_ids),
                "label_replacements": replacements,
                "unscorable_technique_ids": list(unscorable),
            }
        )
    return output, remap_count, unscorable_count


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare pinned public ATT&CK mapping benchmarks")
    parser.add_argument("--destination", type=Path, default=Path("data/benchmarks/public"))
    parser.add_argument("--kb", type=Path, default=Path("data/kb/attack_kb.json"))
    parser.add_argument("--bundle", type=Path, default=Path("data/raw/enterprise-attack.json"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    manifest = prepare(
        args.destination,
        kb_path=args.kb if args.kb.exists() else None,
        bundle_path=args.bundle if args.bundle.exists() else None,
        timeout=args.timeout,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
