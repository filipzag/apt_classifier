"""Joint government advisories as real-report attribution cases (development data: the combiner's
real-report training lists, prereg ``docs/preregistration/attribution_real_report_combiner_2026-10-22.json``).

Reads the FBI IC3 joint cybersecurity advisories already downloaded by
``scripts/cribl_finetune/prepare_advisories.py download`` (``data/benchmarks/advisories/raw/``, US
government works; nothing is fetched here) and writes ``data/benchmarks/advisories/reports.jsonl``
in the format of ``prepare_cti_hal.py``, read by ``attribution_benchmark.py --report-set
advisories`` and ``fit_attribution_combiner.py --real-reports``.

Rules, fixed before any attribution run:

- an advisory is a narrative one when it has at least ``MIN_TAGS`` sentences with a bracketed
  ATT&CK technique tag (as for the head's training rows);
- its label is the ATT&CK group whose name or an alias (``MIN_NAME_CHARS``+ characters, whole word,
  case-sensitive) occurs at least ``MIN_MENTIONS`` times in the advisory and at least
  ``DOMINANCE`` times as often as any other group's; the group must document at least
  ``MIN_GROUP_TECHNIQUES`` techniques; otherwise the advisory is left out;
- an advisory with a statement that overlaps a CTI-HAL or Elastic statement (the test sets) is
  left out whole (normalised 10-token runs);
- statements: the prose sentences before the first "Mitigations" heading, technique and other
  bracket tags removed, at least ``MIN_WORDS`` words, no hash-like token; each carries its
  paragraph number for the strict anonymiser;
- an advisory re-issued under the same CISA product id (an update) counts once: the latest
  issue is kept;
- ``hidden_texts``: every ATT&CK procedure example (group, software or campaign ``uses`` a
  technique) that cites the advisory, found by its source names (an external reference whose URL
  holds the advisory's IC3 path or its CISA product id, or whose name holds the product id), and
  every one that shares a 10-token run with a statement (citation stripped; joint advisories led
  by another agency carry no CISA product id in the PDF); the benchmark hides them and the
  statements from the indexes and the text-based members.

    PYTHONPATH=. .venv/bin/python scripts/prepare_advisory_reports.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "cribl_finetune"))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402

RAW = REPO_ROOT / "data" / "benchmarks" / "advisories" / "raw"
OUTPUT = REPO_ROOT / "data" / "benchmarks" / "advisories"
TEST_SETS = (
    REPO_ROOT / "data" / "benchmarks" / "cti_hal" / "reports.jsonl",
    REPO_ROOT / "data" / "benchmarks" / "elastic" / "reports.jsonl",
    REPO_ROOT / "data" / "benchmarks" / "elastic" / "reports_open_set.jsonl",
)
MIN_TAGS = 5
MIN_NAME_CHARS = 3
MIN_MENTIONS = 10
DOMINANCE = 3.0
MIN_GROUP_TECHNIQUES = 5
MIN_WORDS = 6
SHARED_RUN = 10
LICENCE = "US government works (FBI IC3 joint cybersecurity advisories); local copies"

PRODUCT_ID = re.compile(r"\bAA\d{2}-\d{3}[A-Z]\b")
MITIGATIONS = re.compile(r"^\s*mitigations\b", re.I)
HASH_LIKE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")
WORD = re.compile(r"[a-z0-9]+")


def group_patterns(kb: AttackKnowledgeBase) -> dict[str, re.Pattern[str]]:
    """group id -> whole-word, case-sensitive pattern over its name and aliases."""
    patterns = {}
    for group_id, group in kb.groups.items():
        names = sorted(
            {
                n.strip()
                for n in [group.name, *group.aliases]
                if n and len(n.strip()) >= MIN_NAME_CHARS
            },
            key=len,
            reverse=True,
        )
        if names:
            patterns[group_id] = re.compile(
                r"(?<![\w-])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w-])"
            )
    return patterns


def label(
    text: str, kb: AttackKnowledgeBase, patterns: Mapping[str, re.Pattern[str]]
) -> tuple[str | None, dict[str, int]]:
    """The advisory's group under the registered rule, and the top mention counts."""
    counts = Counter({g: len(p.findall(text)) for g, p in patterns.items()})
    counts = Counter({g: n for g, n in counts.items() if n})
    top = counts.most_common(5)
    if not top:
        return None, {}
    leader, mentions = top[0]
    runner = top[1][1] if len(top) > 1 else 0
    documented = len(getattr(kb.groups[leader], "technique_ids", ()) or ())
    ok = (
        mentions >= MIN_MENTIONS
        and mentions >= DOMINANCE * runner
        and documented >= MIN_GROUP_TECHNIQUES
    )
    return (leader if ok else None), dict(top)


def statements(text: str) -> list[dict[str, Any]]:
    """Prose sentences before the first Mitigations heading, tags removed."""
    from prepare_advisories import paragraphs, strip_tags

    found: list[dict[str, Any]] = []
    for number, paragraph in enumerate(paragraphs(text)):
        if MITIGATIONS.match(paragraph):
            break
        for sentence in SENTENCE_END.split(paragraph):
            clean = strip_tags(sentence)
            if len(clean.split()) < MIN_WORDS or HASH_LIKE.search(clean):
                continue
            if not clean[:1].isalnum() and clean[:1] not in "\"“'(":
                continue
            found.append(
                {
                    "text": clean,
                    "technique_ids": [],
                    "paragraph": number,
                    "section_attribution": False,
                }
            )
    return found


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def runs(seq: list[str], size: int = SHARED_RUN) -> set[tuple[str, ...]]:
    return {tuple(seq[i : i + size]) for i in range(len(seq) - size + 1)}


def test_runs(paths: Iterable[Path]) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for item in json.loads(line).get("statements") or []:
                    found |= runs(tokens(str(item.get("text") or "")))
    return found


def citing_names(bundle: Mapping[str, Any], ic3_path: str, product_id: str | None) -> set[str]:
    """Source names under which ATT&CK cites the advisory."""
    needles = [f"ic3.gov/csa/{ic3_path}".lower()]
    if product_id:
        needles.append(product_id.lower())
    names = set()
    for item in bundle.get("objects", []):
        for ref in item.get("external_references") or []:
            url = str(ref.get("url") or "").lower()
            name = str(ref.get("source_name") or "")
            cites = any(n in url for n in needles) or bool(
                product_id and product_id.lower() in name.lower()
            )
            if cites and name:
                names.add(name)
    return names


def procedure_examples(bundle: Mapping[str, Any]) -> list[tuple[str, set[tuple[str, ...]]]]:
    """Every group, software and campaign procedure example (raw ATT&CK text) with the
    10-token runs of its citation-stripped text."""
    from apt_classifier.attack.parse import strip_attack_citations

    kinds = {
        item["id"]: item.get("type")
        for item in bundle.get("objects", [])
        if "id" in item and not item.get("revoked") and not item.get("x_mitre_deprecated")
    }
    found = []
    for item in bundle.get("objects", []):
        if item.get("type") != "relationship" or item.get("relationship_type") != "uses":
            continue
        if item.get("revoked") or item.get("x_mitre_deprecated"):
            continue
        if kinds.get(item.get("source_ref")) not in (
            "intrusion-set",
            "malware",
            "tool",
            "campaign",
        ):
            continue
        if kinds.get(item.get("target_ref")) != "attack-pattern":
            continue
        description = item.get("description") or ""
        if description:
            found.append((description, runs(tokens(strip_attack_citations(description)))))
    return found


def hidden_procedures(
    examples: list[tuple[str, set[tuple[str, ...]]]],
    names: Iterable[str],
    statement_runs: set[tuple[str, ...]],
) -> tuple[list[str], int, int]:
    """The examples citing any of ``names`` or sharing a 10-token run with a statement;
    returns them with the two counts (cited, copied without a matching citation)."""
    markers = [f"(Citation: {name})" for name in names]
    found: set[str] = set()
    cited = copied = 0
    for description, example_runs in examples:
        if any(marker in description for marker in markers):
            cited += description not in found
            found.add(description)
        elif example_runs & statement_runs:
            copied += description not in found
            found.add(description)
    return sorted(found), cited, copied


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(raw: Path, output: Path, bundle_path: Path) -> dict[str, Any]:
    from prepare_advisories import pdf_text, tagged_sentences

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    patterns = group_patterns(kb)
    held = test_runs(TEST_SETS)
    examples = procedure_examples(bundle)
    rows: list[dict[str, Any]] = []
    per_advisory: dict[str, Any] = {}
    status: Counter[str] = Counter()
    for pdf in sorted(raw.glob("ic3_csa_*.pdf")):
        stem = pdf.stem.removeprefix("ic3_csa_")
        ic3_path = f"20{stem[:2]}/{stem}"
        text = pdf_text(pdf)
        tags = tagged_sentences(text)
        if len(tags) < MIN_TAGS:
            status["few_tags"] += 1
            continue
        group_id, top = label(text, kb, patterns)
        info: dict[str, Any] = {"tags": len(tags), "mentions": top, "sha256": _sha256(pdf)}
        if group_id is None:
            status["no_dominant_group"] += 1
            per_advisory[ic3_path] = {**info, "status": "no_dominant_group"}
            continue
        items = statements(text)
        overlap = sum(1 for item in items if runs(tokens(item["text"])) & held)
        if overlap:
            status["test_overlap"] += 1
            per_advisory[ic3_path] = {**info, "status": "test_overlap", "statements": overlap}
            continue
        if len(items) < 3:
            status["few_statements"] += 1
            per_advisory[ic3_path] = {**info, "status": "few_statements"}
            continue
        header = text[:4000]
        product = PRODUCT_ID.search(header) or PRODUCT_ID.search(text)
        product_id = product.group(0) if product else None
        names = citing_names(bundle, ic3_path, product_id)
        statement_runs = set().union(*(runs(tokens(item["text"])) for item in items))
        hidden, cited, copied = hidden_procedures(examples, names, statement_runs)
        report_id = f"advisory-{stem}"
        rows.append(
            {
                "_product": product_id,
                "report_id": report_id,
                "group_id": group_id,
                "accept": [],
                "kind": "advisory",
                "title": f"IC3 CSA {ic3_path}" + (f" ({product_id})" if product_id else ""),
                "statements": items,
                "hidden_texts": hidden,
            }
        )
        status["kept"] += 1
        per_advisory[ic3_path] = {
            **info,
            "status": "kept",
            "group_id": group_id,
            "group": kb.groups[group_id].name,
            "product_id": product_id,
            "citing_source_names": sorted(names),
            "hidden_texts": len(hidden),
            "hidden_cited": cited,
            "hidden_copied": copied,
            "statements": len(items),
        }
    # an update re-issued under the same product id counts once (the latest issue)
    latest: dict[str, str] = {}
    for row in rows:
        if row["_product"]:
            latest[row["_product"]] = max(latest.get(row["_product"], ""), row["report_id"])
    kept_rows = []
    for row in rows:
        product = row.pop("_product")
        key = row["report_id"].removeprefix("advisory-")
        ic3 = f"20{key[:2]}/{key}"
        if product and latest[product] != row["report_id"]:
            status["kept"] -= 1
            status["superseded_update"] += 1
            per_advisory[ic3]["status"] = "superseded_update"
            continue
        kept_rows.append(row)
    rows = kept_rows
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reports.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "source": "FBI IC3 joint cybersecurity advisories (data/benchmarks/advisories/raw)",
        "licence": LICENCE,
        "role": "development: the combiner's real-report training lists (never CTI-HAL or Elastic)",
        "prepared_at": datetime.now(UTC).isoformat(),
        "attack_version": kb.version,
        "rules": {
            "min_tags": MIN_TAGS,
            "min_name_chars": MIN_NAME_CHARS,
            "min_mentions": MIN_MENTIONS,
            "dominance": DOMINANCE,
            "min_group_techniques": MIN_GROUP_TECHNIQUES,
            "min_words": MIN_WORDS,
            "test_overlap_run": SHARED_RUN,
        },
        "rules_sha256": _sha256(Path(__file__)),
        "status": dict(status),
        "reports": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "statements": sum(len(row["statements"]) for row in rows),
        "hidden_texts": sum(len(row["hidden_texts"]) for row in rows),
        "outputs_sha256": {path.name: _sha256(path)},
        "per_advisory": per_advisory,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--bundle", type=Path, default=REPO_ROOT / "data" / "raw" / "enterprise-attack.json"
    )
    args = parser.parse_args(argv)
    manifest = build(args.raw, args.output, args.bundle)
    print(
        f"{manifest['reports']} advisories, {manifest['groups']} groups, "
        f"{manifest['statements']} statements, {manifest['hidden_texts']} hidden texts; "
        f"status {manifest['status']}"
    )
    for key, info in manifest["per_advisory"].items():
        if info["status"] == "kept":
            print(
                f"  {key} {info['group']} ({info['group_id']}) {info['statements']} statements, "
                f"{info['hidden_texts']} hidden, product {info['product_id']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
