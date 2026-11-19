"""Joint government cybersecurity advisories as family-head training rows (project .venv; README.md
here; prereg ``cribl_family_data_2026-10-14.json``).

The FBI's IC3 site lists the joint cybersecurity advisories (CSAs, most co-sealed with CISA and NSA)
as PDFs per year (``https://www.ic3.gov/CSA/<year>``); CISA's own pages refuse automated clients,
so the IC3 copies are used. Their authors tag behaviours inline with ATT&CK ids ("used PowerShell
[T1059.001] to ..."). Each sentence with a bracketed technique tag becomes one row labelled with
those ids; the tags (and tactic, software and footnote brackets) are removed from the text.
Technique tables are not read (their "Use" cells repeat the tagged narrative).

  download:  fetch the listings and every PDF once (one request per 2 s, resumable, logged)
             -> data/benchmarks/advisories/raw/ (local; US government works)
  build:     -> data/audits/cribl_ft/cases_advisories_train.jsonl + cases_manifest.json counts

Leakage guards in ``build``: an advisory that names a CTI-HAL group, one of its ATT&CK aliases or
software ATT&CK links to it and to at most one other group is dropped whole; so is an advisory with
any sentence overlapping a held-out text (public dev/test, CTI-HAL, AnnoCTR dev/test; the rules of
``export_cases.drop_near_duplicates`` and ``export_attack_procedures.drop_overlaps``); the kept
rows pass the same row-level filter. No CTI-HAL text is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

RAW = REPO_ROOT / "data" / "benchmarks" / "advisories" / "raw"
LISTING = "https://www.ic3.gov/CSA/{year}"
PDF = "https://www.ic3.gov/CSA/{path}.pdf"
YEARS = tuple(range(2021, 2027))
USER_AGENT = "apt-classifier-research/1.0 (local training data)"
DELAY_S = 2.0
MIN_TAGS = 5  # an advisory with fewer bracketed technique tags is a guidance or vulnerability note

_PDF_LINK = re.compile(r"/CSA/(20\d\d/\d{6})\.pdf")
TAG = re.compile(r"\s*[\[(]\s*(T\d{4}(?:\.\d{3})?(?:\s*[,;/]\s*T\d{4}(?:\.\d{3})?)*)\s*[\])]")
TECHNIQUE_ID = re.compile(r"T\d{4}(?:\.\d{3})?")
OTHER_BRACKET = re.compile(r"\s*\[(?:TA\d{4}|S\d{4}|G\d{4}|C\d{4}|M\d{4}|\d{1,3})\]")
PAGE_NOISE = re.compile(
    r"^(TLP:\s*\w+.*|Page \d+ of \d+.*|.*Product ID:.*|[\d\s|]+|FBI\s*\|.*|CISA\s*\|.*)$", re.I
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")
_TACTIC = (
    "(?i:reconnaissance|resource development|initial access|execution|persistence|"
    "privilege escalation|defense evasion|stealth|credential access|discovery|lateral movement|"
    "collection|command and control|exfiltration|impact)"
)
_SEPARATOR = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*)"
_UPDATE_MARK = r"\((?:New|Updated)[^)]*\)"
LEADING_HEADING = re.compile(  # "End Update", "(New, May 2, 2024)", "Persistence and Discovery"
    rf"^(?:(?:End Update|Update)\s+|{_UPDATE_MARK}\s*|"
    rf"{_TACTIC}(?:{_SEPARATOR}{_TACTIC})*\s+(?:{_UPDATE_MARK}\s*)?(?=[A-Z]))+"
)


# --- download -----------------------------------------------------------------------------------
def listing_links(html: str, year: int) -> list[str]:
    """``<year>/<6 digits>`` paths of the year's advisory PDFs, in page order, once each."""
    found = [path for path in _PDF_LINK.findall(html) if path.startswith(f"{year}/")]
    return list(dict.fromkeys(found))


def _get(client: Any, url: str) -> Any:
    response = client.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    time.sleep(DELAY_S)
    return response


def download(raw: Path = RAW) -> None:
    import httpx

    raw.mkdir(parents=True, exist_ok=True)
    log_path = raw / "download_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    with httpx.Client(timeout=60.0) as client:
        paths: list[str] = []
        for year in YEARS:
            response = _get(client, LISTING.format(year=year))
            response.raise_for_status()
            paths += listing_links(response.text, year)
        for count, path in enumerate(paths, start=1):
            target = raw / f"ic3_csa_{path.split('/')[1]}.pdf"
            if target.exists() and target.stat().st_size:
                if path not in log:  # fetched earlier (the 10-advisory sample)
                    content = target.read_bytes()
                    log[path] = {
                        "url": PDF.format(path=path),
                        "status": 200,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "fetched_at": "sample",
                    }
                continue
            response = _get(client, PDF.format(path=path))
            entry = {
                "url": PDF.format(path=path),
                "status": response.status_code,
                "bytes": len(response.content),
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            if response.status_code == 200 and response.content[:4] == b"%PDF":
                target.write_bytes(response.content)
                entry["sha256"] = hashlib.sha256(response.content).hexdigest()
            log[path] = entry
            log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"{count}/{len(paths)} {path} {entry['status']} {entry['bytes']}", flush=True)
    ok = sum(1 for entry in log.values() if entry.get("sha256"))
    print(f"{ok} of {len(log)} advisories saved in {raw}")


# --- parsing ------------------------------------------------------------------------------------
def pdf_text(path: Path) -> str:
    """Reading-order text of a PDF (poppler ``pdftotext``)."""
    result = subprocess.run(
        ["pdftotext", str(path), "-"], capture_output=True, check=True, text=True
    )
    return result.stdout


def paragraphs(text: str) -> list[str]:
    """Paragraphs of ``pdftotext`` output: blank lines and page furniture split them, line-end
    hyphens are joined."""
    found: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or PAGE_NOISE.match(stripped):
            if current:
                found.append(" ".join(current))
                current = []
            continue
        if current and re.search(r"[a-z]-$", current[-1]):
            current[-1] = current[-1][:-1] + stripped
        else:
            current.append(stripped)
    if current:
        found.append(" ".join(current))
    return found


def strip_tags(sentence: str) -> str:
    text = TAG.sub("", sentence)
    text = OTHER_BRACKET.sub("", text)
    text = re.sub(r"\s+", " ", text)
    stripped = LEADING_HEADING.sub("", text)
    text = stripped if len(stripped.split()) >= 4 else text  # keep a sentence that is all heading
    return re.sub(r"\s+([.,;:)])", r"\1", text).strip()


def tagged_sentences(text: str) -> list[tuple[str, list[str]]]:
    """(sentence without tags, its technique ids) for every sentence with a bracketed tag."""
    found = []
    for paragraph in paragraphs(text):
        for sentence in SENTENCE_END.split(paragraph):
            ids = [
                item for match in TAG.finditer(sentence) for item in TECHNIQUE_ID.findall(match[1])
            ]
            if ids:
                found.append((strip_tags(sentence), list(dict.fromkeys(ids))))
    return found


def technique_name_only(text: str, names: Iterable[str]) -> bool:
    """A list line that only names a technique ("Indicator Removal: File Deletion")."""
    key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return key in names


def excluded_names(kb: Any, group_ids: Iterable[str]) -> dict[str, list[str]]:
    """group id -> its name, ATT&CK aliases, and the names of software ATT&CK links to it and to at
    most one other group (its own tools, not shared ones such as Cobalt Strike)."""
    found: dict[str, list[str]] = {}
    for group_id in group_ids:
        group = kb.groups[group_id]
        names = [group.name, *group.aliases]
        for software_id in group.software_ids:
            item = kb.software.get(software_id)
            if item is not None and len(set(item.group_ids)) <= 2:
                names += [item.name, *item.aliases]
        found[group_id] = sorted({name for name in names if len(name) >= 4})
    return found


def names_present(text: str, names: Mapping[str, Sequence[str]]) -> list[str]:
    """``group: name`` for every excluded name that occurs as a whole word in ``text``."""
    hits = []
    for group_id, group_names in names.items():
        for name in group_names:
            if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text, re.I):
                hits.append(f"{group_id}: {name}")
    return hits


# --- build --------------------------------------------------------------------------------------
def resolve_ids(ids: Iterable[str], revoked: Mapping[str, str], families: set[str]) -> list[str]:
    """Revoked ids replaced by their successor; ids whose family is not listed dropped."""
    resolved = [revoked.get(item, item) for item in ids]
    return list(dict.fromkeys(item for item in resolved if item.split(".")[0] in families))


def advisory_rows(
    text: str,
    source: str,
    revoked: Mapping[str, str],
    families: set[str],
    technique_names: set[str],
) -> tuple[list[dict[str, Any]], Counter]:
    from clean_cases import broken_reason

    rows, dropped = [], Counter()
    for sentence, ids in tagged_sentences(text):
        gold = resolve_ids(ids, revoked, families)
        if not gold:
            dropped["no_listed_family"] += 1
        elif technique_name_only(sentence, technique_names):
            dropped["technique_name_only"] += 1
        elif not sentence[:1].isalnum() and sentence[:1] not in "\"“'(":
            dropped["starts_mid_sentence"] += 1
        elif reason := broken_reason(sentence):
            dropped[reason] += 1
        else:
            rows.append({"text": sentence, "gold_ids": gold, "source": source})
    return rows, dropped


def build(raw: Path = RAW) -> None:
    from common import read_jsonl, write_jsonl
    from export_attack_procedures import HELD_OUT, drop_overlaps
    from export_cases import OUT, drop_near_duplicates

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from scripts.prepare_elastic_reports import revoked_technique_map

    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    catalog = json.loads((OUT / "catalog.json").read_text(encoding="utf-8"))["techniques"]
    families = {item["family"] for item in catalog}
    technique_names = {
        re.sub(r"[^a-z0-9]+", " ", item["label"].rsplit(" (", 1)[0].lower()).strip()
        for item in catalog
    } | {
        re.sub(r"[^a-z0-9]+", " ", kb.techniques[item["id"]].name.lower()).strip()
        for item in catalog
        if item["id"] in kb.techniques
    }
    revoked = revoked_technique_map(settings.attack_bundle_path)
    manifest_cti_hal = REPO_ROOT / "data" / "benchmarks" / "cti_hal" / "MANIFEST.json"
    cti_hal_groups = sorted(set(json.loads(manifest_cti_hal.read_text())["group_ids"].values()))
    names = excluded_names(kb, cti_hal_groups)
    held_out = [
        row["text"] for split in HELD_OUT for row in read_jsonl(OUT / f"cases_{split}.jsonl")
    ]
    log = json.loads((raw / "download_log.json").read_text(encoding="utf-8"))

    per_advisory: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    sentence_drops: Counter = Counter()
    for path_key, entry in sorted(log.items()):
        pdf = raw / f"ic3_csa_{path_key.split('/')[1]}.pdf"
        if not entry.get("sha256") or not pdf.exists():
            per_advisory[path_key] = {"status": "not_downloaded"}
            continue
        text = pdf_text(pdf)
        tags = sum(len(TECHNIQUE_ID.findall(match[1])) for match in TAG.finditer(text))
        info: dict[str, Any] = {"tags": tags}
        if tags < MIN_TAGS:
            per_advisory[path_key] = {**info, "status": "few_tags"}
            continue
        hits = names_present(text, names)
        if hits:
            per_advisory[path_key] = {**info, "status": "cti_hal_group", "names": hits[:5]}
            continue
        rows, dropped = advisory_rows(text, path_key, revoked, families, technique_names)
        sentence_drops.update(dropped)
        _, near = drop_near_duplicates(rows, held_out)
        if near["held_out_duplicates"]:
            per_advisory[path_key] = {**info, "status": "held_out_overlap", "near": near}
            continue
        per_advisory[path_key] = {**info, "status": "kept", "rows": len(rows)}
        candidates += rows
    survivors, overlaps = drop_overlaps(candidates, held_out)
    kept_ids = {id(row) for row in survivors}
    overlapping = {row["source"] for row in candidates if id(row) not in kept_ids}
    for path_key in overlapping:
        per_advisory[path_key]["status"] = "held_out_overlap"
    candidates = [row for row in candidates if row["source"] not in overlapping]
    rows, repeats = drop_near_duplicates(candidates, held_out)
    out_rows = [
        {
            "case_id": f"advisories-{index:06d}",
            "dataset": "advisories",
            "split": "advisories_train",
            "text": row["text"],
            "gold_ids": row["gold_ids"],
            "candidates": [],
            "gold_in_candidates": False,
            "gold_family_in_candidates": False,
            "pipeline_rank1": None,
            "jev_c1": None,
            "source": f"ic3 CSA {row['source']}",
        }
        for index, row in enumerate(rows, start=1)
    ]
    path = OUT / "cases_advisories_train.jsonl"
    write_jsonl(path, out_rows)
    status = Counter(item["status"] for item in per_advisory.values())
    info = {
        "source": "https://www.ic3.gov/CSA/<year> (joint cybersecurity advisories, US government works)",
        "attack_version": kb.version,
        "advisories": dict(sorted(status.items())),
        "excluded_groups": cti_hal_groups,
        "sentence_drops": dict(sorted(sentence_drops.items())),
        "overlap_drops": {"advisories_with_overlap": len(overlapping), **overlaps},
        "repeats": repeats,
        "cases": len(out_rows),
        "families": len({item.split(".")[0] for row in out_rows for item in row["gold_ids"]}),
        "per_advisory": per_advisory,
    }
    manifest_path = OUT / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("splits", {})["advisories_train"] = info
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{path}: {len(out_rows)} rows, {info['families']} families; advisories {dict(status)}")
    print(f"sentence drops {dict(sentence_drops)}; overlaps {overlaps}; repeats {repeats}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("download", "build"))
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    if args.command == "download":
        download(args.raw)
    else:
        build(args.raw)


if __name__ == "__main__":
    main()
