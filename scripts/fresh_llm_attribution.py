"""Zero-shot frontier-LLM attribution on the fresh reports (the head-to-head arm).

Prereg ``fresh_reports_head_to_head_2026-10-28.json`` (amended 2026-10-29: GPT-5.6 Sol on every
report, Claude Opus 5.5 on the reports published from 2026-07-01, ``--since``). The model reads exactly the statements the
product reads: the same report file, the same statement selection and anonymisation
(``attribution_benchmark.load_report_folds`` + ``anonymize_report_folds`` at the given level), all
statements, in report order. It returns up to three ATT&CK groups and a decision: name the first
one, leave it open, or say the actor is not an ATT&CK group. One call per report, cached per
report (``--cache``), so a stopped run resumes and a rerun costs nothing.

    PYTHONPATH=. .venv/bin/python scripts/fresh_llm_attribution.py --dry-run
    OPENROUTER_API_KEY=... PYTHONPATH=. .venv/bin/python scripts/fresh_llm_attribution.py \
        --model openai/gpt-5.6-sol --max-cost 30 --output data/audits/fresh_h2h/llm_gpt56sol_strict.json
    ... --model anthropic/claude-opus-5.5 --since 2026-07-01 --output .../llm_opus55_strict.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from scripts.attribution_benchmark import (  # noqa: E402
    anonymize_report_folds,
    load_report_folds,
)
from scripts.report_anonymizer import ReportAnonymizer  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.6-sol"
FRESH = REPO_ROOT / "data" / "benchmarks" / "fresh"
SYSTEM = "You are an experienced cyber threat intelligence analyst."
PROMPT = """Below are the statements of one threat report, in report order. Every threat-actor \
name has been replaced with "the adversary", and statements about the attribution itself may \
have been removed.

Which MITRE ATT&CK Enterprise group (ATT&CK v20, group IDs G0001-G1099) carried out the activity \
described?

Answer with up to three candidate groups, most likely first, each with its ATT&CK group ID, its \
ATT&CK name and your probability, and one decision:
- "name": you would name the first candidate as the responsible group in a report to a customer;
- "open": the evidence does not single out one group;
- "not_in_attack": the activity most likely belongs to an actor ATT&CK does not document as a \
group.

Statements:
{statements}"""
SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "group_name": {"type": "string"},
                    "probability": {"type": "number"},
                },
                "required": ["group_id", "group_name", "probability"],
                "additionalProperties": False,
            },
        },
        "decision": {"type": "string", "enum": ["name", "open", "not_in_attack"]},
        "rationale": {"type": "string"},
    },
    "required": ["candidates", "decision", "rationale"],
    "additionalProperties": False,
}


def report_texts(path: Path, kb: AttackKnowledgeBase, level: str) -> list[dict[str, Any]]:
    """The statements the product reads for each report at ``level``."""
    published = {
        row["report_id"]: row.get("published")
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }
    anonymizer = ReportAnonymizer(kb, level) if level in ("strict", "tradecraft") else None
    folds = load_report_folds(path, max_observations=None, anonymizer=anonymizer, kb=kb)
    folds, _, _ = anonymize_report_folds(folds, kb, level=level)
    return [
        {
            "report_id": fold.report_id,
            "group_id": fold.group_id,
            "published": published.get(fold.report_id),
            "statements": list(fold.observations),
        }
        for fold in folds
    ]


def prompt_for(statements: list[str]) -> str:
    return PROMPT.format(
        statements="\n".join(f"{index}. {text}" for index, text in enumerate(statements, 1))
    )


def name_index(kb: AttackKnowledgeBase) -> dict[str, str]:
    index: dict[str, set[str]] = {}
    for group_id, group in kb.groups.items():
        for name in (group.name, *group.aliases):
            if name:
                index.setdefault(name.strip().lower(), set()).add(group_id)
    return {name: next(iter(ids)) for name, ids in index.items() if len(ids) == 1}


def map_candidate(candidate: dict[str, Any], kb: AttackKnowledgeBase, names: dict[str, str]):
    """The ATT&CK group a candidate names: its ID when it is a current group, else its name."""
    group_id = str(candidate.get("group_id") or "").strip().upper()
    if re.fullmatch(r"G\d{4}", group_id) and group_id in kb.groups:
        return group_id
    return names.get(str(candidate.get("group_name") or "").strip().lower())


def call(model: str, prompt: str, key: str, *, effort: str, timeout: float = 600.0):
    import httpx

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "attribution", "strict": True, "schema": SCHEMA},
        },
        "reasoning": {"effort": effort},
        "max_tokens": 16000,
        "usage": {"include": True},
        "provider": {"require_parameters": True},
    }
    for attempt in range(3):
        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
        except httpx.HTTPError as error:
            if attempt == 2:
                return {"error": f"{type(error).__name__}: {error}"[:300]}
            time.sleep(10 * (attempt + 1))
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(15 * (attempt + 1))
            continue
        if response.status_code != 200:
            return {"error": f"http_{response.status_code}: {response.text[:300]}"}
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        usage = data.get("usage") or {}
        try:
            answer = json.loads(message.get("content") or "")
        except json.JSONDecodeError:
            answer = None
        return {
            "answer": answer,
            "raw": None if answer is not None else (message.get("content") or "")[:2000],
            "model": data.get("model"),
            "provider": data.get("provider"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "cost": usage.get("cost"),
        }
    return {"error": "retries exhausted"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=FRESH / "reports.jsonl")
    parser.add_argument("--open-set", type=Path, default=FRESH / "open_set.jsonl")
    parser.add_argument("--level", choices=("names", "strict", "tradecraft"), default="strict")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--kb", type=Path, default=REPO_ROOT / "data" / "kb" / "attack_kb.json")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "audits" / "fresh_h2h")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cost", type=float, default=0.0, help="stop above this spend (USD)")
    parser.add_argument(
        "--price", type=float, nargs=2, default=(4.0, 20.0), help="USD per M tokens in/out"
    )
    parser.add_argument("--since", help="only reports published on or after this date (ISO)")
    parser.add_argument("--dry-run", action="store_true", help="count tokens and estimate cost")
    args = parser.parse_args(argv)
    kb = AttackKnowledgeBase.load(args.kb)
    rows = [
        {**row, "arm": arm}
        for arm, path in (("known", args.reports), ("open_set", args.open_set))
        if path.is_file()
        for row in report_texts(path, kb, args.level)
        if not args.since or (row["published"] or "") >= args.since
    ]
    prompts = {row["report_id"]: prompt_for(row["statements"]) for row in rows}
    input_tokens = sum(len(SYSTEM + text) / 3.6 for text in prompts.values())
    if args.dry_run:
        low, high = 1500 * len(rows), 5000 * len(rows)
        cost = [(input_tokens * args.price[0] + out * args.price[1]) / 1e6 for out in (low, high)]
        print(
            f"{len(rows)} reports ({sum(r['arm'] == 'known' for r in rows)} known, "
            f"{sum(r['arm'] == 'open_set' for r in rows)} open set), ~{input_tokens / 1e6:.2f} M "
            f"input tokens, estimated ${cost[0]:.2f}-${cost[1]:.2f} with 1.5k-5k output tokens "
            f"per report at ${args.price[0]}/${args.price[1]} per M"
        )
        return 0
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    if args.output is None:
        parser.error("--output is required for a run")
    cache = args.cache / "llm_cache" / args.model.replace("/", "_") / args.level
    cache.mkdir(parents=True, exist_ok=True)
    names = name_index(kb)
    spent = 0.0
    results = []
    for row in rows:
        path = cache / f"{row['report_id']}.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            if args.max_cost and spent >= args.max_cost:
                print(f"stopped at ${spent:.2f} (--max-cost {args.max_cost})", flush=True)
                break
            result = call(args.model, prompts[row["report_id"]], key, effort=args.effort)
            if "error" not in result:
                path.write_text(json.dumps(result, indent=1), encoding="utf-8")
            spent += float(result.get("cost") or 0.0)
            print(
                f"{row['report_id']} {result.get('error') or (result['answer'] or {}).get('decision')}"
                f" ${spent:.2f}",
                flush=True,
            )
        answer = result.get("answer") or {}
        mapped = [
            map_candidate(candidate, kb, names) for candidate in answer.get("candidates") or []
        ][:3]
        results.append(
            {
                "report_id": row["report_id"],
                "arm": row["arm"],
                "published": row["published"],
                "group_id": row["group_id"],
                "statements": len(row["statements"]),
                "candidates": mapped,
                "raw_candidates": answer.get("candidates"),
                "decision": answer.get("decision"),
                "rationale": answer.get("rationale"),
                "error": result.get("error"),
                "cost": result.get("cost"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "served_by": result.get("model"),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "effort": args.effort,
                "level": args.level,
                "since": args.since,
                "prompt": PROMPT,
                "system": SYSTEM,
                "reports": str(args.reports),
                "open_set": str(args.open_set),
                "cost": round(sum(float(r["cost"] or 0) for r in results), 4),
                "rows": results,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{len(results)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
