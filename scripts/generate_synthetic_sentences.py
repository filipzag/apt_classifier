"""Generate synthetic observation sentences per ATT&CK technique (SynthCTI-style).

For each technique the generator is grounded *only* in the knowledge base: the technique's
name, tactics, description and a few anonymized procedure examples, plus the names of its
sibling sub-techniques to steer away from. It asks the LLM for a small set of varied,
realistic observation sentences (analyst prose, alert-style one-liners, report language)
and filters them (length, no technique IDs, no actor names, not identical to indexed text,
deduplicated). Benchmark text is never used, so indexing the output cannot leak evaluation
data. Output: ``data/kb/synthetic_sentences.json``; the run is resumable and records cost.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings
from apt_classifier.llm import OpenRouterClient
from apt_classifier.vectorstore import (
    _actor_name_pattern,
    _procedure_sentences,
    normalize_procedure_text,
    procedure_sentence_texts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "kb" / "synthetic_sentences.json"
PROMPT_VERSION = 1
_TECHNIQUE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sentences"],
}

SYSTEM_PROMPT = (
    "You write realistic incident observations that a security analyst, EDR alert or "
    "threat report might contain, for use as retrieval examples in a MITRE ATT&CK technique "
    "index. Each sentence must describe the given technique specifically (not its sibling "
    "sub-techniques or parent in general), be self-contained, and contain no attacker or "
    "group names, no technique IDs and no citations. Vary register: some plain analyst "
    "prose, some alert-style one-liners with plausible process names, command fragments, "
    "registry paths or ports, some formal report language. 8 to 40 words each. Return JSON."
)


def _technique_context(
    kb: AttackKnowledgeBase, technique_id: str, procedures: dict[str, list[str]]
) -> dict[str, Any]:
    technique = kb.techniques[technique_id]
    siblings = [
        other.name
        for other_id, other in kb.techniques.items()
        if technique.parent_id
        and other.parent_id == technique.parent_id
        and other_id != technique_id
    ]
    if not technique.is_subtechnique:
        siblings = [
            other.name
            for other_id, other in kb.techniques.items()
            if other.parent_id == technique_id
        ][:12]
    description = (
        " ".join(technique.sentences.get("description", [])[:8])
        if technique.sentences
        else technique.description
    )
    return {
        "technique_id": technique_id,
        "name": technique.name,
        "parent": kb.techniques[technique.parent_id].name
        if technique.parent_id in kb.techniques
        else None,
        "tactics": list(technique.tactics),
        "platforms": list(technique.platforms)[:6],
        "description": description[:1400],
        "examples": procedures.get(technique_id, [])[:5],
        "do_not_describe": siblings[:12],
    }


def _filter(
    sentences: list[Any],
    *,
    indexed: frozenset[str],
    actor_pattern: re.Pattern[str] | None,
    seen: set[str],
    min_words: int,
    max_words: int,
) -> list[str]:
    kept: list[str] = []
    for item in sentences:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text or _TECHNIQUE_ID_RE.search(text):
            continue
        if actor_pattern is not None and actor_pattern.search(text):
            continue
        words = len(text.split())
        if words < min_words or words > max_words:
            continue
        key = normalize_procedure_text(text)
        if not key or key in indexed or key in seen:
            continue
        seen.add(key)
        kept.append(text)
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-technique", type=int, default=6)
    parser.add_argument(
        "--max-procedures",
        type=int,
        default=12,
        help="only techniques with fewer indexed procedure sentences than this",
    )
    parser.add_argument("--limit", type=int, help="stop after this many techniques (pilot)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=None, help="defaults to EXTRACT_MODEL")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true", help="list target techniques only")
    args = parser.parse_args(argv)

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    procedures = _procedure_sentences(kb)
    indexed = procedure_sentence_texts(kb) | frozenset(
        normalize_procedure_text(sentence)
        for technique in kb.techniques.values()
        for sentence in (technique.sentences or {}).get("description", [])
    )
    names = tuple(
        sorted(
            {
                n.strip()
                for g in kb.groups.values()
                for n in (g.name, *g.aliases)
                if n and n.strip()
            },
            key=lambda item: (-len(item), item.casefold()),
        )
    )
    actor_pattern = _actor_name_pattern(names)
    existing: dict[str, Any] = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    sentences: dict[str, list[str]] = dict(existing.get("sentences", {}))
    targets = [
        technique_id
        for technique_id in sorted(kb.techniques)
        if len(procedures.get(technique_id, [])) < args.max_procedures
        and technique_id not in sentences
    ]
    if args.limit is not None:
        targets = targets[: args.limit]
    print(f"targets: {len(targets)} techniques (already done: {len(sentences)})")
    if args.dry_run or not targets:
        return 0

    model = args.model or settings.extract_model
    local = threading.local()
    lock = threading.Lock()
    usage = {"cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "errors": 0}
    error_messages: dict[str, int] = {}

    def flush() -> dict[str, Any]:
        payload = {
            "metadata": {
                **existing.get("metadata", {}),
                "generated_at": datetime.now(UTC).isoformat(),
                "model": model,
                "effort": args.effort,
                "prompt_version": PROMPT_VERSION,
                "attack_version": kb.version,
                "per_technique": args.per_technique,
                "max_procedures": args.max_procedures,
                "technique_count": len(sentences),
                "sentence_count": sum(len(v) for v in sentences.values()),
                "usage": {
                    key: existing.get("metadata", {}).get("usage", {}).get(key, 0) + value
                    for key, value in usage.items()
                },
                "last_run_errors": dict(sorted(error_messages.items(), key=lambda kv: -kv[1])[:5]),
            },
            "sentences": dict(sorted(sentences.items())),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        return payload

    def client() -> OpenRouterClient:
        if not hasattr(local, "client"):
            local.client = OpenRouterClient(settings)
        return local.client

    def generate(technique_id: str) -> tuple[str, list[str], Any]:
        context = _technique_context(kb, technique_id, procedures)
        prompt = (
            f"Generate {args.per_technique} distinct observation sentences for this ATT&CK "
            "technique. Cover different sub-behaviours and telemetry angles mentioned in the "
            "description. Do not describe the techniques listed under do_not_describe.\n\n"
            + json.dumps(context, ensure_ascii=False)
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = client().complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    effort=args.effort,
                    schema=SCHEMA,
                    schema_name="synthetic_sentences",
                    max_tokens=1200,
                    seed=11 if model.startswith("openai/") else None,
                )
                parsed = response.parsed if isinstance(response.parsed, dict) else {}
                return technique_id, list(parsed.get("sentences", [])), response.usage
            except Exception as error:  # noqa: BLE001 - retried with backoff; summarized at exit
                last_error = error
                if "402" in str(error):
                    break  # out of credit: retrying cannot help
                time.sleep(2.0 * (2**attempt))
        assert last_error is not None
        raise last_error

    started = time.time()
    seen_global: set[str] = {
        normalize_procedure_text(t) for items in sentences.values() for t in items
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, technique_id): technique_id for technique_id in targets}
        done = 0
        for future in as_completed(futures):
            technique_id = futures[future]
            try:
                _, raw, call_usage = future.result()
            except Exception as error:  # noqa: BLE001 - one failed technique must not stop the run
                with lock:
                    usage["errors"] += 1
                    message = f"{type(error).__name__}: {str(error)[:160]}"
                    error_messages[message] = error_messages.get(message, 0) + 1
                continue
            with lock:
                kept = _filter(
                    raw,
                    indexed=indexed,
                    actor_pattern=actor_pattern,
                    seen=seen_global,
                    min_words=args.min_words,
                    max_words=args.max_words,
                )
                sentences[technique_id] = kept
                usage["cost_usd"] += float(getattr(call_usage, "cost_usd", 0.0) or 0.0)
                usage["prompt_tokens"] += int(getattr(call_usage, "prompt_tokens", 0) or 0)
                usage["completion_tokens"] += int(getattr(call_usage, "completion_tokens", 0) or 0)
                usage["calls"] += 1
                done += 1
                if done % 25 == 0 or done == len(targets):
                    payload = flush()
                    print(
                        f"  {done}/{len(targets)} techniques, "
                        f"{payload['metadata']['sentence_count']} sentences, "
                        f"${usage['cost_usd']:.3f} this run, {time.time() - started:.0f}s"
                    )
    payload = flush()
    for message, count in payload["metadata"]["last_run_errors"].items():
        print(f"  error x{count}: {message}", file=sys.stderr)
    print(
        f"done: {usage['calls']} calls, {usage['errors']} errors, ${usage['cost_usd']:.3f}; wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
