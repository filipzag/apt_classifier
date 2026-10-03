"""Per-sentence prose standardisation with the local LLM (docs 16.4a).

A local model rewriting a whole report drops actions (CTI-HAL technique F1 0.431 -> 0.370 on
14 reports), so prose is split into sentences first and the model only rewrites each sentence
1:1 into one standardised attack-action sentence (thinking off). Coverage is decided by the
split, never by the model: an omitted sentence, a rewrite under three words ("None.") and a
failed call all keep the split sentence. Measured over all 55 CTI-HAL reports, technique F1
0.459 -> 0.472 against the plain split (22 reports better, 20 worse).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from apt_classifier.evidence.local_extract import MIN_WORDS
from apt_classifier.evidence.models import Observation

# Sentences per call; the measured reports had at most 52, so they ran as one call each.
CHUNK_SIZE = 60

STANDARDISE_SYSTEM = (
    "Rewrite each numbered sentence from a threat report into ONE standardised attack-action "
    "sentence: who or what acted (actor, malware, process), the action, and its object or "
    "target. Keep tool and malware names, commands, file paths, registry keys, hosts, "
    "accounts and protocols verbatim. Add no facts, intent, attribution or ATT&CK technique "
    "names. If a sentence describes no attacker action, return it unchanged. Return exactly "
    "one sentence for every number."
)
STANDARDISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sentences"],
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "sentence"],
                "properties": {"index": {"type": "integer"}, "sentence": {"type": "string"}},
            },
        }
    },
}


def standardise_observations(
    observations: Sequence[Observation],
    client: Any,
    *,
    model: str,
    diagnostics: dict[str, Any] | None = None,
) -> list[Observation]:
    """Rewrite each split sentence with ``model``; every observation is kept, one for one."""
    observations = list(observations)
    stats = {"rewritten": 0, "kept": 0, "failed_calls": 0}
    output: list[Observation] = []
    for start in range(0, len(observations), CHUNK_SIZE):
        chunk = observations[start : start + CHUNK_SIZE]
        rewritten: dict[int, str] = {}
        try:
            rewritten = _rewrite(chunk, client, model)
        except Exception:  # noqa: BLE001 - a failed call keeps the split sentences
            stats["failed_calls"] += 1
        for index, observation in enumerate(chunk, 1):
            sentence = rewritten.get(index, "")
            if len(sentence.split()) < MIN_WORDS:
                stats["kept"] += 1
                output.append(observation)
                continue
            stats["rewritten"] += 1
            provenance = observation.provenance
            metadata = dict(getattr(provenance, "metadata", None) or {})
            metadata.update(
                normalization="local_llm_sentence",
                original_sentence=observation.text,
                model=model,
            )
            output.append(
                observation.model_copy(
                    update={
                        "text": sentence,
                        "provenance": provenance.model_copy(update={"metadata": metadata}),
                    }
                )
            )
    if diagnostics is not None:
        diagnostics["standardisation"] = {"model": model, **stats}
    return output


def _rewrite(chunk: Sequence[Observation], client: Any, model: str) -> dict[int, str]:
    numbered = "\n".join(f"{index}. {item.text}" for index, item in enumerate(chunk, 1))
    response = client.complete(
        [
            {"role": "system", "content": STANDARDISE_SYSTEM},
            {"role": "user", "content": numbered},
        ],
        model=model,
        effort="low",
        schema=STANDARDISE_SCHEMA,
        schema_name="standardised_sentences",
        max_tokens=max(1200, 90 * len(chunk)),
        seed=0,
    )
    parsed = response.parsed if isinstance(response.parsed, dict) else {}
    rewritten: dict[int, str] = {}
    for item in parsed.get("sentences", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        sentence = " ".join(str(item.get("sentence") or "").split())
        if 1 <= index <= len(chunk) and sentence and index not in rewritten:
            rewritten[index] = sentence
    return rewritten
