import json
from dataclasses import asdict, dataclass
from typing import Any

from apt_classifier.evidence.models import (
    TechniqueCandidate,
    TechniqueCandidateState,
    TechniqueVerification,
    TechniqueVerificationVerdict,
)
from apt_classifier.llm import LLMUsage, OpenRouterClient

CYBERRE_CLASSIFIER_NAME = "cyberre_llm"

FILTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevant": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["relevant", "reason"],
}

RANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "technique_id": {"type": "string"},
                    "rank": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["technique_id", "rank", "reason"],
            },
        }
    },
    "required": ["ranking"],
}

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "technique_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "unsupported", "uncertain"],
                    },
                    "support_span": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["technique_id", "verdict", "support_span", "reason"],
            },
        }
    },
    "required": ["decisions"],
}


@dataclass(frozen=True)
class CyberRETechnique:
    technique_id: str
    technique_name: str
    reason: str


@dataclass(frozen=True)
class CyberRERankedTechnique:
    technique_id: str
    technique_name: str
    rank: int
    reason: str


@dataclass(frozen=True)
class CyberREResult:
    observation_id: str
    observation_text: str
    filtered: tuple[CyberRETechnique, ...]
    ranking: tuple[CyberRERankedTechnique, ...]
    model: str
    filter_calls: int
    usage: LLMUsage

    @property
    def applicable(self) -> bool:
        return bool(self.ranking)

    def technique_scores(self) -> dict[str, float]:
        return {item.technique_id: 1.0 / item.rank for item in self.ranking if item.rank > 0}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usage"] = asdict(self.usage)
        return payload


@dataclass(frozen=True)
class TechniqueCandidateVerificationResult:
    candidates: tuple[TechniqueCandidate, ...]
    model: str
    scope: str
    usage: LLMUsage
    prompt: str
    response: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "model": self.model,
            "scope": self.scope,
            "usage": asdict(self.usage),
            "prompt": self.prompt,
            "response": self.response,
            "error": self.error,
        }


def verify_technique_candidates(
    client: OpenRouterClient,
    kb: Any,
    observation: Any,
    candidates: list[TechniqueCandidate] | tuple[TechniqueCandidate, ...],
    *,
    model: str,
    effort: str,
    scope: str = "ambiguous",
) -> TechniqueCandidateVerificationResult:
    if scope not in {"off", "ambiguous", "all"}:
        raise ValueError("unsupported verifier scope")
    supplied = tuple(candidates)
    # Hierarchy-derived parents are never direct observation evidence and stay NEEDS_REVIEW
    # whatever the verdict, so sending them only spends tokens.
    verifiable = tuple(
        candidate for candidate in supplied if candidate.evidence_for("hierarchy_parent") is None
    )
    selected = (
        tuple(
            candidate
            for candidate in verifiable
            if candidate.state is TechniqueCandidateState.NEEDS_REVIEW
        )
        if scope == "ambiguous"
        else verifiable
        if scope == "all"
        else ()
    )
    if not selected:
        return TechniqueCandidateVerificationResult(
            supplied,
            model,
            scope,
            LLMUsage(),
            "",
            "",
        )
    observation_text = str(getattr(observation, "text", "")).strip()
    provenance = getattr(observation, "provenance", None)
    source_excerpt = str(getattr(provenance, "raw_excerpt", "") or "").strip()
    candidate_payload = [
        {
            "technique_id": candidate.technique_id,
            "name": candidate.technique_name,
            "parent_id": candidate.parent_technique_id,
            "description": _description(getattr(kb, "techniques", {}).get(candidate.technique_id))[
                :1800
            ],
            "retrieval_evidence": [
                {
                    "source": evidence.source,
                    "field": evidence.field,
                    "text": evidence.text,
                }
                for evidence in candidate.evidence
            ],
        }
        for candidate in selected
    ]
    prompt = (
        f"NORMALIZED OBSERVATION:\n{observation_text}\n\n"
        f"SOURCE EXCERPT:\n{source_excerpt or observation_text}\n\n"
        f"CANDIDATE TECHNIQUES:\n{json.dumps(candidate_payload, sort_keys=True)}"
    )
    try:
        response = client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Verify each supplied MITRE ATT&CK candidate independently against the "
                        "normalized observation and source excerpt. Use only supplied evidence and "
                        "technique definitions. Multiple candidates may be supported. Mark supported "
                        "only for directly stated behavior and copy an exact supporting span. Mark "
                        "unsupported when contradicted or unrelated, and uncertain when detail is "
                        "insufficient. Do not add candidates, infer intent, or use actor identity."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            effort=effort,
            schema=VERIFY_SCHEMA,
            schema_name="technique_candidate_verification",
            max_tokens=max(1200, min(8000, 350 * len(selected))),
            seed=47 if model.startswith("openai/") else None,
        )
    except Exception as exc:
        updated = tuple(
            candidate.model_copy(
                update={
                    "verification": TechniqueVerification(
                        verdict=TechniqueVerificationVerdict.ERROR,
                        reason=str(exc)[:500],
                        model=model,
                    )
                }
            )
            if candidate in selected
            else candidate
            for candidate in supplied
        )
        return TechniqueCandidateVerificationResult(
            updated,
            model,
            scope,
            LLMUsage(),
            prompt,
            "",
            str(exc)[:500],
        )
    parsed = response.parsed if isinstance(response.parsed, dict) else {}
    allowed = {candidate.technique_id for candidate in selected}
    decisions: dict[str, dict[str, Any]] = {}
    for item in parsed.get("decisions", []):
        if not isinstance(item, dict):
            continue
        technique_id = str(item.get("technique_id") or "").upper()
        if technique_id in allowed and technique_id not in decisions:
            decisions[technique_id] = item
    source_corpus = f"{observation_text}\n{source_excerpt}".casefold()
    updated_candidates = []
    for candidate in supplied:
        if candidate not in selected:
            updated_candidates.append(candidate)
            continue
        decision = decisions.get(candidate.technique_id)
        if decision is None:
            verdict = TechniqueVerificationVerdict.UNCERTAIN
            reason = "Verifier omitted this candidate."
            support_span = None
        else:
            verdict = TechniqueVerificationVerdict(str(decision.get("verdict") or "uncertain"))
            reason = _compact(str(decision.get("reason") or "")) or None
            support_span = _compact(str(decision.get("support_span") or "")) or None
            if verdict is TechniqueVerificationVerdict.SUPPORTED and (
                not support_span or support_span.casefold() not in source_corpus
            ):
                verdict = TechniqueVerificationVerdict.UNCERTAIN
                reason = "Verifier support span was not present in the supplied evidence."
                support_span = None
        state = (
            TechniqueCandidateState.AUTO_ACCEPTED
            if verdict is TechniqueVerificationVerdict.SUPPORTED
            else TechniqueCandidateState.AUTO_REJECTED
            if verdict is TechniqueVerificationVerdict.UNSUPPORTED
            else TechniqueCandidateState.NEEDS_REVIEW
        )
        updated_candidates.append(
            candidate.model_copy(
                update={
                    "verification": TechniqueVerification(
                        verdict=verdict,
                        reason=reason,
                        support_span=support_span,
                        model=model,
                    ),
                    "state": state,
                    "state_reason": f"OpenRouter verifier: {verdict.value}",
                }
            )
        )
    return TechniqueCandidateVerificationResult(
        tuple(updated_candidates),
        model,
        scope,
        response.usage,
        prompt,
        response.content,
    )


def run_cyberre_scan(
    client: OpenRouterClient,
    kb: Any,
    observation: Any,
    *,
    model: str,
    effort: str,
    max_ranked_techniques: int = 50,
    prefilter_store: Any = None,
    prefilter_top_k: int = 50,
) -> CyberREResult:
    """Run the two-stage CyberRE-LLM technique filter and ranker.

    The filter evaluates techniques in the supplied ATT&CK knowledge base
    independently. The ranker then receives only techniques accepted by the
    filter. This is intentionally analyst-triggered because the filter is
    potentially hundreds of LLM calls for one observation.

    When ``prefilter_store`` is supplied, the technique space is narrowed to
    the top ``prefilter_top_k`` ATTACK-BERT vector hits for the observation
    text before LLM filtering. This preserves the CyberRE-LLM methodology
    (independent per-technique relevance decisions followed by ranking) while
    reducing the number of LLM calls from the full KB size to roughly
    ``prefilter_top_k``. Techniques outside the pre-filter are not evaluated
    and cannot be inferred.
    """
    text = str(getattr(observation, "text", "")).strip()
    observation_id = str(getattr(observation, "id", ""))
    techniques = getattr(kb, "techniques", {})
    candidate_ids = _select_candidate_techniques(
        techniques, text, prefilter_store=prefilter_store, prefilter_top_k=prefilter_top_k
    )
    filtered: list[CyberRETechnique] = []
    usage = LLMUsage()
    for technique_id in candidate_ids:
        technique = techniques[technique_id]
        name = str(_record_value(technique, "name", technique_id))
        description = _description(technique)
        response = client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the filtering stage of CyberRE-LLM. Decide only whether the "
                        "supplied incident sentence is directly related to the supplied MITRE "
                        "ATT&CK technique description. Use only the supplied text and description; "
                        "do not use outside MITRE knowledge, infer missing behavior, or infer intent."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"INCIDENT SENTENCE:\n{text}\n\n"
                        f"TECHNIQUE: {name} ({technique_id})\n"
                        f"DESCRIPTION:\n{description}\n\n"
                        "Return relevant=true only when the sentence directly supports this technique."
                    ),
                },
            ],
            model=model,
            effort=effort,
            schema=FILTER_SCHEMA,
            schema_name="cyberre_technique_filter",
            max_tokens=500,
            seed=41 if model.startswith("openai/") else None,
        )
        usage = _add_usage(usage, response.usage)
        parsed = response.parsed if isinstance(response.parsed, dict) else {}
        if bool(parsed.get("relevant")):
            filtered.append(
                CyberRETechnique(
                    technique_id=technique_id,
                    technique_name=name,
                    reason=_compact(str(parsed.get("reason") or "Directly related technique.")),
                )
            )

    ranking = _rank_filtered_techniques(
        client,
        kb,
        text,
        filtered,
        model=model,
        effort=effort,
        max_ranked_techniques=max_ranked_techniques,
    )
    if ranking[1] is not None:
        usage = _add_usage(usage, ranking[1].usage)
    return CyberREResult(
        observation_id=observation_id,
        observation_text=text,
        filtered=tuple(filtered),
        ranking=ranking[0],
        model=model,
        filter_calls=len(candidate_ids),
        usage=usage,
    )


def _select_candidate_techniques(
    techniques: dict[str, Any],
    text: str,
    *,
    prefilter_store: Any = None,
    prefilter_top_k: int = 50,
) -> list[str]:
    """Return the sorted list of technique IDs to evaluate for one observation.

    When ``prefilter_store`` is supplied and exposes ``query_techniques``,
    the candidate set is the union of the top ``prefilter_top_k`` vector hits
    for the observation text plus any parent techniques of those hits. When the
    store is not supplied, the full KB is used (original CyberRE-LLM behavior).
    """
    if prefilter_store is None or not callable(getattr(prefilter_store, "query_techniques", None)):
        return sorted(techniques)
    if not text.strip():
        return sorted(techniques)
    top_k = max(1, int(prefilter_top_k))
    hit_lists = prefilter_store.query_techniques([text], n_results=top_k)
    if not hit_lists:
        return sorted(techniques)
    candidate_ids: set[str] = set()
    for hit in hit_lists[0]:
        metadata = getattr(hit, "metadata", {})
        technique_id = str(metadata.get("technique_id") or "")
        if technique_id in techniques:
            candidate_ids.add(technique_id)
        parent_id = str(metadata.get("parent_id") or "")
        if parent_id and parent_id in techniques:
            candidate_ids.add(parent_id)
    if not candidate_ids:
        return sorted(techniques)
    return sorted(candidate_ids)


def _rank_filtered_techniques(
    client: OpenRouterClient,
    kb: Any,
    text: str,
    filtered: list[CyberRETechnique],
    *,
    model: str,
    effort: str,
    max_ranked_techniques: int,
) -> tuple[tuple[CyberRERankedTechnique, ...], Any]:
    if not filtered:
        return (), None
    limit = max(1, int(max_ranked_techniques))
    candidates = filtered[:limit]
    candidate_text = [
        {
            "technique_id": item.technique_id,
            "technique_name": item.technique_name,
            "description": _description(getattr(kb, "techniques", {}).get(item.technique_id)),
            "filter_reason": item.reason,
        }
        for item in candidates
    ]
    response = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are the ranking stage of CyberRE-LLM. Rank only the supplied ATT&CK "
                    "techniques for the incident sentence. Use only the supplied sentence and "
                    "technique descriptions. Do not add techniques, use outside MITRE knowledge, "
                    "or infer behavior that is not stated."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"INCIDENT SENTENCE:\n{text}\n\nCANDIDATE TECHNIQUES:\n"
                    f"{json.dumps(candidate_text, sort_keys=True)}"
                ),
            },
        ],
        model=model,
        effort=effort,
        schema=RANK_SCHEMA,
        schema_name="cyberre_technique_ranking",
        max_tokens=max(1000, min(8000, 250 * len(candidates))),
        seed=43 if model.startswith("openai/") else None,
    )
    parsed = response.parsed if isinstance(response.parsed, dict) else {}
    allowed = {item.technique_id: item for item in candidates}
    supplied: dict[str, dict[str, Any]] = {}
    raw_ranking = parsed.get("ranking")
    if isinstance(raw_ranking, list):
        for item in raw_ranking:
            if not isinstance(item, dict):
                continue
            technique_id = str(item.get("technique_id", ""))
            if technique_id in allowed and technique_id not in supplied:
                supplied[technique_id] = item
    ordered = sorted(
        supplied.items(),
        key=lambda pair: (int(pair[1].get("rank") or len(supplied) + 1), pair[0]),
    )
    ranked = tuple(
        CyberRERankedTechnique(
            technique_id=technique_id,
            technique_name=allowed[technique_id].technique_name,
            rank=index,
            reason=_compact(str(item.get("reason") or allowed[technique_id].reason)),
        )
        for index, (technique_id, item) in enumerate(ordered, 1)
    )
    return ranked, response


def _description(technique: Any) -> str:
    description = str(_record_value(technique, "description", ""))
    return description[:4000]


def _record_value(record: Any, name: str, default: Any = "") -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _compact(value: str) -> str:
    return " ".join(value.split())[:1000]


def _add_usage(first: LLMUsage, second: LLMUsage) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        completion_tokens=first.completion_tokens + second.completion_tokens,
        cost_usd=first.cost_usd + second.cost_usd,
    )
