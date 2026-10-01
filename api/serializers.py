"""AnalysisState -> JSON-serialisable dictionaries for the HTTP API.

Every function is a pure read of the state; nothing here mutates or calls the engine, so a
serialisation failure can never corrupt a case. Group ids are always accompanied by the ATT&CK
name so the front end never needs the knowledge base.
"""

from __future__ import annotations

from typing import Any

from apt_classifier.platforms import HOST_PLATFORMS, ledger_platforms
from apt_classifier.report import STAGE_INFO, group_name


def _name(kb: Any, group_id: str) -> str:
    group = getattr(kb, "groups", {}).get(group_id)
    return str(group.name) if group else group_id


def observations(state: Any) -> list[dict[str, Any]]:
    # operating systems each observation was mapped under, and where that came from
    diagnostics = getattr(getattr(state, "retrieval", None), "diagnostics", None) or {}
    scopes = diagnostics.get("platform_scopes") or {}
    rows = []
    for index, item in enumerate(state.ledger.observations, 1):
        provenance = getattr(item, "provenance", None)
        rows.append(
            {
                "id": item.id,
                "index": index,
                "text": item.text,
                "kind": item.kind,
                "status": item.status.value,
                "source": getattr(provenance, "source", "") if provenance else "",
                "source_name": getattr(provenance, "source_name", "") if provenance else "",
                "occurrences": int(getattr(item, "occurrence_count", 1) or 1),
                "lines": [int(n) for n in (getattr(item, "line_numbers", ()) or ())],
                "platform": scopes.get(item.id),
            }
        )
    return rows


def _evidence_items(candidate: Any) -> list[dict[str, Any]]:
    rows = []
    for item in candidate.evidence:
        rows.append(
            {
                "source": item.source,
                "field": item.field,
                "score": item.score,
                "rank": item.rank,
                "text": (item.text or "")[:240] or None,
                # exemplar evidence names the labelled corpus its sentence comes from, so the
                # UI can attribute it (tumeteor/mitre-ttp-mapping is CC-BY-4.0)
                "dataset": (getattr(item, "metadata", None) or {}).get("source_dataset"),
            }
        )
    return rows


def candidates(state: Any) -> list[dict[str, Any]]:
    order = {obs.id: index for index, obs in enumerate(state.ledger.observations, 1)}
    rows = []
    for candidate in getattr(state.retrieval, "candidates", ()) or ():
        verification = candidate.verification
        probability = next(
            (
                float(item.score)
                for item in candidate.evidence
                if item.source == "acceptance_model" and item.score is not None
            ),
            None,
        )
        rows.append(
            {
                "key": f"{candidate.observation_id}|{candidate.technique_id}",
                "observation_id": candidate.observation_id,
                "observation_index": order.get(candidate.observation_id, 0),
                "technique_id": candidate.technique_id,
                "technique_name": candidate.technique_name,
                "parent_technique_id": candidate.parent_technique_id,
                "is_subtechnique": bool(candidate.is_subtechnique),
                "state": candidate.state.value,
                "state_reason": candidate.state_reason or "",
                "rank": int(candidate.fused_rank),
                "fused_score": float(candidate.fused_score),
                "probability": probability,
                "evidence": _evidence_items(candidate),
                "verification": (
                    {"verdict": verification.verdict.value, "reason": verification.reason or ""}
                    if verification is not None and verification.verdict.value != "not_run"
                    else None
                ),
            }
        )
    return rows


def candidate_counts(state: Any) -> dict[str, int]:
    counts = {"auto_accepted": 0, "needs_review": 0, "auto_rejected": 0}
    for candidate in getattr(state.retrieval, "candidates", ()) or ():
        counts[candidate.state.value] = counts.get(candidate.state.value, 0) + 1
    return counts


def techniques(state: Any) -> list[dict[str, Any]]:
    accepted = getattr(state.retrieval, "observed_techniques", {}) or {}
    return [
        {
            "technique_id": m.technique_id,
            "technique_name": m.technique_name,
            "confidence": round(float(accepted.get(m.technique_id, m.score)), 3),
            "observation_ids": list(m.observation_ids),
            "matched_text": (m.matched_text or "")[:300],
        }
        for m in sorted(getattr(state.retrieval, "matches", ()) or (), key=lambda m: -m.score)
    ]


def decisions(state: Any) -> list[dict[str, Any]]:
    texts = {obs.id: obs.text for obs in state.ledger.observations}
    return [
        {
            "observation_id": d.observation_id,
            "observation": texts.get(d.observation_id, d.observation_id)[:160],
            "technique_id": d.technique_id,
            "action": d.action.value,
            "replacement": d.replacement_technique_id,
            "rationale": d.rationale or "",
        }
        for d in getattr(state.ledger, "technique_reviews", []) or []
    ]


def ranking(kb: Any, state: Any, limit: int = 20) -> list[dict[str, Any]]:
    discrimination = getattr(state.context, "discrimination", None)
    by_group = discrimination.by_group() if discrimination is not None else {}
    rows = []
    for entry in state.fusion.entries[:limit]:
        item = by_group.get(entry.group_id)
        rows.append(
            {
                "group_id": entry.group_id,
                "name": _name(kb, entry.group_id),
                "rank": int(entry.rank),
                "score": float(entry.score),
                "relative_score": float(entry.relative_score),
                "classifier_ranks": dict(entry.classifier_ranks),
                "rank_min": int(entry.rank_min),
                "rank_max": int(entry.rank_max),
                "rank_std": float(entry.rank_std),
                "contested": bool(entry.contested),
                "discrimination": item.to_dict() if item is not None else None,
            }
        )
    return rows


def decision(kb: Any, state: Any, thresholds: Any) -> dict[str, Any] | None:
    item = getattr(state, "attribution_decision", None)
    if item is None:
        return None
    info = STAGE_INFO.get(item.stage.value, {})
    decisive = getattr(state.context, "decisive_software", {}) or {}
    factor = float(decisive.get("factor", 1.0) or 1.0)
    software = getattr(kb, "software", {}) or {}
    return {
        "stage": item.stage.value,
        "label": info.get("label", item.stage.value),
        "summary": info.get("summary", ""),
        "advice": info.get("advice", ""),
        "named": item.stage.value == "known_group_hypothesis",
        "leading_group_id": item.leading_group_id,
        "leading_group_name": item.leading_group_name
        or (_name(kb, item.leading_group_id) if item.leading_group_id else ""),
        "leading_score": float(item.leading_score),
        "margin": float(item.margin),
        "maturity_score": float(item.maturity_score),
        "maturity_factors": {k: float(v) for k, v in (item.maturity_factors or {}).items()},
        "separation": float(getattr(item, "separation", 0.0) or 0.0),
        "separation_factors": dict(getattr(item, "separation_factors", {}) or {}),
        "runner_up_group_id": getattr(item, "runner_up_group_id", None),
        "runner_up_name": (
            _name(kb, item.runner_up_group_id)
            if getattr(item, "runner_up_group_id", None)
            else None
        ),
        "runner_up_related": bool(getattr(item, "runner_up_related", False)),
        "related_groups": [
            {"group_id": gid, "name": _name(kb, gid), "reason": reason}
            for gid, reason in (getattr(item, "related_leading_groups", {}) or {}).items()
        ],
        "rationale": item.rationale,
        "cluster_id": item.cluster_id,
        "gate_mode": str(getattr(item, "gate_mode", "maturity")),
        "calibrated_probability": (
            float(item.calibrated_probability)
            if getattr(item, "is_calibrated", False) and item.calibrated_probability is not None
            else None
        ),
        "calibration_note": str(getattr(item, "calibration_note", "") or ""),
        "thresholds": {
            "maturity": float(getattr(thresholds, "hypothesis_maturity_score", 0.0)),
            "margin": float(getattr(thresholds, "hypothesis_margin", 0.0)),
            "separation": float(getattr(thresholds, "hypothesis_separation", 0.0)),
            "min_discriminative": float(getattr(thresholds, "hypothesis_min_discriminative", 0.0)),
            "gate_mode": str(getattr(thresholds, "gate_mode", "maturity")),
        },
        "decisive_software": (
            {
                "factor": factor,
                "software": [
                    {"id": sid, "name": software[sid].name if sid in software else sid}
                    for sid in decisive.get("software_ids", [])
                ],
            }
            if factor > 1.0
            else None
        ),
    }


def origin(kb: Any, state: Any) -> dict[str, Any]:
    assessment = state.origin_assessment
    origin_item = assessment.origin
    threat = assessment.threat_type
    hypotheses = list(origin_item.hypotheses)
    top = getattr(origin_item, "top_country", None) or (
        hypotheses[0].country if hypotheses else None
    )
    return {
        "top_country": top,
        "confidence": origin_item.confidence,
        "coverage": float(origin_item.candidate_coverage),
        "caveats": list(origin_item.caveats),
        "hypotheses": [
            {
                "country": h.country,
                "relative_support": float(h.relative_support),
                "supporting": [
                    {"group_id": gid, "name": _name(kb, gid)} for gid in h.supporting_group_ids
                ],
            }
            for h in hypotheses
        ],
        "threat_type": {
            "label": threat.label,
            "confidence": threat.confidence,
            "likely_apt": threat.likely_apt,
            "apt_support": float(threat.apt_like_support),
            "criminal_support": float(threat.criminal_support),
            "unresolved_support": float(threat.unresolved_support),
            "rationale": list(threat.rationale),
        },
    }


def behaviour_matches(kb: Any, state: Any, limit: int = 40) -> list[dict[str, Any]]:
    texts = {o.id: o.text for o in state.ledger.observations}
    matches = getattr(state.retrieval, "specific_behavior_matches", ()) or ()
    return [
        {
            "group_id": m.group_id,
            "name": _name(kb, m.group_id),
            "technique_id": m.technique_id,
            "similarity": round(float(m.score), 3),
            "observed": texts.get(m.observation_id, m.observation_id)[:240],
            "documented": (m.matched_text or "")[:240],
        }
        for m in sorted(matches, key=lambda i: (-i.score, i.group_id))[:limit]
    ]


def classifier_matrix(kb: Any, results: Any) -> dict[str, Any]:
    applicable = [r for r in results if r.applicable]
    candidate_ids = {e.group_id for r in applicable for e in r.entries[:15]}
    rows = []
    for group_id in candidate_ids:
        ranks = {}
        for result in applicable:
            item = result.by_group().get(group_id)
            ranks[result.classifier] = int(item.rank) if item else None
        rows.append({"group_id": group_id, "name": _name(kb, group_id), "ranks": ranks})
    names = [r.classifier for r in applicable]
    rows.sort(key=lambda row: tuple(row["ranks"].get(n) or 10**6 for n in names))
    return {"classifiers": names, "rows": rows}


def strategic_matches(kb: Any, state: Any) -> list[dict[str, Any]]:
    result = next(
        (
            r
            for r in state.classifier_results
            if r.classifier == "strategic_context" and r.applicable
        ),
        None,
    )
    if result is None:
        return []
    return [
        {
            "rank": int(e.rank),
            "group_id": e.group_id,
            "name": _name(kb, e.group_id),
            "coverage": round(float(e.score), 3),
            "factors": list(e.signal_ids),
        }
        for e in result.entries[:20]
    ]


def review(kb: Any, state: Any) -> dict[str, Any] | None:
    outcome = getattr(state, "review", None)
    if outcome is None:
        return None
    final = outcome.final.parsed or {}
    baseline = {e.group_id: e.rank for e in state.fusion.entries}
    bias = outcome.fame_bias
    advocate = outcome.advocate.parsed or {}
    return {
        "headline": final.get("headline", "Independent AI review"),
        "confidence_label": str(final.get("confidence_label", "low")),
        "assessment": final.get("assessment", ""),
        "advisory_ranking": [
            {
                "rank": item.get("rank"),
                "group_id": str(item.get("candidate_id", "")),
                "name": _name(kb, str(item.get("candidate_id", ""))),
                "deterministic_rank": baseline.get(str(item.get("candidate_id", ""))),
                "justification": item.get("justification", ""),
            }
            for item in final.get("advisory_ranking", []) or []
        ],
        "limitations": list(final.get("limitations", []) or []),
        "disconfirmation_tests": list(final.get("disconfirmation_tests", []) or []),
        "fame_bias": {
            "kendall_tau": float(bias.kendall_tau),
            "prominence_correlation": float(bias.prominence_correlation),
            "largest_promotions": [
                _name(kb, gid) for gid in list(bias.largest_promotions or ())[:5]
            ],
        },
        "advocate": {
            "counter_case": advocate.get("counter_case", ""),
            "missing_evidence": list(advocate.get("missing_evidence", []) or []),
        },
        "audit_path": outcome.audit_path,
    }


def cyberre(state: Any) -> dict[str, Any]:
    results = getattr(state, "cyberre_results", {}) or {}
    rows = {}
    for oid, result in results.items():
        usage = result.usage
        rows[oid] = {
            "observation_id": oid,
            "observation_text": result.observation_text,
            "model": result.model,
            "filter_calls": int(result.filter_calls),
            "kept": len(result.filtered),
            "applicable": bool(getattr(result, "applicable", True)),
            "tokens": int(usage.prompt_tokens + usage.completion_tokens),
            "cost_usd": round(float(usage.cost_usd), 4),
            "ranking": [
                {
                    "rank": int(i.rank),
                    "technique_id": i.technique_id,
                    "technique_name": i.technique_name,
                    "reason": i.reason,
                }
                for i in result.ranking
            ],
            "filtered": [
                {
                    "technique_id": i.technique_id,
                    "technique_name": i.technique_name,
                    "reason": i.reason,
                }
                for i in result.filtered
            ],
        }
    return {"results": rows}


def intel(state: Any) -> dict[str, Any]:
    snapshot = getattr(state, "threat_intel_snapshot", None)
    matches = getattr(state, "threat_intel_matches", ()) or ()
    if snapshot is None:
        return {"snapshot": None, "matches": []}
    items = snapshot.item_by_id()
    return {
        "snapshot": snapshot_summary(snapshot),
        "matches": [
            {
                "item_id": m.item_id,
                "score": round(float(m.score), 2),
                "reasons": list(m.reasons),
                "verification": m.verification,
                "group_ids": list(m.group_ids),
                "published_at": items[m.item_id].published_at[:10],
                "source": items[m.item_id].source,
                "title": items[m.item_id].title,
                "url": items[m.item_id].url,
            }
            for m in matches
            if m.item_id in items
        ],
    }


def snapshot_summary(snapshot: Any) -> dict[str, Any]:
    return {
        "fetched_at": snapshot.fetched_at,
        "records": len(snapshot.items),
        "authoritative": sum(i.verification == "authoritative" for i in snapshot.items),
        "claims": sum(i.verification == "unverified_extortion_claim" for i in snapshot.items),
        "sources": [
            {"source": s.source, "ok": bool(s.ok), "error": getattr(s, "error", None)}
            for s in snapshot.sources
        ],
    }


def snapshot_items(snapshot: Any, match_scores: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "published_at": item.published_at[:10],
            "source": item.source,
            "kind": item.kind,
            "title": item.title,
            "summary": item.summary[:400],
            "actors": list(item.actors),
            "cves": list(item.cves),
            "products": list(getattr(item, "products", ()) or ()),
            "industries": list(item.industries),
            "verification": item.verification,
            "match_score": match_scores.get(item.id),
            "url": item.url,
        }
        for item in snapshot.items
    ]


def graph(model: Any) -> dict[str, Any]:
    """Canvas or Diamond model as plain nodes and edges."""
    nodes = []
    for node in model.nodes:
        row = {
            "id": node.id,
            "label": node.label,
            "score": float(node.score),
            "details": node.details,
        }
        if hasattr(node, "kind"):
            row["kind"] = node.kind
        if hasattr(node, "vertex"):
            row["vertex"] = node.vertex
            row["status"] = node.status
        nodes.append(row)
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "relationship": e.relationship,
            "weight": float(e.weight),
            "details": e.details,
        }
        for e in model.edges
    ]
    return {"nodes": nodes, "edges": edges, "caveats": list(getattr(model, "caveats", ()) or ())}


def next_step(state: Any) -> dict[str, str]:
    counts = candidate_counts(state)
    if not state.fusion.entries:
        return {
            "message": "No candidate group has documented overlap yet. Add more concrete evidence.",
            "page": "intake",
        }
    reviewed = len(getattr(state.ledger, "technique_reviews", []) or [])
    if counts["needs_review"] and not reviewed:
        return {
            "message": f"{counts['needs_review']} technique candidates need a decision so only "
            "trusted mappings vote.",
            "page": "techniques",
        }
    if not state.ledger.question_answers and not state.interview_complete:
        return {
            "message": "Answer the interview leads to separate the shortlisted groups.",
            "page": "interview",
        }
    return {
        "message": "Read the assessment, then export the report or run the independent review.",
        "page": "assessment",
    }


def case_payload(
    kb: Any,
    state: Any,
    meta: Any,
    thresholds: Any,
    summary: dict[str, Any],
    top3: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = state.ledger.strategic_context
    return {
        "case_id": meta.case_id,
        "name": meta.name,
        "created_at": meta.created_at,
        "updated_at": meta.updated_at,
        "owner": dict(meta.owner) if getattr(meta, "owner", None) else None,
        "ingestion": dict(meta.ingestion or {}),
        "summary": summary,
        "next_step": next_step(state),
        "observations": observations(state),
        "target_platforms": ledger_platforms(state.ledger),
        "platform_options": list(HOST_PLATFORMS),
        "strategic_context": {
            name: values for name, values in context.field_values().items() if values
        }
        if not context.is_empty
        else {},
        "candidate_counts": candidate_counts(state),
        "candidates": candidates(state),
        "decisions": decisions(state),
        "techniques": techniques(state),
        "ranking": ranking(kb, state),
        "decision": decision(kb, state, thresholds),
        "top3": top3,
        "origin": origin(kb, state),
        "behaviour_matches": behaviour_matches(kb, state),
        "strategic_matches": strategic_matches(kb, state),
        "classifier_matrix": classifier_matrix(kb, state.classifier_results),
        "effective_weights": {
            k: round(float(v), 4) for k, v in state.fusion.effective_weights.items()
        },
        "weights": dict(state.weights or {}),
        "eligibility": [
            {"group_id": gid, "name": _name(kb, gid), "via": list(reasons)}
            for gid, reasons in sorted(
                (getattr(state.context, "group_eligibility_reasons", {}) or {}).items()
            )
        ],
        "retrieval_diagnostics": dict(getattr(state.retrieval, "diagnostics", {}) or {}),
        "interview": {
            "complete": bool(state.interview_complete),
            "stop_reason": state.interview_stop_reason,
            "answered": len(state.ledger.question_answers),
            "has_question": state.current_question is not None,
        },
        "review": review(kb, state),
        "cyberre": cyberre(state),
        "intel": intel(state),
        "group_label": group_name(kb, state.fusion.entries[0].group_id)
        if state.fusion.entries
        else None,
    }
