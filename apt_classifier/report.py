"""Headless report and export builders shared by the HTTP API and the Streamlit UI.

Everything here reads an ``AnalysisState`` and the knowledge base only, so a report can be
produced from any front end or from a script without importing a UI framework.
"""

from __future__ import annotations

import json
from typing import Any

from apt_classifier.attack_flow import build_attack_flow
from apt_classifier.canvas import build_analyst_canvas
from apt_classifier.navigator import export_layers
from apt_classifier.platforms import ledger_platforms

# Stage vocabulary shown to analysts. Icons carry the meaning together with the label so the
# state is never expressed by colour alone.
STAGE_INFO: dict[str, dict[str, str]] = {
    "insufficient_evidence": {
        "label": "Insufficient evidence",
        "icon": "⬜",
        "summary": "Too little technique evidence to rank groups meaningfully.",
        "advice": "Add concrete behaviour (commands, persistence, lateral movement) or logs.",
    },
    "unresolved_activity_cluster": {
        "label": "Unresolved activity cluster",
        "icon": "🔶",
        "summary": "Techniques were mapped but no documented group stands out.",
        "advice": "Treat this as an unattributed cluster; the shortlist is a starting point only.",
    },
    "known_group_shortlist": {
        "label": "Known-group shortlist",
        "icon": "🔷",
        "summary": "Several documented groups overlap the evidence; none clears the naming bar.",
        "advice": "Answer interview questions or add discriminating evidence to separate them.",
    },
    "known_group_hypothesis": {
        "label": "Named group hypothesis",
        "icon": "🟢",
        "summary": "One documented group clears the evidence bar.",
        "advice": "Check the documented-overlap note and run the independent review before reporting.",
    },
}

SHORT_STAGE = {
    "insufficient_evidence": "Insufficient",
    "unresolved_activity_cluster": "Cluster",
    "known_group_shortlist": "Shortlist",
    "known_group_hypothesis": "Named",
}


def group_name(kb: Any, group_id: str) -> str:
    group = getattr(kb, "groups", {}).get(group_id)
    return f"{group.name} ({group_id})" if group else group_id


def software_name(kb: Any, software_id: str) -> str:
    item = getattr(kb, "software", {}).get(software_id)
    return f"{item.name} ({software_id})" if item else software_id


def assessment_markdown(kb: Any, state: Any, *, case_name: str = "") -> str:
    """Analyst report. Works with or without the AI review; the deterministic result leads."""
    decision = getattr(state, "attribution_decision", None)
    title = "# Vestige attribution assessment" + (f": {case_name}" if case_name else "")
    lines = [title, ""]
    if decision is not None:
        info = STAGE_INFO.get(decision.stage.value, {})
        lines += [
            f"**Attribution stage:** {info.get('label', decision.stage.value)}",
            f"**Maturity:** {decision.maturity_score:.2f} · **Margin:** {decision.margin:.2f} · "
            f"**Separation:** {float(getattr(decision, 'separation', 0.0) or 0.0):.2f}",
        ]
        runner = getattr(decision, "runner_up_group_id", None)
        if runner:
            lines.append(
                f"**Best-supported alternative:** {group_name(kb, runner)}"
                + (" (documented-related)" if getattr(decision, "runner_up_related", False) else "")
            )
        if decision.leading_group_name:
            role = (
                "Leading hypothesis"
                if decision.stage.value == "known_group_hypothesis"
                else "Closest documented group (not an attribution)"
            )
            lines.append(f"**{role}:** {decision.leading_group_name} ({decision.leading_group_id})")
        related = getattr(decision, "related_leading_groups", {}) or {}
        if related:
            lines.append(
                "**Documented overlap:** "
                + "; ".join(f"{group_name(kb, gid)}: {reason}" for gid, reason in related.items())
            )
        decisive = getattr(state.context, "decisive_software", {}) or {}
        if float(decisive.get("factor", 1.0) or 1.0) > 1.0:
            lines.append(
                "**Decisive tooling:** named malware documented for very few groups ("
                + ", ".join(software_name(kb, sid) for sid in decisive.get("software_ids", []))
                + f"); software vote weighted ×{float(decisive['factor']):.1f}."
            )
        lines += ["", decision.rationale, ""]
    lines += [
        "## Deterministic ranking (authoritative)",
        "Family-aware reciprocal-rank fusion over the documented ATT&CK overlap. The relative "
        "index compares candidates within this case; it is not a probability.",
    ]
    for entry in state.fusion.entries[:15]:
        lines.append(
            f"{entry.rank}. **{group_name(kb, entry.group_id)}** — relative index "
            f"{entry.relative_score:.1f}{' (contested)' if entry.contested else ''}"
        )
    accepted = getattr(state.retrieval, "observed_techniques", {}) or {}
    if accepted:
        lines += ["", "## Accepted ATT&CK techniques"]
        target = ledger_platforms(state.ledger)
        if target:
            lines += [
                f"Target OS: {', '.join(target)}. Techniques documented only for other operating "
                "systems were not considered.",
                "",
            ]
        names = {m.technique_id: m.technique_name for m in getattr(state.retrieval, "matches", ())}
        lines += [f"- {tid} {names.get(tid, '') or ''}".rstrip() for tid in sorted(accepted)]
    answers = list(getattr(state.ledger, "question_answers", []) or [])
    if answers:
        lines += ["", "## Interview answers"]
        lines += [
            f"- {answer.question} — **{answer.status.value}**"
            + (f" ({answer.answer_text})" if answer.answer_text else "")
            for answer in answers
        ]
    origin = state.origin_assessment.origin
    threat_type = state.origin_assessment.threat_type
    lines += [
        "",
        "## Origin and intrusion-type hypotheses (from ATT&CK reporting on the ranked groups)",
        f"- Most supported origin: {origin.top_country or 'Unresolved'} ({origin.confidence})",
        f"- Intrusion type: {threat_type.label} ({threat_type.confidence})",
    ]
    review = getattr(state, "review", None)
    if review is not None:
        final = review.final.parsed
        lines += [
            "",
            f"## Independent AI review: {final.get('headline', '')}".rstrip(": "),
            f"Confidence: {final.get('confidence_label', 'low')}",
            "",
            str(final.get("assessment", "")),
        ]
        advisory = final.get("advisory_ranking", [])
        if advisory:
            lines += ["", "Advisory ordering (not applied to the deterministic ranking):"]
            lines += [
                f"{item.get('rank')}. {group_name(kb, str(item.get('candidate_id', '')))} — "
                f"{item.get('justification', '')}"
                for item in advisory
            ]
        if final.get("limitations"):
            lines += ["", "Limitations:"] + [f"- {item}" for item in final["limitations"]]
        if final.get("disconfirmation_tests"):
            lines += ["", "Disconfirmation tests:"] + [
                f"- {item}" for item in final["disconfirmation_tests"]
            ]
    snapshot = getattr(state, "threat_intel_snapshot", None)
    matches = getattr(state, "threat_intel_matches", ()) or ()
    if snapshot is not None and matches:
        items = snapshot.item_by_id()
        lines += ["", "## Non-voting live threat-intelligence matches"]
        lines += [
            f"- [{items[m.item_id].source}] {items[m.item_id].title} ({m.verification}): "
            f"{'; '.join(m.reasons)}"
            for m in matches[:10]
            if m.item_id in items
        ]
    lines += [
        "",
        "## Caveats",
        "- Ranking measures overlap with documented ATT&CK tradecraft; shared tooling, reporting "
        "gaps and false flags can invalidate attribution.",
        "- Interview leads are predictions, not findings.",
    ]
    return "\n".join(lines)


def navigator_layers(kb: Any, state: Any) -> Any:
    return export_layers(
        kb,
        observed_techniques=dict(getattr(state.retrieval, "observed_techniques", {})),
        negative_techniques=getattr(state.context, "negative_techniques", set()),
        candidate_technique_ids=[m.technique_id for m in getattr(state.retrieval, "matches", ())],
        profile=getattr(state, "telemetry_profile", None),
    )


def audit_payload(kb: Any, state: Any, *, case_name: str = "") -> dict[str, Any]:
    """Complete machine-readable record of the case (ledger, ranking, decision, graphs)."""
    decision = getattr(state, "attribution_decision", None)
    snapshot = getattr(state, "threat_intel_snapshot", None)
    review = getattr(state, "review", None)
    return {
        "case_id": state.ledger.case_id,
        "case_name": case_name,
        "ledger": state.ledger.to_dict(),
        "fusion": state.fusion.to_dict(),
        "attribution_decision": decision.to_dict() if decision is not None else None,
        "origin_and_threat_type": state.origin_assessment.to_dict(),
        "cyberre_results": {
            k: v.to_dict() for k, v in (getattr(state, "cyberre_results", {}) or {}).items()
        },
        "analyst_canvas": build_analyst_canvas(kb, state, top_groups=10, max_evidence=20).to_dict(),
        "navigator_layers": navigator_layers(kb, state),
        "attack_flow": build_attack_flow(kb, state, case_id=state.ledger.case_id).to_dict(),
        "live_threat_intel": {
            "snapshot": snapshot.to_dict() if snapshot is not None else None,
            "matches": [m.to_dict() for m in getattr(state, "threat_intel_matches", ()) or ()],
            "voting": False,
        },
        "review": review.to_dict() if review is not None else None,
    }


EXPORT_KINDS = ("markdown", "json", "navigator", "attack-flow")


def export_document(kb: Any, state: Any, kind: str, *, case_name: str = "") -> tuple[str, str, str]:
    """(content, media type, file name) for one export kind."""
    stem = f"vestige-{case_name or state.ledger.case_id}".replace(" ", "-")
    if kind == "markdown":
        return assessment_markdown(kb, state, case_name=case_name), "text/markdown", f"{stem}.md"
    if kind == "json":
        payload = audit_payload(kb, state, case_name=case_name)
        return json.dumps(payload, indent=2, default=str), "application/json", f"{stem}.json"
    if kind == "navigator":
        layers = navigator_layers(kb, state)
        return (
            json.dumps(layers, indent=2, default=str),
            "application/json",
            f"{stem}-navigator.json",
        )
    if kind == "attack-flow":
        flow = build_attack_flow(kb, state, case_id=state.ledger.case_id)
        return (
            json.dumps(flow.to_stix(), indent=2, default=str),
            "application/json",
            f"{stem}-attack-flow.json",
        )
    raise ValueError(f"unknown export kind {kind!r}; expected one of {', '.join(EXPORT_KINDS)}")
