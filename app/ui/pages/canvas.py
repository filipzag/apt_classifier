"""Analyst canvas: the evidence-to-group relationship map with a candidate inspector."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import evidence_table, group_name, status_strip
from apt_classifier.canvas import build_analyst_canvas, build_candidate_dossier
from apt_classifier.config import Settings
from apt_classifier.session import AttributionEngine

KIND_LABELS = {
    "evidence": "Evidence",
    "context": "Strategic context",
    "technique": "ATT&CK techniques",
    "tool": "Tools/software",
    "campaign": "Campaigns",
    "live_intel": "Live threat intelligence",
    "group": "Candidate groups",
    "country": "Country hypotheses",
    "assessment": "Threat type",
}
COLUMN_BY_KIND = {
    "evidence": 0.0,
    "context": 0.0,
    "technique": 1.0,
    "tool": 1.0,
    "campaign": 1.0,
    "live_intel": 1.0,
    "group": 2.0,
    "country": 3.0,
    "assessment": 3.0,
}
COLOR_BY_KIND = {
    "evidence": "#4C78A8",
    "context": "#72B7B2",
    "technique": "#F2CF5B",
    "tool": "#F58518",
    "campaign": "#B279A2",
    "live_intel": "#FF9DA6",
    "group": "#E45756",
    "country": "#54A24B",
    "assessment": "#9D755D",
}


def canvas_label(value: str, limit: int = 36) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def layered_spec(
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    *,
    color_field: str,
    color_domain: list[str],
    color_range: list[str],
    color_title: str,
    x_domain: list[float],
    height: int,
) -> dict[str, Any]:
    axis = {"axis": None, "scale": {"domain": x_domain}}
    vertical_axis = {"axis": None, "scale": {"domain": [0.0, 1.0]}}
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": height,
        "layer": [
            {
                "data": {"values": edge_rows},
                "mark": {"type": "rule", "color": "#8A94A6", "strokeWidth": 1.5},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", **axis},
                    "y": {"field": "y", "type": "quantitative", **vertical_axis},
                    "x2": {"field": "x2"},
                    "y2": {"field": "y2"},
                    "opacity": {
                        "field": "weight",
                        "type": "quantitative",
                        "scale": {"domain": [0.0, 1.0], "range": [0.2, 0.85]},
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "relationship", "title": "Connection"},
                        {"field": "details", "title": "Evidence"},
                        {"field": "weight", "title": "Weight", "format": ".3f"},
                    ],
                },
            },
            {
                "data": {"values": node_rows},
                "mark": {"type": "circle", "filled": True, "stroke": "white", "strokeWidth": 1},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", **axis},
                    "y": {"field": "y", "type": "quantitative", **vertical_axis},
                    "size": {"field": "size", "type": "quantitative", "legend": None},
                    "color": {
                        "field": color_field,
                        "type": "nominal",
                        "scale": {"domain": color_domain, "range": color_range},
                        "title": color_title,
                    },
                    "tooltip": [
                        {"field": "full_label", "title": "Node"},
                        {"field": color_field, "title": color_title},
                        {"field": "score", "title": "Score", "format": ".3f"},
                        {"field": "details", "title": "Details"},
                    ],
                },
            },
            {
                "data": {"values": node_rows},
                "mark": {"type": "text", "align": "left", "dx": 9, "fontSize": 10},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", **axis},
                    "y": {"field": "y", "type": "quantitative", **vertical_axis},
                    "text": {"field": "label"},
                    "color": {"value": "#D6DAE3"},
                },
            },
        ],
        "config": {
            "background": "#111827",
            "view": {"stroke": None},
            "legend": {"orient": "bottom", "labelColor": "#D6DAE3", "titleColor": "#D6DAE3"},
        },
    }


def _canvas_spec(nodes: list[Any], edges: list[Any]) -> dict[str, Any]:
    grouped: dict[float, list[Any]] = {}
    for node in nodes:
        grouped.setdefault(COLUMN_BY_KIND[node.kind], []).append(node)
    node_rows = []
    coordinates: dict[str, tuple[float, float]] = {}
    for column, values in grouped.items():
        ordered = sorted(values, key=lambda item: (item.kind, -item.score, item.label))
        for index, node in enumerate(ordered, 1):
            y = index / (len(ordered) + 1)
            coordinates[node.id] = (column, y)
            node_rows.append(
                {
                    "id": node.id,
                    "label": canvas_label(node.label),
                    "full_label": node.label,
                    "kind": node.kind,
                    "score": node.score,
                    "details": node.details,
                    "x": column,
                    "y": y,
                    "size": 180 + 520 * max(min(node.score, 1.0), 0.0),
                }
            )
    edge_rows = []
    for edge in edges:
        if edge.source not in coordinates or edge.target not in coordinates:
            continue
        sx, sy = coordinates[edge.source]
        tx, ty = coordinates[edge.target]
        edge_rows.append(
            {
                "relationship": edge.relationship.replace("_", " "),
                "details": edge.details,
                "weight": edge.weight,
                "x": sx,
                "y": sy,
                "x2": tx,
                "y2": ty,
            }
        )
    return layered_spec(
        node_rows,
        edge_rows,
        color_field="kind",
        color_domain=list(COLOR_BY_KIND),
        color_range=list(COLOR_BY_KIND.values()),
        color_title="Layer",
        x_domain=[-0.15, 3.55],
        height=680,
    )


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## Canvas")
    if state is None:
        st.info("Run an analysis on the Intake page first.")
        return
    st.caption(
        "Evidence and context → ATT&CK techniques, tools and campaigns → candidate groups → "
        "documented origin and intrusion type. Hover a node or link for its provenance."
    )
    controls = st.columns(2)
    candidate_max = max(2, min(15, len(state.fusion.entries)))
    top_groups = controls[0].slider(
        "Candidate groups",
        1,
        candidate_max,
        min(8, max(len(state.fusion.entries), 1)),
        key="canvas_top_groups",
    )
    max_evidence = controls[1].slider(
        "Evidence items",
        4,
        max(20, min(len(state.ledger.observations), 30)),
        min(12, max(len(state.ledger.observations), 4)),
        key="canvas_max_evidence",
    )
    canvas = build_analyst_canvas(
        engine.kb, state, top_groups=top_groups, max_evidence=max_evidence
    )
    available_kinds = [
        kind for kind in KIND_LABELS if any(node.kind == kind for node in canvas.nodes)
    ]
    available_relationships = sorted({edge.relationship for edge in canvas.edges})
    filters = st.columns(2)
    visible_kinds = filters[0].multiselect(
        "Layers",
        available_kinds,
        default=available_kinds,
        format_func=lambda kind: KIND_LABELS[kind],
    )
    visible_relationships = filters[1].multiselect(
        "Connection types",
        available_relationships,
        default=available_relationships,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    visible_ids = {node.id for node in canvas.nodes if node.kind in visible_kinds}
    nodes = [node for node in canvas.nodes if node.id in visible_ids]
    edges = [
        edge
        for edge in canvas.edges
        if edge.source in visible_ids
        and edge.target in visible_ids
        and edge.relationship in visible_relationships
    ]
    graph, inspector = st.columns([2.4, 1.0])
    with graph:
        if nodes:
            st.vega_lite_chart(spec=_canvas_spec(nodes, edges), use_container_width=True)
        else:
            st.info("Select at least one layer.")
    with inspector:
        st.markdown("#### Candidate inspector")
        candidate_ids = [entry.group_id for entry in state.fusion.entries[:top_groups]]
        if not candidate_ids:
            st.warning("No candidate group has documented overlap with the evidence yet.")
        else:
            selected = st.selectbox(
                "Candidate",
                candidate_ids,
                format_func=lambda gid: group_name(engine, gid),
                key="canvas_candidate",
            )
            dossier = build_candidate_dossier(engine.kb, state, selected)
            st.metric(
                "Rank",
                dossier["rank"],
                f"relative index {dossier['relative_score']:.1f}",
                delta_color="off",
            )
            if dossier["contested"]:
                st.warning("Classifier ranks disagree on this candidate.")
            if dossier["matched_techniques"]:
                st.dataframe(
                    pd.DataFrame(dossier["matched_techniques"]),
                    use_container_width=True,
                    hide_index=True,
                )
            if dossier["tools"]:
                st.caption("Named tools: " + ", ".join(item["name"] for item in dossier["tools"]))
            if dossier["unmatched_techniques"]:
                with st.expander("Observed but not documented for this candidate"):
                    st.dataframe(
                        pd.DataFrame(dossier["unmatched_techniques"]),
                        use_container_width=True,
                        hide_index=True,
                    )
    with st.expander("Evidence ledger and unresolved observations", expanded=False):
        st.dataframe(evidence_table(state), use_container_width=True, hide_index=True)
        linked = {
            edge.source.removeprefix("evidence:")
            for edge in canvas.edges
            if edge.source.startswith("evidence:")
        }
        unresolved = [o for o in state.ledger.observations if o.id not in linked]
        if unresolved:
            st.warning(
                f"{len(unresolved)} observation(s) have no technique, tool or actor-behaviour connection."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Source": o.provenance.source, "Kind": o.kind, "Observation": o.text}
                        for o in unresolved
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
