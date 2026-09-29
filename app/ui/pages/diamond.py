"""Diamond Model view: adversary, capability, infrastructure and victim vertices."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import status_strip
from app.ui.pages.canvas import canvas_label, layered_spec
from apt_classifier.config import Settings
from apt_classifier.diamond import DiamondModel, build_diamond_model
from apt_classifier.session import AttributionEngine

VERTEX_LABELS = ("Adversary", "Capability", "Infrastructure", "Victim")
VERTEX_KEYS = {
    "Adversary": "adversary",
    "Capability": "capability",
    "Infrastructure": "infrastructure",
    "Victim": "victim",
}
COLOR_BY_VERTEX = {
    "adversary": "#E45756",
    "capability": "#F2CF5B",
    "infrastructure": "#4C78A8",
    "victim": "#54A24B",
}


def _diamond_spec(model: DiamondModel) -> dict[str, Any]:
    x_by_vertex = {vertex: index for index, vertex in enumerate(VERTEX_KEYS.values())}
    grouped: dict[str, list[Any]] = {vertex: [] for vertex in x_by_vertex}
    for node in model.nodes:
        grouped[node.vertex].append(node)
    coordinates: dict[str, tuple[float, float]] = {}
    node_rows = []
    for vertex, nodes in grouped.items():
        ordered = sorted(nodes, key=lambda node: (-node.score, node.label))
        for index, node in enumerate(ordered, 1):
            y = index / (len(ordered) + 1)
            coordinates[node.id] = (float(x_by_vertex[vertex]), y)
            node_rows.append(
                {
                    "id": node.id,
                    "label": canvas_label(node.label, 28),
                    "full_label": node.label,
                    "vertex": vertex,
                    "score": node.score,
                    "status": node.status,
                    "details": f"{node.status}: {node.details}",
                    "x": float(x_by_vertex[vertex]),
                    "y": y,
                    "size": 180 + 520 * max(min(node.score, 1.0), 0.0),
                }
            )
    edge_rows = []
    for edge in model.edges:
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
        color_field="vertex",
        color_domain=list(COLOR_BY_VERTEX),
        color_range=list(COLOR_BY_VERTEX.values()),
        color_title="Vertex",
        x_domain=[-0.15, 3.15],
        height=520,
    )


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## Diamond Model")
    if state is None:
        st.info("Run an analysis on the Intake page first.")
        return
    st.caption(
        "The case organised into adversary, capability, infrastructure and victim vertices. "
        "An evidence map, not an attribution verdict."
    )
    controls = st.columns(3)
    top_groups = controls[0].slider(
        "Adversary hypotheses",
        1,
        max(2, min(15, len(state.fusion.entries))),
        min(8, max(len(state.fusion.entries), 1)),
        key="diamond_top_groups",
    )
    max_capabilities = controls[1].slider(
        "Capability nodes", 4, 30, 15, key="diamond_max_capabilities"
    )
    max_infrastructure = controls[2].slider(
        "Infrastructure nodes", 4, 30, 15, key="diamond_max_infrastructure"
    )
    model = build_diamond_model(
        engine.kb,
        state,
        top_groups=top_groups,
        max_capabilities=max_capabilities,
        max_infrastructure=max_infrastructure,
    )
    metrics = st.columns(4)
    for column, vertex in zip(metrics, VERTEX_LABELS, strict=True):
        column.metric(vertex, len(model.nodes_for(VERTEX_KEYS[vertex])))
    st.vega_lite_chart(_diamond_spec(model), use_container_width=True)
    tabs = st.tabs([*VERTEX_LABELS, "Relationships", "Caveats"])
    for tab, vertex_label in zip(tabs[:4], VERTEX_LABELS, strict=True):
        with tab:
            rows = [
                {
                    "Node": node.label,
                    "Status": node.status,
                    "Score": round(node.score, 3),
                    "Details": node.details,
                }
                for node in model.nodes_for(VERTEX_KEYS[vertex_label])
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(
            pd.DataFrame(
                {
                    "Source": edge.source,
                    "Target": edge.target,
                    "Relationship": edge.relationship.replace("_", " ").title(),
                    "Weight": round(edge.weight, 3),
                    "Details": edge.details,
                }
                for edge in model.edges
            ),
            use_container_width=True,
            hide_index=True,
        )
    with tabs[5]:
        for caveat in model.caveats:
            st.markdown(f"- {caveat}")
