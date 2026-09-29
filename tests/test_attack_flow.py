"""Tests for Attack Flow construction, export, and similarity."""

from types import SimpleNamespace

from apt_classifier.attack_flow import (
    AttackFlow,
    FlowAction,
    FlowAsset,
    FlowEdge,
    build_attack_flow,
    flow_similarity,
)


def _state(matches, observations):
    return SimpleNamespace(
        retrieval=SimpleNamespace(matches=matches),
        ledger=SimpleNamespace(observations=observations),
    )


def _match(technique_id, technique_name, observation_ids, score=0.9, matched_text="text"):
    return SimpleNamespace(
        technique_id=technique_id,
        technique_name=technique_name,
        observation_ids=observation_ids,
        score=score,
        matched_text=matched_text,
    )


def _observation(obs_id, text="action", metadata=None, source="free_text"):
    return SimpleNamespace(
        id=obs_id,
        text=text,
        metadata=metadata or {},
        provenance=SimpleNamespace(source=source),
    )


def _kb(techniques=None):
    return SimpleNamespace(
        techniques=techniques
        or {
            "T1059.001": SimpleNamespace(id="T1059.001", name="PowerShell", tactics=["execution"]),
            "T1003.001": SimpleNamespace(
                id="T1003.001", name="LSASS Memory", tactics=["credential-access"]
            ),
        }
    )


def test_build_attack_flow_creates_actions_from_matches():
    kb = _kb()
    obs = [_observation("obs-1", "PowerShell encoded command")]
    matches = [_match("T1059.001", "PowerShell", ["obs-1"])]
    state = _state(matches, obs)
    flow = build_attack_flow(kb, state, case_id="case-1")
    assert len(flow.actions) == 1
    assert flow.actions[0].technique_id == "T1059.001"
    assert flow.actions[0].technique_name == "PowerShell"
    assert flow.actions[0].tactic == "execution"
    assert flow.actions[0].observation_id == "obs-1"


def test_build_attack_flow_adds_chronological_edges_from_timestamps():
    kb = _kb()
    obs = [
        _observation("obs-1", "first action", {"timestamp": "2024-01-01T10:00:00Z"}),
        _observation("obs-2", "second action", {"timestamp": "2024-01-01T11:00:00Z"}),
    ]
    matches = [
        _match("T1059.001", "PowerShell", ["obs-1"]),
        _match("T1003.001", "LSASS Memory", ["obs-2"]),
    ]
    state = _state(matches, obs)
    flow = build_attack_flow(kb, state)
    precedes_edges = [e for e in flow.edges if e.edge_type == "precedes"]
    assert len(precedes_edges) == 1
    assert precedes_edges[0].is_chronological is True
    assert "Chronological" in precedes_edges[0].details


def test_build_attack_flow_no_chronological_edges_without_timestamps():
    kb = _kb()
    obs = [
        _observation("obs-1", "first action"),
        _observation("obs-2", "second action"),
    ]
    matches = [
        _match("T1059.001", "PowerShell", ["obs-1"]),
        _match("T1003.001", "LSASS Memory", ["obs-2"]),
    ]
    state = _state(matches, obs)
    flow = build_attack_flow(kb, state)
    precedes_edges = [e for e in flow.edges if e.edge_type == "precedes"]
    assert len(precedes_edges) == 0


def test_build_attack_flow_adds_assets_from_metadata():
    kb = _kb()
    obs = [
        _observation("obs-1", "action on host", {"host": "WORKSTATION-01"}),
    ]
    matches = [_match("T1059.001", "PowerShell", ["obs-1"])]
    state = _state(matches, obs)
    flow = build_attack_flow(kb, state)
    assert len(flow.assets) == 1
    assert flow.assets[0].name == "WORKSTATION-01"
    acts_on_edges = [e for e in flow.edges if e.edge_type == "acts_on"]
    assert len(acts_on_edges) == 1


def test_attack_flow_to_stix_export():
    flow = AttackFlow(
        id="attack-flow:test",
        name="Test Flow",
        actions=(
            FlowAction(
                id="action:1",
                technique_id="T1059.001",
                technique_name="PowerShell",
                description="test",
                observation_id="obs-1",
                timestamp="2024-01-01T10:00:00Z",
                tactic="execution",
                confidence=0.9,
            ),
        ),
        assets=(FlowAsset(id="asset:1", name="HOST-1", asset_type="system"),),
        edges=(
            FlowEdge(
                source="action:1",
                target="asset:1",
                edge_type="acts_on",
                confidence=0.5,
            ),
        ),
    )
    stix = flow.to_stix()
    assert stix["type"] == "bundle"
    assert stix["id"] == "attack-flow:test"
    assert any(obj.get("type") == "attack-flow" for obj in stix["objects"])
    assert any(obj.get("type") == "x-attack-flow-asset" for obj in stix["objects"])


def test_flow_similarity_identical_flows():
    flow_a = AttackFlow(
        id="a",
        name="A",
        actions=(
            FlowAction(
                id="1", technique_id="T1", technique_name="T1", description="", observation_id="o1"
            ),
            FlowAction(
                id="2", technique_id="T2", technique_name="T2", description="", observation_id="o2"
            ),
        ),
        assets=(),
        edges=(),
    )
    flow_b = AttackFlow(
        id="b",
        name="B",
        actions=(
            FlowAction(
                id="1", technique_id="T1", technique_name="T1", description="", observation_id="o1"
            ),
            FlowAction(
                id="2", technique_id="T2", technique_name="T2", description="", observation_id="o2"
            ),
        ),
        assets=(),
        edges=(),
    )
    assert abs(flow_similarity(flow_a, flow_b) - 1.0) < 1e-9


def test_flow_similarity_partial_overlap():
    flow_a = AttackFlow(
        id="a",
        name="A",
        actions=(
            FlowAction(
                id="1", technique_id="T1", technique_name="T1", description="", observation_id="o1"
            ),
            FlowAction(
                id="2", technique_id="T2", technique_name="T2", description="", observation_id="o2"
            ),
        ),
        assets=(),
        edges=(),
    )
    flow_b = AttackFlow(
        id="b",
        name="B",
        actions=(
            FlowAction(
                id="1", technique_id="T2", technique_name="T2", description="", observation_id="o1"
            ),
            FlowAction(
                id="2", technique_id="T3", technique_name="T3", description="", observation_id="o2"
            ),
        ),
        assets=(),
        edges=(),
    )
    # intersection={T2}, union={T1,T2,T3} → 1/3
    assert abs(flow_similarity(flow_a, flow_b) - 1 / 3) < 1e-9


def test_flow_similarity_no_overlap():
    flow_a = AttackFlow(
        id="a",
        name="A",
        actions=(
            FlowAction(
                id="1", technique_id="T1", technique_name="T1", description="", observation_id="o1"
            ),
        ),
        assets=(),
        edges=(),
    )
    flow_b = AttackFlow(
        id="b",
        name="B",
        actions=(
            FlowAction(
                id="1", technique_id="T2", technique_name="T2", description="", observation_id="o1"
            ),
        ),
        assets=(),
        edges=(),
    )
    assert flow_similarity(flow_a, flow_b) == 0.0


def test_flow_to_dict_serializable():
    flow = AttackFlow(
        id="test",
        name="Test",
        actions=(
            FlowAction(
                id="a1",
                technique_id="T1",
                technique_name="T1",
                description="d",
                observation_id="o1",
            ),
        ),
        assets=(FlowAsset(id="as1", name="Host"),),
        edges=(FlowEdge(source="a1", target="as1", edge_type="acts_on"),),
    )
    d = flow.to_dict()
    assert "actions" in d
    assert "assets" in d
    assert "edges" in d
    assert d["actions"][0]["technique_id"] == "T1"
