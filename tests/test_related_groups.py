"""Documented group overlap derived from ATT&CK data."""

from __future__ import annotations

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software
from apt_classifier.attack.related import related_groups, related_to


def _kb() -> AttackKnowledgeBase:
    return AttackKnowledgeBase(
        version="t",
        groups={
            "G0008": Group(
                id="G0008",
                name="Carbanak",
                aliases=["Anunak"],
                description="Carbanak is a group that shares tooling with FIN7.",
                software_ids=["S0030", "S0002"],
            ),
            "G0046": Group(
                id="G0046",
                name="FIN7",
                description="FIN7 is financially motivated.",
                software_ids=["S0030", "S0002"],
            ),
            "G0037": Group(
                id="G0037",
                name="FIN6",
                description="FIN6 targets POS systems.",
                software_ids=["S0002"],
            ),
            "G0016": Group(
                id="G0016",
                name="APT29",
                aliases=["Cozy Bear"],
                description="APT29 is Russian.",
                software_ids=["S0002"],
            ),
            "G0034": Group(
                id="G0034",
                name="Sandworm Team",
                description="Sandworm has overlaps with APT28.",
                software_ids=[],
            ),
            "G0007": Group(
                id="G0007", name="APT28", description="APT28 is a GRU unit.", software_ids=[]
            ),
        },
        software={
            "S0030": Software(id="S0030", name="Carbanak malware"),
            "S0002": Software(id="S0002", name="Mimikatz"),
        },
    )


def test_related_groups_from_mentions_and_rare_software_are_symmetric():
    kb = _kb()
    related = related_groups(kb)
    assert "G0046" in related["G0008"] and "G0008" in related["G0046"]
    assert (
        "Carbanak malware" in related["G0046"]["G0008"] or "references" in related["G0046"]["G0008"]
    )
    # commodity software shared by four groups does not relate anyone
    assert "G0016" not in related["G0037"] and "G0037" not in related["G0016"]
    # description cross-reference relates Sandworm and APT28 both ways
    assert related["G0034"]["G0007"].startswith("Sandworm Team description references APT28")
    assert "G0034" in related["G0007"]
    assert related_to(kb, "G0016") == {}
    assert related_to(kb, "G9999") == {}


def test_related_groups_are_memoized_per_knowledge_base():
    kb = _kb()
    first = related_groups(kb)
    assert related_groups(kb) is first


def test_shared_admin_utility_does_not_relate_groups():
    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software
    from apt_classifier.attack.related import related_groups

    kb = AttackKnowledgeBase(
        techniques={},
        groups={
            "G1": Group(id="G1", name="One", software_ids=["S_tool", "S_mal"]),
            "G2": Group(id="G2", name="Two", software_ids=["S_tool", "S_mal"]),
            "G3": Group(id="G3", name="Three", software_ids=["S_tool"]),
        },
        software={
            "S_tool": Software(id="S_tool", name="PsExec", software_type="tool"),
            "S_mal": Software(id="S_mal", name="Implant", software_type="malware"),
        },
    )
    related = related_groups(kb)
    assert "G2" in related["G1"] and "Implant" in related["G1"]["G2"]
    assert "G3" not in related["G1"]  # only the utility is shared with G3


def test_memo_does_not_serve_one_user_limit_for_another():
    kb = _kb()
    default = related_groups(kb)
    strict = related_groups(kb, rare_software_max_users=2)
    assert strict is not default
    # S0030 is used by exactly two groups (Carbanak, FIN7) in this KB, so both limits keep it
    assert "G0046" in strict.get("G0008", {}) and "G0046" in default.get("G0008", {})
    assert related_groups(kb) == default
