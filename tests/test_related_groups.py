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


def test_names_match_as_proper_nouns():
    from apt_classifier.attack.related import name_pattern

    # an all-capital vendor name matches only as written
    assert name_pattern("BITTER").search("the BITTER group")
    assert not name_pattern("BITTER").search("Operation Bitter Biscuit (2017)")
    assert not name_pattern("ZINC").search("zinc and Zinc")
    # other names keep a capital first letter; the rest may differ in case
    assert name_pattern("Winnti Group").search("the Winnti group cluster")
    assert not name_pattern("Snake").search("a snake in the grass")
    assert name_pattern("Snake").search("used Snake malware")
    # a name that starts in lowercase may also start a sentence
    assert name_pattern("menuPass").search("MenuPass targeted")
    assert name_pattern("menuPass").search("linked to menuPass")
    # whole words only, as before
    assert not name_pattern("APT41").search("APT410")
    assert not name_pattern("Threat Group-3390").search("Threat Group-33901")


def test_operation_names_and_ordinary_words_do_not_link_groups():
    kb = AttackKnowledgeBase(
        version="proper-nouns",
        groups={
            "G0131": Group(
                id="G0131",
                name="Tonto Team",
                description="Tonto Team acted through Operation Bitter Biscuit (2017).",
            ),
            "G1002": Group(id="G1002", name="BITTER", description="BITTER is a South Asian group."),
            "G0010": Group(
                id="G0010",
                name="Turla",
                aliases=["Snake"],
                description="Turla is a Russian group.",
            ),
            "G0044": Group(id="G0044", name="Winnti Group", description="Winnti Group is Chinese."),
            "G0096": Group(
                id="G0096",
                name="APT41",
                description="APT41 overlaps with the Winnti group; a snake was drawn on a slide.",
            ),
            "G0001": Group(id="G0001", name="Axiom", description="Axiom works with BITTER."),
        },
    )
    related = related_groups(kb)
    assert "G1002" not in related["G0131"]  # "Bitter" in an operation name is not BITTER
    assert "G0010" not in related["G0096"]  # "snake" is not Turla's alias Snake
    assert "G0044" in related["G0096"]  # "Winnti group" is Winnti Group
    assert "G1002" in related["G0001"]  # "BITTER" as written is


def test_the_real_knowledge_base_loses_only_the_false_match():
    from pathlib import Path

    import pytest

    from apt_classifier.attack.kb import load_knowledge_base
    from apt_classifier.attack.related import description_links

    path = Path(__file__).resolve().parents[1] / "data" / "kb" / "attack_kb.json"
    if not path.exists():
        pytest.skip("ATT&CK knowledge base not ingested")
    kb = load_knowledge_base(path)
    if str(kb.version) != "19.2":
        pytest.skip(f"counts were checked on ATT&CK 19.2, not {kb.version}")
    related = related_groups(kb)
    pairs = {tuple(sorted((a, b))) for a, others in related.items() for b in others}
    described = {
        tuple(sorted((a, b))) for a, others in description_links(kb).items() for b in others
    }
    # 87 related pairs (37 by description) before the proper-noun matching; Tonto Team / BITTER
    # through "Operation Bitter Biscuit" was the only false one
    assert len(pairs) == 86 and len(described) == 36
    assert ("G0131", "G1002") not in pairs
