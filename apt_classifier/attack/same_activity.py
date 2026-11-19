"""ATT&CK group pairs whose own descriptions document the same or overlapping activity.

ATT&CK links many groups in their descriptions, but often to say they are *different* actors:
"the two groups appear to be distinct" (Axiom / Winnti Group), different GRU units (APT28 /
Sandworm Team), "partnered with" (Mustard Tempest / Indrik Spider), members of one broader
collective (ShinyHunters / Scattered Spider). A documented-overlap cluster may only be named for a
pair ATT&CK documents as the same or overlapping activity (product-owner decision 2026-10-02,
docs 10.11 and 21.3.15 A′), so every description-linked pair of the reviewed ATT&CK version is
listed here with its verdict.

The review (2026-10-02, ATT&CK 19.2) read the description sentences in which one group names the
other and the alias lists. Five independent classification passes and a judge on the three split
pairs labelled the 37 pairs: 13 same activity, 18 distinct, 5 unclear, 1 false match ("Bitter"
inside "Operation Bitter Biscuit"). Only "same activity" counts; unclear counts as not. Since
``attack/related.py`` matches names as proper nouns the false match is no longer a description
link at all (36 remain). Its entry stays so that the cluster rule and the Top-three view would
still treat the pair as no link if a looser matcher revived it; ``related_groups`` itself does
not read this table.

Fail closed: a description link that is not in this table (a newer ATT&CK version) is never same
activity until it is reviewed, and a listed pair counts only while the knowledge base still
links the two groups by description. ``unreviewed_description_pairs`` lists what a new ATT&CK
version needs reviewed.
"""

from __future__ import annotations

from typing import Any

from apt_classifier.attack.related import description_links

REVIEWED_ATTACK_VERSION = "19.2"


def _pair(first: str, second: str) -> frozenset[str]:
    return frozenset((first, second))


# pair -> what ATT&CK documents (a short paraphrase of its own description text)
SAME_ACTIVITY: dict[frozenset[str], str] = {
    _pair("G0008", "G0046"): "ATT&CK: FIN7 may be linked to the Carbanak Group",
    _pair("G0008", "G0080"): "ATT&CK: Carbanak may be linked to Cobalt Group",
    _pair("G0009", "G0073"): (
        "ATT&CK: some analysts track Deep Panda and APT19 as the same group (ATT&CK calls it unclear)"
    ),
    _pair(
        "G0021", "G0090"
    ): "ATT&CK: WIRTE is believed to be a subgroup of Gaza Cybergang (Molerats)",
    _pair("G0030", "G0076"): "ATT&CK lists Thrip as an alias of Lotus Blossom",
    _pair("G0032", "G0067"): "ATT&CK: some researchers report APT37 activity under Lazarus Group",
    _pair("G0032", "G0082"): "ATT&CK: some researchers report APT38 activity under Lazarus Group",
    _pair(
        "G0032", "G0094"
    ): "ATT&CK: some researchers consolidate Kimsuky activity under Lazarus Group",
    _pair("G0032", "G0138"): "ATT&CK: Andariel is considered a sub-set of Lazarus Group",
    _pair("G0032", "G1049"): "ATT&CK: AppleJeus is associated with the Lazarus Group umbrella",
    _pair(
        "G0044", "G0096"
    ): "ATT&CK: APT41 overlaps at least partially with Winnti Group reporting",
    _pair(
        "G0045", "G1054"
    ): "ATT&CK: MirrorFace is believed to be a subgroup under the menuPass umbrella",
    _pair(
        "G0129", "G1014"
    ): "ATT&CK: some researchers concluded LuminousMoth is connected to Mustang Panda",
}

# reviewed and not the same activity: distinct, unclear or a false match (vetoes nesting too)
NOT_SAME_ACTIVITY: dict[frozenset[str], str] = {
    _pair("G0001", "G0044"): "distinct: the two groups appear to be distinct",
    _pair("G0002", "G0017"): "unclear: a direct or indirect relationship",
    _pair("G0004", "G0044"): "unclear: closely linked other groups",
    _pair("G0007", "G0034"): "distinct: different GRU units that assisted each other",
    _pair("G0013", "G0019"): "distinct: the two groups do not appear to be exact matches",
    _pair("G0024", "G0029"): "distinct: not concluded that the groups are the same",
    _pair("G0025", "G0044"): "unclear: closely linked other groups",
    _pair("G0032", "G1036"): "distinct: has differentiated its tradecraft since 2023",
    _pair("G0040", "G0142"): "distinct: similarities in malware code and targets only",
    _pair("G0044", "G1006"): "distinct: techniques and infrastructure are separate",
    _pair("G0049", "G1001"): "distinct: tracked as a separate entity",
    _pair("G0055", "G0056"): "distinct: similarity through overlapping victims only",
    _pair("G0055", "G0063"): "distinct: no evidence that the group names are aliases",
    _pair("G0059", "G1044"): "distinct: appear to be distinct entities",
    _pair("G0064", "G1001"): "distinct: tracked as a separate entity",
    _pair("G0096", "G1006"): "distinct: techniques and infrastructure are separate",
    _pair("G0096", "G1042"): "unclear: overlaps with various PRC-linked groups",
    _pair("G0102", "G1011"): "unclear: closely linked with ransomware deployment",
    _pair("G0119", "G1020"): "distinct: partnered to provide access",
    _pair("G0121", "G1008"): "distinct: mimics the other group's infection chain",
    # no longer matched since attack/related.py matches names as proper nouns; kept for the
    # cluster rule and the Top-three view (related_groups does not read this table)
    _pair("G0131", "G1002"): "false match: 'Bitter' inside the name Operation Bitter Biscuit",
    _pair("G1003", "G1031"): "distinct: analysis indicates these are distinct clusters",
    _pair("G1004", "G1057"): "distinct: members of one broader collective",
    _pair("G1015", "G1057"): "distinct: members of one broader collective",
}


def review_verdict(first: str, second: str) -> str | None:
    """The reviewed verdict text for a pair that is not the same activity, else None."""
    return NOT_SAME_ACTIVITY.get(_pair(first, second))


def is_false_match(first: str, second: str) -> bool:
    """The review found the description match refers to something else (ATT&CK does not link
    the two by description at all)."""
    return str(NOT_SAME_ACTIVITY.get(_pair(first, second), "")).startswith("false match")


def review_status(kb: Any, first: str, second: str) -> str | None:
    """ "same_activity", "not_same_activity" or None (not reviewed, or no description link now)."""
    key = _pair(first, second)
    if not description_links(kb).get(first, {}).get(second):
        # a listed pair counts only while ATT&CK still links the two by description
        return "not_same_activity" if key in NOT_SAME_ACTIVITY else None
    if key in SAME_ACTIVITY:
        return "same_activity"
    if key in NOT_SAME_ACTIVITY:
        return "not_same_activity"
    return None


def same_activity_statement(kb: Any, first: str, second: str) -> str | None:
    """What ATT&CK documents when it documents the pair as the same activity, else None."""
    if review_status(kb, first, second) != "same_activity":
        return None
    return SAME_ACTIVITY[_pair(first, second)]


def unreviewed_description_pairs(kb: Any) -> list[tuple[str, str]]:
    """Description-linked pairs this review has not labelled (empty for ATT&CK 19.2)."""
    links = description_links(kb)
    pairs = {tuple(sorted((first, second))) for first, others in links.items() for second in others}
    return sorted(
        (first, second)
        for first, second in pairs
        if _pair(first, second) not in SAME_ACTIVITY
        and _pair(first, second) not in NOT_SAME_ACTIVITY
    )
