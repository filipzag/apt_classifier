import math
from typing import Any

from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    build_result,
    group_ids,
    kb_mapping,
)

COMMODITY_TOOL_WEIGHT_CAP = 1.5
DECISIVE_MALWARE_MAX_USERS = 2


def software_rarity_weights(
    kb: Any,
    software_ids: set[str] | frozenset[str],
    *,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
) -> dict[str, float]:
    """Rarity weight per observed software id: ``log((N + 1) / (n + 1)) + 1`` over the groups
    documented for it, with ATT&CK *tool* entries capped at ``tool_weight_cap``."""
    software_groups = kb_mapping(kb, "software_groups")
    software = kb_mapping(kb, "software")
    group_count = len(kb_mapping(kb, "groups")) or 1
    weights: dict[str, float] = {}
    for software_id in software_ids:
        rarity = (
            math.log((group_count + 1) / (len(set(software_groups.get(software_id, ()))) + 1)) + 1.0
        )
        kind = str(getattr(software.get(software_id), "software_type", "") or "")
        weights[software_id] = min(rarity, tool_weight_cap) if kind == "tool" else rarity
    return weights


def decisive_software_factor(
    kb: Any,
    software_ids: set[str] | frozenset[str],
    *,
    boost: float,
    max_users: int = DECISIVE_MALWARE_MAX_USERS,
) -> tuple[float, tuple[str, ...]]:
    """Vote multiplier for the software classifier when the case names a bespoke implant.

    Rank fusion cannot express evidence strength: a report that names a malware family
    documented for one or two groups gets the same credit as one that names PsExec. When at
    least one observed *malware* entry (never a tool) is documented for at most ``max_users``
    groups, the software vote is multiplied by ``1 + boost``; when the rarest observed malware
    has at most twice that many users the multiplier is ``1 + boost / 2``. Returns the factor
    and the ids that triggered it so the decision can say why.
    """
    if boost <= 0 or not software_ids:
        return 1.0, ()
    software = kb_mapping(kb, "software")
    software_groups = kb_mapping(kb, "software_groups")
    decisive: list[str] = []
    supporting: list[str] = []
    for software_id in sorted(software_ids):
        entry = software.get(software_id)
        if entry is None or str(getattr(entry, "software_type", "") or "") != "malware":
            continue
        users = len(set(software_groups.get(software_id, ())))
        if 0 < users <= max_users:
            decisive.append(software_id)
        elif 0 < users <= 2 * max_users:
            supporting.append(software_id)
    if decisive:
        return 1.0 + float(boost), tuple(decisive)
    if supporting:
        return 1.0 + float(boost) / 2.0, tuple(supporting)
    return 1.0, ()


class SoftwareClassifier:
    """Software/tool link overlap between observed tools and a group's documented tools.

    With ``use_idf`` (default) each tool is weighted by its rarity across groups,
    ``log((N + 1) / (n_groups_using_it + 1)) + 1``, so a commodity tool documented for
    dozens of groups (Mimikatz, PsExec, Cobalt Strike) counts far less than a bespoke
    implant. ATT&CK marks each software entry as ``malware`` or ``tool``; a *tool* is a
    legitimate utility (cmd, Arp, PsExec, Mimikatz) whose sparse group documentation says
    nothing about who used it, so its weight is capped at ``tool_weight_cap`` (default 1.5)
    while malware keeps its full rarity weight: a named implant documented for one group is
    decisive, a named admin utility is a hint. The score is the rarity-weighted fraction of
    observed software the group is linked to; with ``use_idf=False`` it is the plain fraction.
    """

    name = "software_tools"

    def __init__(
        self, use_idf: bool = True, tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP
    ) -> None:
        self.use_idf = use_idf
        self.tool_weight_cap = float(tool_weight_cap)

    def _weights(self, context: ClassifierContext) -> dict[str, float]:
        if not self.use_idf:
            return dict.fromkeys(context.software_ids, 1.0)
        return software_rarity_weights(
            context.kb, context.software_ids, tool_weight_cap=self.tool_weight_cap
        )

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        software_groups = kb_mapping(context.kb, "software_groups")
        group_software = kb_mapping(context.kb, "group_software")
        weights = self._weights(context)
        # fsum: exactly rounded, so neither sum depends on set or dict iteration order
        total = math.fsum(weights.values())
        scores: dict[str, float] = {}
        signals: dict[str, tuple[str, ...]] = {}
        for group_id in group_ids(context):
            linked = set(group_software.get(group_id, ())) & context.software_ids
            if not linked:
                linked = {
                    software_id
                    for software_id in context.software_ids
                    if group_id in set(software_groups.get(software_id, ()))
                }
            if linked and total > 0:
                scores[group_id] = math.fsum(weights[software_id] for software_id in linked) / total
                signals[group_id] = tuple(sorted(linked))
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(scores),
            signals=signals,
        )
