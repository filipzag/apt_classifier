"""Labelled ATT&CK free text for a text -> group model (ATT&CK only, never benchmark data).

Writes corpus.json (STUDY_DIR, or the path given as the first argument) with records {text, source, owner, groups, technique_id, campaign_dup}:
- source "group": the group's procedure examples (intrusion-set uses attack-pattern), owner = group;
- source "software": malware/tool procedure examples (software uses attack-pattern), owner =
  software id, groups = every group ATT&CK documents using that software (may be empty);
- source "group_desc": sentences of the group description, owner = group;
- source "campaign": campaign procedure examples (campaign uses attack-pattern / malware / tool),
  owner = campaign id, groups = attributed groups. Used as the development set.

Every text: ATT&CK citations stripped, every group name/alias -> "the adversary" (the benchmarks'
evidence pattern), every campaign name/alias -> "the operation" (campaign names also appear in
group procedure sentences, a shortcut between training and the development set). Software names
stay (an analyst reads them; the tradecraft test level masks them in the reports).
``campaign_dup`` marks a training text whose normalized form equals a campaign text.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from harness import STUDY_DIR

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.attack.parse import strip_attack_citations
from apt_classifier.config import Settings
from scripts.attribution_benchmark import _evidence_name_pattern, actor_names

out = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY_DIR / "corpus.json"
settings = Settings()
kb = AttackKnowledgeBase.load(settings.kb_path)
raw = json.loads(
    Path(settings.kb_path).parent.parent.joinpath("raw", "enterprise-attack.json").read_text()
)
objects = raw["objects"]


def attack_id(obj):
    return next(
        (
            r["external_id"]
            for r in obj.get("external_references", [])
            if r.get("source_name") == "mitre-attack"
        ),
        None,
    )


live = [o for o in objects if not o.get("revoked") and not o.get("x_mitre_deprecated")]
ids = {o["id"]: attack_id(o) for o in live if "id" in o}
types = {o["id"]: o["type"] for o in live if "id" in o}

actor_pattern = _evidence_name_pattern(actor_names(kb))
campaign_names = sorted(
    {n for c in kb.campaigns.values() for n in (c.name, *c.aliases) if n and len(n) > 2},
    key=lambda s: (-len(s), s.casefold()),
)
campaign_pattern = re.compile(
    r"(?<![\w.-])(?:" + "|".join(re.escape(n) for n in campaign_names) + r")(?![\w-]|\.\w)",
    re.IGNORECASE,
)
ARTICLE = re.compile(r"\b(?:the|a|an)\s+(the (?:adversary|operation))", re.IGNORECASE)


def clean(text):
    text = strip_attack_citations(text or "").strip()
    text = actor_pattern.sub(lambda m: "the adversary" + (m.group("possessive") or ""), text)
    text = campaign_pattern.sub("the operation", text)
    text = ARTICLE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


software_groups = {}
for group_id, software_ids in kb.group_software.items():
    for software_id in software_ids:
        software_groups.setdefault(software_id, set()).add(group_id)

records = []
for group_id, group in sorted(kb.groups.items()):
    for technique_id, values in sorted(group.technique_behaviors.items()):
        for value in values:
            text = clean(value)
            if len(text.split()) >= 3:
                records.append(
                    {
                        "text": text,
                        "source": "group",
                        "owner": group_id,
                        "groups": [group_id],
                        "technique_id": technique_id,
                    }
                )
    for sentence in re.split(
        r"(?<=[.!?])\s+(?=[A-Z(])", strip_attack_citations(group.description or "")
    ):
        text = clean(sentence)
        if len(text.split()) >= 4:
            records.append(
                {
                    "text": text,
                    "source": "group_desc",
                    "owner": group_id,
                    "groups": [group_id],
                    "technique_id": None,
                }
            )

campaign_groups = {}
for o in live:
    if o.get("type") == "relationship" and o.get("relationship_type") == "attributed-to":
        c, g = ids.get(o["source_ref"]), ids.get(o["target_ref"])
        if c and g and c.startswith("C") and g in kb.groups:
            campaign_groups.setdefault(c, set()).add(g)

for o in live:
    if (
        o.get("type") != "relationship"
        or o.get("relationship_type") != "uses"
        or not o.get("description")
    ):
        continue
    source_type = types.get(o["source_ref"])
    source, target = ids.get(o["source_ref"]), ids.get(o["target_ref"])
    target_type = types.get(o["target_ref"])
    if not source or not target:
        continue
    text = clean(o["description"])
    if len(text.split()) < 3:
        continue
    if source_type in ("malware", "tool") and target_type == "attack-pattern":
        records.append(
            {
                "text": text,
                "source": "software",
                "owner": source,
                "groups": sorted(software_groups.get(source, ())),
                "technique_id": target,
            }
        )
    elif source_type == "campaign" and target_type in ("attack-pattern", "malware", "tool"):
        records.append(
            {
                "text": text,
                "source": "campaign",
                "owner": source,
                "groups": sorted(campaign_groups.get(source, ())),
                "technique_id": target if target_type == "attack-pattern" else None,
                "software_id": target if target_type != "attack-pattern" else None,
            }
        )

campaign_norms = {norm(r["text"]) for r in records if r["source"] == "campaign"}
for r in records:
    r["campaign_dup"] = r["source"] != "campaign" and norm(r["text"]) in campaign_norms
print(Counter(r["source"] for r in records))
print("training texts identical to a campaign text:", sum(r["campaign_dup"] for r in records))
print(
    "software texts with no documented group:",
    sum(1 for r in records if r["source"] == "software" and not r["groups"]),
)
dev = Counter(r["owner"] for r in records if r["source"] == "campaign" and len(r["groups"]) == 1)
print(
    "single-group campaigns:", len(dev), "with >=3 texts:", sum(1 for v in dev.values() if v >= 3)
)
left = sum(1 for r in records if actor_pattern.search(r["text"]))
print("texts still matching a group name/alias:", left)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(records))
