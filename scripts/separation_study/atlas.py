"""How separable are the ATT&CK groups by each kind of evidence, inside ATT&CK itself?

Three readings of the knowledge base, no report involved:

1. profiles and software: exclusive and rare techniques (exact ids and parent level), exclusive
   software, nested profiles (a group's techniques mostly inside another group's);
2. ideal identifiability: k techniques drawn from the group's own profile (no noise, no extraction
   error), ranked by IDF cosine over every group, exact ids against parent level;
3. ATT&CK's single-group campaigns: a real operation whose technique and software lists ATT&CK
   documents directly (no extraction error), ranked against the group profiles by techniques
   (exact, parent, rare only) and by the campaign's software (raw bundle; rarity weighted).

    PYTHONPATH=. .venv/bin/python scripts/separation_study/atlas.py
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import load_knowledge_base  # noqa: E402

KB_PATH = REPO_ROOT / "data/kb/attack_kb.json"
BUNDLE_PATH = REPO_ROOT / "data/raw/enterprise-attack.json"


def parent(technique: str) -> str:
    return technique.split(".")[0]


def spread(values: list[float]) -> str:
    ordered = sorted(values)
    return (
        f"median {statistics.median(ordered):.0f}, p10 {ordered[len(ordered) // 10]:.0f}, "
        f"p90 {ordered[9 * len(ordered) // 10]:.0f}"
    )


class Profiles:
    def __init__(self, profiles: dict[str, set[str]]) -> None:
        self.profiles = profiles
        self.count = len(profiles)
        self.df = Counter(t for v in profiles.values() for t in v)
        self.idf = {t: math.log((self.count + 1) / (n + 1)) + 1 for t, n in self.df.items()}
        self.unseen = math.log(self.count + 1) + 1
        self.norm = {g: math.sqrt(sum(self.idf[t] ** 2 for t in v)) for g, v in profiles.items()}

    def rank(self, case: set[str], truth: str) -> int:
        """The truth's rank by IDF cosine (ties counted in its favour)."""
        case_norm = math.sqrt(sum(self.idf.get(t, self.unseen) ** 2 for t in case)) or 1.0
        scores = {
            g: sum(self.idf[t] ** 2 for t in case & v) / (case_norm * self.norm[g])
            for g, v in self.profiles.items()
        }
        return 1 + sum(1 for value in scores.values() if value > scores[truth])


def campaign_software(path: Path) -> dict[str, set[str]]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    objects = {o["id"]: o for o in bundle["objects"]}

    def external_id(obj: dict) -> str | None:
        return next(
            (
                ref["external_id"]
                for ref in obj.get("external_references", [])
                if ref.get("source_name") == "mitre-attack"
            ),
            None,
        )

    out: dict[str, set[str]] = {}
    for obj in bundle["objects"]:
        if (
            obj["type"] == "relationship"
            and obj["relationship_type"] == "uses"
            and obj["source_ref"].startswith("campaign--")
            and obj["target_ref"].split("--")[0] in ("malware", "tool")
            and not obj.get("revoked")
        ):
            campaign = external_id(objects[obj["source_ref"]])
            software = external_id(objects[obj["target_ref"]])
            if campaign and software:
                out.setdefault(campaign, set()).add(software)
    return out


def main() -> int:
    kb = load_knowledge_base(KB_PATH)
    techniques = {g: set(v) for g, v in kb.group_techniques.items() if v}
    software = {g: set(kb.group_software.get(g, ())) for g in techniques}
    exact = Profiles(techniques)
    parents = Profiles({g: {parent(t) for t in v} for g, v in techniques.items()})
    sdf = Counter(s for v in software.values() for s in v)
    n = exact.count
    print(f"ATT&CK {kb.version}: {n} groups with techniques")

    print("\n== profiles")
    print("techniques per group:", spread([len(v) for v in techniques.values()]))
    print(
        "groups documenting a group's median technique:",
        spread([statistics.median(exact.df[t] for t in v) for v in techniques.values()]),
    )
    exclusive = {g: [t for t in v if exact.df[t] == 1] for g, v in techniques.items()}
    rare = {g: [t for t in v if exact.df[t] <= 3] for g, v in techniques.items()}
    print(
        f"groups with a technique no other group has: {sum(1 for v in exclusive.values() if v)}/{n}"
    )
    print(
        f"groups with a technique documented for <=3 groups: {sum(1 for v in rare.values() if v)}/{n}"
    )
    print(
        "groups with a parent technique no other group has: "
        f"{sum(1 for g, v in parents.profiles.items() if any(parents.df[t] == 1 for t in v))}/{n}"
    )
    print(
        "share of a profile that is sub-techniques: "
        f"{statistics.mean(sum(1 for t in v if '.' in t) / len(v) for v in techniques.values()):.2f}"
    )

    print("\n== software")
    print("software per group:", spread([len(v) for v in software.values()]))
    print(
        f"groups with software no other group uses: {sum(1 for v in software.values() if any(sdf[s] == 1 for s in v))}/{n}"
    )
    print(f"groups with no software: {sum(1 for v in software.values() if not v)}/{n}")

    print("\n== nested profiles")
    largest = set(sorted(techniques, key=lambda g: -len(techniques[g]))[:10])
    contained = {}
    for g, v in techniques.items():
        contained[g] = max((len(v & w) / len(v), h) for h, w in techniques.items() if h != g)
    nested = [g for g, (share, _) in contained.items() if share >= 0.8]
    print(
        f"groups with >=80% of their techniques inside another group's profile: {len(nested)}/{n} "
        f"(inside one of the 10 largest profiles: {sum(1 for g in nested if contained[g][1] in largest)})"
    )

    print("\n== ideal identifiability: k techniques of the group's own profile, IDF cosine")
    rng = random.Random(0)
    for k in (3, 5, 10, 20):
        hits = parent_hits = draws = 0
        for g, v in techniques.items():
            if len(v) < k:
                continue
            for _ in range(5):
                case = set(rng.sample(sorted(v), k))
                hits += exact.rank(case, g) == 1
                parent_hits += parents.rank({parent(t) for t in case}, g) == 1
                draws += 1
        print(
            f"  k={k:2d}: top-1 exact ids {hits / draws:.2f}, parent level {parent_hits / draws:.2f} ({draws} draws)"
        )

    print("\n== ATT&CK single-group campaigns against the group profiles")
    camp_soft = campaign_software(BUNDLE_PATH)
    rows = []
    for campaign, groups in kb.campaign_groups.items():
        case = set(kb.campaign_techniques.get(campaign) or ())
        if len(groups) != 1 or groups[0] not in techniques or not case:
            continue
        g = groups[0]
        rare_case = {t for t in case if 0 < exact.df.get(t, 0) <= 5}
        named = [s for s in camp_soft.get(campaign, ()) if sdf.get(s)]
        soft_rank = None
        if named:
            scores = {h: sum(1 / sdf[s] for s in named if s in software[h]) for h in techniques}
            if scores[g] > 0:
                soft_rank = 1 + sum(1 for value in scores.values() if value > scores[g])
        rows.append(
            {
                "inside": len(case & techniques[g]) / len(case),
                "exact": exact.rank(case, g),
                "parent": parents.rank({parent(t) for t in case}, g),
                "rare": exact.rank(rare_case, g) if rare_case else None,
                "software": soft_rank,
                "software_links": len(camp_soft.get(campaign, ())),
                "software_inside": sum(1 for s in camp_soft.get(campaign, ()) if s in software[g]),
            }
        )
    total = len(rows)
    print(
        f"{total} campaigns; their techniques inside the group's own profile: median "
        f"{statistics.median(r['inside'] for r in rows):.2f}; their software documented for the "
        f"group too: {sum(r['software_inside'] for r in rows) / max(1, sum(r['software_links'] for r in rows)):.2f}"
    )
    for label, key in (
        ("techniques, exact ids", "exact"),
        ("techniques, parent level", "parent"),
        ("rare techniques only (<=5 groups)", "rare"),
        ("software by rarity", "software"),
    ):
        fired = [r[key] for r in rows if r[key] is not None]
        print(
            f"  {label:34s}: answers {len(fired)}/{total}, top-1 "
            f"{sum(1 for v in fired if v == 1)}/{total}, top-5 {sum(1 for v in fired if v <= 5)}/{total}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
