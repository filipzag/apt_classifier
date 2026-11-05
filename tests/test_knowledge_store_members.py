"""The knowledge store's two ensemble members (members.py), the name pins (pins.py) and the
naming gate's limit on the store software member."""

from __future__ import annotations

import gzip
import json
from datetime import date
from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique
from apt_classifier.attribution import naming_limits, supporting_families
from apt_classifier.knowledge_store.members import (
    STORE_SOFTWARE_NAME,
    Case,
    StoreMembers,
    StoreState,
    citation_keys,
    load_store_members_for,
    members_path,
    statement_key,
    statement_keys,
    store_bound_fingerprint,
    write_artifact,
)
from apt_classifier.knowledge_store.pins import (
    PinnedNames,
    bundle_group_names,
    pin_for,
    pin_root,
)
from apt_classifier.text_group import DEFAULT_PARAMS as TEXT_PARAMS
from apt_classifier.text_group import TextGroupModel, prominence_log_prior

ALPHA = [
    "Alpha sent spearphishing emails with macro documents to finance staff.",
    "The macro documents dropped a loader that decrypted its payload in memory.",
    "The loader beaconed over https to a domain registered last month.",
]
BRAVO = [
    "Bravo dumped lsass memory with a renamed copy of procdump.",
    "They moved laterally over smb with stolen domain admin credentials.",
    "They staged archives on a file server before exfiltration with rclone.",
]
BRAVO_LATER = [
    "Bravo deployed a wiper on domain controllers through group policy.",
    "The wiper overwrote the master boot record of every reachable host.",
    "Group policy scheduled the wiper at midnight local time.",
]


def _kb() -> AttackKnowledgeBase:
    kb = AttackKnowledgeBase(
        version="test",
        techniques={"T1003": Technique(id="T1003", name="OS Credential Dumping")},
        groups={
            "G1": Group(id="G1", name="Alpha", technique_ids=["T1003"]),
            "G2": Group(id="G2", name="Bravo", technique_ids=["T1003"]),
            "G3": Group(id="G3", name="Charlie", technique_ids=["T1003"]),
        },
        software={
            "S0001": Software(
                id="S0001", name="Alphaload", software_type="malware", group_ids=["G1"]
            ),
            "S0002": Software(
                id="S0002", name="Mimikatz", software_type="tool", group_ids=["G2", "G3"]
            ),
        },
    )
    kb.recompute_statistics()
    return kb


def _link(key, actor, software_id, kind="attack_malware", hedge=None):
    return {"key": key, "actor": actor, "software_id": software_id, "kind": kind, "hedge": hedge}


def _payload() -> dict:
    """Posts: three labelled (G1 early, G2 early, G2 later), one copy of the early G2 post, and
    link posts. securelist and kaspersky_ics are one publisher (kaspersky)."""

    def post(source, day, statements=(), cited=0):
        return [source, day, cited, sorted(statement_keys(statements))]

    posts = {
        "a/1": post("talos", "2024-03-01", ALPHA),
        "b/1": post("eset", "2024-06-01", BRAVO),
        "b/1-copy": post("eset", "2024-06-02", BRAVO),
        "b/2": post("securelist", "2025-02-01", BRAVO_LATER),
        "l/1": post("talos", "2024-05-01"),
        "l/2": post("securelist", "2024-07-01"),
        "l/3": post("kaspersky_ics", "2024-08-01"),
        "l/4": post("securelist", "2024-08-02"),
        "l/5": post("eset", "2024-09-02"),
        "l/6": post("unit42", "2024-09-01", cited=1),
        "n/1": post("unit42", "2024-02-01"),
        "n/2": post("unit42", "2024-09-01"),
    }
    text = [
        {"key": key, "group": group, "text": line.replace("Alpha", "the adversary")}
        for key, group, lines in (
            ("a/1", "G1", ALPHA),
            ("b/1", "G2", BRAVO),
            ("b/2", "G2", BRAVO_LATER),
        )
        for line in lines
    ]
    links = [
        _link("l/1", "G3", "S0001"),  # talos
        _link("l/2", "G3", "S0001"),  # kaspersky: a second publisher
        _link("l/3", "G2", "S0001"),  # kaspersky
        _link("l/4", "G2", "S0001"),  # kaspersky again: still one publisher
        _link("l/5", "misp:foo", "S0001"),  # an actor outside ATT&CK: a user, never scored
        _link("l/5", "G1", "S0002", kind="attack_tool", hedge="possible"),  # never counts
        _link("l/6", "G1", "name:hookedge", kind="report"),
        _link("l/5", "G3", "misp:snake", kind="misp"),
    ]
    names = [
        {"software_id": "name:hookedge", "key": "n/1", "introduced": 0},
        {"software_id": "name:hookedge", "key": "n/2", "introduced": 1},
    ]
    software = {
        "S0001": ["Alphaload", "attack_malware"],
        "S0002": ["Mimikatz", "attack_tool"],
        "name:hookedge": ["HOOKEDGE", "report"],
        "misp:snake": ["SnakeSkin", "misp"],
    }
    return {
        "metadata": {"attack_version": "test", "introduced_share": 0.25, "pin": None},
        "posts": posts,
        "duplicates": {"b/1-copy": "b/1"},
        "text": text,
        "links": links,
        "names": names,
        "software": software,
        "masking_names": ["hookedge"],
    }


def _members(tmp_path, payload=None, pin=None) -> StoreMembers:
    path = members_path(SimpleNamespace(knowledge_store_dir=str(tmp_path)), pin)
    write_artifact(path, payload or _payload())
    loaded = StoreMembers.load(path, _kb())
    assert loaded is not None
    return loaded


def test_statement_keys_ignore_names_and_short_statements():
    named = statement_key("Bravo dumped lsass memory with a renamed copy of procdump.")
    masked = statement_key("The adversary dumped lsass memory with a renamed copy of procdump.")
    assert named is not None and named == masked
    assert statement_key("Bravo dumped lsass memory with a renamed copy of rclone.") != named
    assert statement_key("APT28 used X-Agent.") is None  # fewer than four words remain


def test_a_case_hides_its_own_post_its_copies_and_posts_sharing_its_statements(tmp_path):
    members = _members(tmp_path)
    assert members.hidden_for(keys=["b/1-copy"]) == {"b/1-copy", "b/1"}
    assert members.hidden_for(keys=["b/1"]) == {"b/1", "b/1-copy"}
    # the same statements under other names (a vendor's write-up of another vendor's report)
    renamed = [line.replace("Bravo", "Sandstorm") for line in BRAVO]
    assert members.hidden_for(statements=renamed) == {"b/1", "b/1-copy"}
    assert members.hidden_for(statements=renamed[:2]) == frozenset()  # too few to compare
    assert members.cited_posts() == {"l/6"}


def test_the_store_text_reads_only_posts_available_before_the_case(tmp_path):
    members = _members(tmp_path)
    wiper = "A wiper overwrote the master boot record through group policy."
    assert members.text_scores(Case(as_of="2025-01-15"), wiper) == {}
    assert (
        max(members.text_scores(Case(as_of="2025-02-02"), wiper).items(), key=lambda kv: kv[1])[0]
        == "G2"
    )
    assert members.text_scores(Case(as_of="2025-02-01"), wiper) == {}  # strictly before
    # the slice equals a fresh fit on the statements it keeps
    case = Case(as_of="2025-03-01", hidden=frozenset({"b/1"}))
    narrative = "Macro documents dropped a loader; the wiper ran through group policy on smb."
    kb = _kb()
    groups = tuple(sorted(kb.groups))
    kept = [
        {"text": r["text"], "source": "store", "owner": r["key"], "groups": [r["group"]]}
        for r in _payload()["text"]
        if r["key"] != "b/1"
    ]
    fresh = TextGroupModel(kept, groups, prominence_log_prior(kb, groups), params=TEXT_PARAMS)
    sliced = members.text_scores(case, narrative)
    assert set(sliced) == {"G1", "G2"} and sliced.keys() == fresh.scores(narrative).keys()
    for group, value in fresh.scores(narrative).items():
        assert sliced[group] == pytest.approx(value, abs=1e-6)
    # a removed actor has no document
    assert "G2" not in members.text_scores(Case(removed_actors=frozenset({"G2"})), wiper)


def test_the_store_software_scores_links_attack_lacks_and_names_on_two_publishers(tmp_path):
    members = _members(tmp_path)
    scores, naming, signals = members.software_scores(Case(), ["S0001"], "", _kb())
    assert set(scores) == {"G2", "G3"}  # G1 is ATT&CK's own user: no store score
    assert naming == {"G3"}  # talos + kaspersky; G2 only kaspersky (securelist, kaspersky_ics)
    assert signals["G3"] == ("S0001:Alphaload",)
    # before the second publisher's post: ranked, not naming
    early_scores, early_naming, _ = members.software_scores(
        Case(as_of="2024-06-01"), ["S0001"], "", _kb()
    )
    assert set(early_scores) == {"G3"} and early_naming == frozenset()
    # a removed actor neither scores nor counts as a user
    removed, removed_naming, _ = members.software_scores(
        Case(removed_actors=frozenset({"G3"})), ["S0001"], "", _kb()
    )
    assert "G3" not in removed and "G3" not in removed_naming
    # a "possible" link never scores; a hidden post's links are gone
    assert members.software_scores(Case(), ["S0002"], "", _kb())[0] == {}
    hidden, _, _ = members.software_scores(
        Case(hidden=frozenset({"l/1", "l/2"})), ["S0001"], "", _kb()
    )
    assert set(hidden) == {"G2"}


def test_rarity_counts_every_store_user_and_caps_tools(tmp_path):
    members = _members(tmp_path)
    scores, _, _ = members.software_scores(Case(), ["S0001", "S0002"], "", _kb())
    # S0001 (malware): ATT&CK users {G1}, store users {G2, G3, misp:foo}: log(4/5) + 1;
    # S0002 (tool): no store user, ATT&CK users {G2, G3}: log(4/3) + 1, under the cap
    import math

    s1, s2 = math.log(4 / 5) + 1, min(math.log(4 / 3) + 1, 1.5)
    assert scores["G3"] == pytest.approx(s1 / (s1 + s2))
    capped, _, _ = members.software_scores(
        Case(), ["S0001", "S0002"], "", _kb(), tool_weight_cap=0.1
    )
    assert capped["G3"] == pytest.approx(s1 / (s1 + 0.1))


def test_names_posts_introduce_count_only_once_the_share_rule_holds(tmp_path):
    members = _members(tmp_path)
    text = "The actor deployed HOOKEDGE and SnakeSkin; hookedge is lowercase here."
    assert members.software_names_in(Case(), text) == {"name:hookedge", "misp:snake"}
    # before the post that introduces it, HOOKEDGE is only a word one post mentioned
    assert members.software_names_in(Case(as_of="2024-08-01"), text) == {"misp:snake"}
    scores, naming, _ = members.software_scores(Case(), [], text, _kb())
    assert set(scores) == {"G1", "G3"} and naming == frozenset()
    late, _, _ = members.software_scores(Case(as_of="2024-08-01"), [], text, _kb())
    assert set(late) == set()  # SnakeSkin's link post (l/5) is from 2024-09-02


def test_the_artifact_fails_closed(tmp_path):
    settings = SimpleNamespace(
        store_text_weight=0.0, store_software_weight=0.0, knowledge_store_dir=str(tmp_path)
    )
    assert load_store_members_for(settings, _kb()) == (None, "disabled (weight 0)")
    settings.store_text_weight = 1.0
    members, status = load_store_members_for(settings, _kb())
    assert members is None and status.startswith("artifact not found")
    _members(tmp_path)
    members, status = load_store_members_for(settings, _kb())
    assert status == "active" and set(members) == {None}
    # pinned: every pin's artifact must be there
    settings.knowledge_store_pinned = True
    assert load_store_members_for(settings, _kb())[1].startswith("artifact not found")
    for pin in ("before_2025", "before_2026_03"):
        _members(pin_root(tmp_path, pin), pin=None)
    members, status = load_store_members_for(settings, _kb())
    assert status == "active" and set(members) == {None, "before_2025", "before_2026_03"}
    # tampered content or another ATT&CK release
    path = members_path(settings)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["links"] = []
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    assert load_store_members_for(settings, _kb())[1].startswith("invalid artifact")
    write_artifact(path, {**_payload(), "metadata": {"attack_version": "19.2"}})
    assert load_store_members_for(settings, _kb())[1].startswith("incompatible artifact")


def test_the_bound_fingerprint_follows_the_artifacts_and_the_weights(tmp_path):
    members = {None: _members(tmp_path)}
    base = store_bound_fingerprint("abc", members, {"store_text": 1.0})
    assert base != store_bound_fingerprint("abc", members, {"store_text": 2.0})
    assert base != store_bound_fingerprint("abd", members, {"store_text": 1.0})
    other = _payload()
    other["links"] = other["links"][:1]
    changed = {None: _members(tmp_path / "other", other)}
    assert base != store_bound_fingerprint("abc", changed, {"store_text": 1.0})


def test_a_dated_case_reads_its_pin_and_an_undated_one_the_store(tmp_path):
    main = _members(tmp_path)
    early = _members(tmp_path / "early")
    state = StoreState({None: main, "before_2025": early})
    case = state.set_case(as_of=date(2025, 6, 1), keys=["b/1"])
    assert state.pin == "before_2025" and state.current is early
    assert case.as_of == "2025-06-01" and case.hidden == {"b/1", "b/1-copy"}
    state.set_case(as_of="2026-03-02")  # its pin is not loaded: the store's own artifact
    assert state.pin is None and state.current is main
    state.set_case(as_of="2025-06-01", pinned=False)
    assert state.pin is None
    state.set_case(as_of="2025-06-01", removed_actors=["G2"], hidden=["x"])
    assert state.case.removed_actors == {"G2"} and "x" in state.case.hidden
    state.clear()
    assert state.case == Case() and state.current is main


def test_pins_follow_the_report_date():
    assert pin_for("2026-02-28") == "before_2025"
    assert pin_for(date(2025, 1, 1)) == "before_2025"
    assert pin_for("2026-03-01") == "before_2026_03"


def test_an_older_release_gives_the_groups_its_names(tmp_path):
    def intrusion_set(group_id, name, aliases=(), **flags):
        return {
            "type": "intrusion-set",
            "name": name,
            "aliases": list(aliases),
            "external_references": [{"source_name": "mitre-attack", "external_id": group_id}],
            **flags,
        }

    bundle = tmp_path / "old.json"
    bundle.write_text(
        json.dumps(
            {
                "objects": [
                    intrusion_set("G1", "Old Alpha", ["Old Alpha", "Alpha", "AlphaTeam"]),
                    intrusion_set("G2", "Bravo", revoked=True),
                    intrusion_set("G3", "Charlie", x_mitre_deprecated=True),
                    {"type": "intrusion-set", "name": "No id", "external_references": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    older = bundle_group_names(bundle)
    assert older == {"G1": ("Old Alpha", "Alpha", "AlphaTeam")}
    kb = _kb()
    pinned = PinnedNames(kb, older)
    assert pinned.groups["G1"].name == "Alpha"
    assert pinned.groups["G1"].aliases == ("Old Alpha", "AlphaTeam")
    assert pinned.groups["G2"].aliases == ()  # later or revoked: the current name only
    assert pinned.software is kb.software and pinned.version == "test"


def test_attack_citations_become_store_keys(tmp_path):
    from apt_classifier.knowledge_store.listing import normalise_url

    bundle = tmp_path / "bundle.json"
    url = "https://www.example.com/blog/alpha-report/?utm_source=x"

    def ref(stix_id, kind, external_id):
        return {
            "type": kind,
            "id": stix_id,
            "external_references": [{"source_name": "mitre-attack", "external_id": external_id}],
        }

    bundle.write_text(
        json.dumps(
            {
                "objects": [
                    ref("intrusion-set--1", "intrusion-set", "G1"),
                    ref("attack-pattern--1", "attack-pattern", "T1003"),
                    {
                        "type": "relationship",
                        "relationship_type": "uses",
                        "source_ref": "intrusion-set--1",
                        "target_ref": "attack-pattern--1",
                        "external_references": [{"source_name": "Report", "url": url}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert citation_keys(bundle) == {("G1", "T1003"): frozenset({normalise_url(url)})}


def test_the_store_software_supports_a_name_only_through_two_publishers():
    families = {STORE_SOFTWARE_NAME: "software", "software": "software"}
    ranks = {STORE_SOFTWARE_NAME: 1}
    limits = naming_limits(SimpleNamespace(store_naming_groups=frozenset({"G1"})))
    assert limits == {STORE_SOFTWARE_NAME: frozenset({"G1"})}
    assert supporting_families(ranks, families, group_id="G1", limits=limits) == {"software"}
    assert supporting_families(ranks, families, group_id="G2", limits=limits) == set()
    assert supporting_families(ranks, families, group_id="G1") == set()  # no limits: never
    assert naming_limits(SimpleNamespace()) == {}
    # the store text is never independent evidence
    assert supporting_families({"store_text": 1}, {"store_text": "learned_profile"}) == set()
