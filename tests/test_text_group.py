"""Free-text group model (apt_classifier/text_group.py)."""

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Campaign, Group, Technique
from apt_classifier.text_group import (
    TEXT_GROUP_NAME,
    TextGroupClassifier,
    TextGroupModel,
    build_corpus,
    fit_text_group,
    load_text_group_for,
    match_key,
    normalize_text,
    text_group_votes,
)


def _kb() -> AttackKnowledgeBase:
    techniques = {f"T{i}": Technique(id=f"T{i}", name=f"Technique {i}") for i in range(1, 6)}
    groups = {
        "G1": Group(
            id="G1",
            name="Alpha",
            aliases=["Alpha", "Red Fox"],
            description="Alpha is an espionage group. Red Fox targets ministries in Lisbon.",
            technique_ids=["T1", "T2"],
            software_ids=["S1"],
            technique_behaviors={
                "T1": ["Alpha has used spearphishing attachments with macro documents."],
                "T2": ["Red Fox staged archives on a compromised sharepoint server."],
            },
        ),
        "G2": Group(
            id="G2",
            name="Bravo",
            description="Bravo is a financially motivated group targeting point of sale systems.",
            technique_ids=["T3", "T4"],
            technique_behaviors={
                "T3": ["Bravo has scraped payment card data from point of sale memory."],
                "T4": ["Bravo used a custom backdoor over raw sockets for command and control."],
            },
        ),
        "G3": Group(
            id="G3",
            name="Charlie",
            description="Charlie is a destructive group attacking power grid operators.",
            technique_ids=["T5"],
            technique_behaviors={
                "T5": ["Charlie deployed wiper malware against power grid workstations."]
            },
        ),
    }
    campaigns = {"C1": Campaign(id="C1", name="Operation Lighthouse", group_ids=["G1"])}
    kb = AttackKnowledgeBase(
        version="t1", techniques=techniques, groups=groups, campaigns=campaigns
    )
    kb.recompute_statistics()
    return kb


def _bundle() -> dict:
    def obj(kind, stix, ext):
        return {
            "type": kind,
            "id": stix,
            "external_references": [{"source_name": "mitre-attack", "external_id": ext}],
        }

    def rel(kind, src, dst, description=""):
        return {
            "type": "relationship",
            "id": f"relationship--{src}-{dst}",
            "relationship_type": kind,
            "source_ref": src,
            "target_ref": dst,
            "description": description,
        }

    return {
        "objects": [
            obj("attack-pattern", "attack-pattern--1", "T1"),
            obj("malware", "malware--1", "S1"),
            obj("malware", "malware--2", "S2"),
            obj("campaign", "campaign--1", "C1"),
            obj("intrusion-set", "intrusion-set--1", "G1"),
            rel(
                "uses",
                "malware--1",
                "attack-pattern--1",
                "FOXGLOVE implant beacons to dropbox folders every hour (Citation: Vendor 2020).",
            ),
            rel("uses", "malware--2", "attack-pattern--1", "LONEWOLF loader reads registry keys."),
            rel(
                "uses",
                "campaign--1",
                "attack-pattern--1",
                "During Operation Lighthouse, Alpha used lighthouse themed lures.",
            ),
            rel("attributed-to", "campaign--1", "intrusion-set--1"),
        ]
    }


def test_corpus_anonymises_and_attributes():
    records = build_corpus(_kb(), _bundle())
    texts = " ".join(r["text"] for r in records)
    assert "Alpha" not in texts and "Red Fox" not in texts and "Lighthouse," not in texts
    assert "Citation" not in texts
    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)
    software = {r["owner"]: r for r in by_source["software"]}
    assert software["S1"]["groups"] == ["G1"] and software["S2"]["groups"] == []
    (campaign,) = by_source["campaign"]
    assert campaign["groups"] == ["G1"] and campaign["text"].startswith("During the operation")
    assert {r["owner"] for r in by_source["group_desc"]} == {"G1", "G2", "G3"}
    assert build_corpus(_kb(), None) and all(
        r["source"] in ("group", "group_desc") for r in build_corpus(_kb(), None)
    )


def _model(**params):
    return fit_text_group(_kb(), _bundle(), {"min_df": 1, **params})


def test_scores_rank_the_matching_group_and_abstain_without_overlap():
    model = _model()
    scores = model.scores("The intruders scraped payment card data from point of sale memory.")
    assert max(scores, key=scores.get) == "G2"
    assert all(v > 0 for v in scores.values()) and abs(sum(scores.values()) - 1) < 1e-9
    # malware names are words like any other: the S1 behaviour points to the group using S1
    named = model.scores("a FOXGLOVE implant beaconed to dropbox")
    assert max(named, key=named.get) == "G1"
    assert model.scores("") == {} and model.scores("zzzz qqqq") == {}
    # the placeholders carry no signal
    assert model.scores("the adversary the operation") == {}


def test_without_mirrors_the_fold_store():
    kb = _kb()
    model = _model()
    held = "Bravo has scraped payment card data from point of sale memory."
    hidden = model.without(kb, group_id="G2", techniques={"T3"}, texts={normalize_text(held)})
    assert not any(r["owner"] == "G2" and r.get("technique_id") == "T3" for r in hidden.records)
    assert any(r["owner"] == "G2" and r.get("technique_id") == "T4" for r in hidden.records)
    assert hidden.metadata["refit"]["dropped_records"] == 1
    # a corpus text that copies a held-out sentence is dropped even under another owner, and the
    # campaign-name placeholder does not defeat the match (fold texts keep the campaign name and
    # arrive normalised, as the benchmarks' held_out_texts do)
    copied = model.without(
        kb,
        texts={
            normalize_text(
                "During Operation Lighthouse, the adversary used lighthouse themed lures."
            )
        },
    )
    assert not any(r["source"] == "campaign" for r in copied.records)
    assert match_key("During the operation, the adversary used X.") == "during used x"
    # unknown-actor arm: the group and its campaign go, the software it used stays unlabelled
    reduced = _kb()
    del reduced.groups["G1"]
    reduced.recompute_statistics()
    removed = model.without(reduced, group_id="G1", remove_group=True)
    assert removed.group_ids == ("G2", "G3")
    assert not any(r["owner"] == "G1" for r in removed.records)
    assert all("G1" not in r["groups"] for r in removed.records)
    assert {r["owner"]: r["groups"] for r in removed.records if r["source"] == "software"}[
        "S1"
    ] == []
    assert "G1" not in removed.scores("FOXGLOVE implant beacons to dropbox folders")
    # cached per fold, vocabulary shared with the base model
    assert model.without(reduced, group_id="G1", remove_group=True) is removed
    assert removed._vectorizer is model._vectorizer


def test_save_load_round_trip_and_fail_closed(tmp_path):
    kb = _kb()
    model = _model()
    path = model.save(tmp_path / "text.json.gz")
    loaded = TextGroupModel.load(path, kb)
    assert loaded.fingerprint == model.fingerprint
    text = "wiper malware against power grid workstations"
    assert loaded.scores(text) == pytest.approx(model.scores(text))
    settings = SimpleNamespace(text_group_enabled=True, text_group_weight=1.0, text_group_path=path)
    assert load_text_group_for(settings, kb)[1] == "active"
    assert load_text_group_for(
        SimpleNamespace(**{**vars(settings), "text_group_weight": 0.0}), kb
    ) == (
        None,
        "disabled (weight 0)",
    )
    assert not text_group_votes(SimpleNamespace(text_group_enabled=False, text_group_weight=1.0))
    other = _kb()
    other.version = "t2"
    assert "incompatible" in load_text_group_for(settings, other)[1]
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["records"][0]["text"] += " tampered"
    bad = tmp_path / "bad.json.gz"
    with gzip.open(bad, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    assert (
        "fingerprint"
        in load_text_group_for(SimpleNamespace(**{**vars(settings), "text_group_path": bad}), kb)[1]
    )
    missing = SimpleNamespace(**{**vars(settings), "text_group_path": tmp_path / "none.json.gz"})
    assert "not found" in load_text_group_for(missing, kb)[1]


def test_classifier_reads_the_narrative_and_respects_eligibility():
    from apt_classifier.classifiers.base import ClassifierContext

    kb = _kb()

    def ctx(narrative, eligible=None):
        c = ClassifierContext(
            kb=kb,
            observed_techniques={},
            technique_observations={},
            negative_techniques=set(),
            semantic_profile_scores={},
            group_text_scores={},
            software_ids=set(),
            narrative=narrative,
        )
        c.eligible_group_ids = eligible
        return c

    classifier = TextGroupClassifier(_model())
    assert classifier.name == TEXT_GROUP_NAME
    assert not classifier.classify(ctx("")).applicable
    assert not TextGroupClassifier(None).classify(ctx("payment card")).applicable
    result = classifier.classify(ctx("payment card data from point of sale memory", {"G1", "G3"}))
    assert {e.group_id for e in result.entries} <= {"G1", "G3"}


def _rare_kb():
    kb = _kb()
    # "zeppelin" occurs in one held-out G2 sentence and one kept G2 sentence: with min_df=2 an
    # exact refit without the held-out sentence drops the term; the hidden one must not keep it
    kb.groups["G2"].technique_behaviors["T3"].append("Bravo hid zeppelin beacons in card dumps.")
    kb.groups["G2"].technique_behaviors["T4"].append("Bravo renamed zeppelin tools after use.")
    kb.recompute_statistics()
    return kb


def test_without_equals_a_fresh_fit_with_min_df():
    kb = _rare_kb()
    model = fit_text_group(kb, _bundle(), {"min_df": 2})
    held = normalize_text("the adversary hid zeppelin beacons in card dumps.")
    fast = model.without(kb, group_id="G2", techniques={"T3"}, texts={held})
    fresh = TextGroupModel(fast.records, fast.group_ids, fast.log_prior, params=model.params)
    for text in ("zeppelin tools renamed", "payment card scraping", "zeppelin beacons"):
        assert fast.scores(text) == pytest.approx(fresh.scores(text), abs=1e-9)
    # the rare word is gone with its second occurrence: it no longer points to G2
    assert "zeppelin" not in fresh._vectorizer.vocabulary_ or not fresh.scores("zeppelin")
    assert fast.scores("zeppelin") == pytest.approx(fresh.scores("zeppelin"), abs=1e-9)


def test_hiding_follows_owners_and_sentences():
    kb = _kb()
    kb.groups["G3"].technique_behaviors["T5"].append(
        "Charlie has scraped payment card data from point of sale memory."
    )
    kb.recompute_statistics()
    bundle = _bundle()
    bundle["objects"].append(
        {
            "type": "relationship",
            "id": "relationship--x",
            "relationship_type": "uses",
            "source_ref": "malware--2",
            "target_ref": "attack-pattern--1",
            "description": "LONEWOLF is loaded first. The adversary has scraped payment card data from "
            "point of sale memory.",
        }
    )
    model = fit_text_group(kb, bundle, {"min_df": 1})
    # a held-out G2 sentence as build_folds sends it: anonymised, then normalised
    held = normalize_text("the adversary has scraped payment card data from point of sale memory.")
    hidden = model.without(kb, group_id="G2", techniques=set(), texts={held})
    owners = [(r["owner"], r["source"]) for r in hidden.records if "payment card" in r["text"]]
    assert ("G2", "group") not in owners  # the truth's own copy goes
    assert ("G3", "group") in owners  # a competitor's identical sentence stays (as in the store)
    # the unattributed software description containing the sentence goes (sentence-level match)
    assert not any(r["owner"] == "S2" and "payment card" in r["text"] for r in hidden.records)


def test_corpus_replaces_names_before_a_hyphen():
    kb = _kb()
    kb.groups["G1"].technique_behaviors["T1"].append("Red Fox-controlled servers relayed traffic.")
    kb.recompute_statistics()
    texts = [r["text"] for r in build_corpus(kb, None)]
    assert any(t.startswith("the adversary-controlled") for t in texts)
    assert not any("Red Fox" in t for t in texts)


def test_fail_closed_on_corrupt_and_unversioned_artifacts(tmp_path):
    kb = _kb()
    good = _model().save(tmp_path / "good.json.gz")
    settings = SimpleNamespace(text_group_enabled=True, text_group_weight=1.0, text_group_path=good)

    def status(path):
        return load_text_group_for(
            SimpleNamespace(**{**vars(settings), "text_group_path": path}), kb
        )

    raw = bytearray(good.read_bytes())
    raw[200:260] = bytes(60)
    (tmp_path / "corrupt.json.gz").write_bytes(bytes(raw))
    model, message = status(tmp_path / "corrupt.json.gz")
    assert model is None and message.startswith("invalid artifact")
    (tmp_path / "list.json.gz").write_bytes(gzip.compress(b"[1, 2, 3]"))
    assert status(tmp_path / "list.json.gz")[0] is None
    with gzip.open(good, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["metadata"]["attack_version"] = ""
    unversioned = tmp_path / "unversioned.json.gz"
    with gzip.open(unversioned, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    model, message = status(unversioned)
    assert model is None  # the version is in the fingerprint: emptying it is tampering


def test_text_group_never_counts_as_an_independent_family():
    from apt_classifier.attribution import NOT_INDEPENDENT_EVIDENCE, _evidence_family_count
    from apt_classifier.fusion import FusionEntry, FusionResult

    assert TEXT_GROUP_NAME in NOT_INDEPENDENT_EVIDENCE
    entry = FusionEntry(
        group_id="G1",
        score=1.0,
        relative_score=1.0,
        rank=1,
        classifier_ranks={"idf_tfidf_cosine": 1, TEXT_GROUP_NAME: 1},
        rank_min=1,
        rank_max=1,
        rank_std=0.0,
        contested=False,
    )
    fusion = FusionResult(entries=(entry,), active_classifiers=(), correlations={}, k=60)
    families = {"idf_tfidf_cosine": "technique_overlap", TEXT_GROUP_NAME: "learned_profile"}
    assert _evidence_family_count(fusion, families) == 1
