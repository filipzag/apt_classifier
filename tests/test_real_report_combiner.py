"""Real-report combiner lists (prereg attribution_real_report_combiner_2026-10-22): development
report sets hide what ATT&CK cites or copies from them, the advisory labelling rule, and the
real-weight cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Campaign, Group, Technique
from apt_classifier.combiner import FEATURE_NAMES, NULL_FEATURE_NAMES, CombinerList
from apt_classifier.vectorstore import VectorHit, normalize_procedure_text
from tests.test_elastic_benchmark import kb  # noqa: F401  (fixture)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_development_rows_hide_cited_texts_statements_and_campaign(kb, tmp_path):  # noqa: F811
    from scripts.attribution_benchmark import load_report_folds

    cited = "APT32 has used PowerShell to run its loader.(Citation: Vendor 2020)"
    statement = "The adversary used PowerShell to run a loader on the host."
    path = _write(
        tmp_path / "dev.jsonl",
        [
            {
                "report_id": "campaign-C1",
                "group_id": "G0050",
                "campaign_id": "C1",
                "statements": [{"text": statement}],
                "hidden_texts": [cited],
            },
            {"report_id": "cti", "group_id": "G0050", "statements": [{"text": statement}]},
        ],
    )
    dev, test = load_report_folds(path, max_observations=None, kb=kb)
    assert dev.hidden_campaign == "C1"
    assert set(dev.hidden_raw) == {cited, statement}
    assert normalize_procedure_text("APT32 has used PowerShell to run its loader.") in (
        dev.hidden_texts
    )
    assert normalize_procedure_text(statement) in dev.hidden_texts
    # a test-set row hides nothing and is read as before
    assert test.hidden_texts == frozenset() and test.hidden_raw == () and not test.hidden_campaign
    with pytest.raises(ValueError):
        load_report_folds(path, max_observations=None)


def test_hidden_campaign_leaves_and_comes_back():
    from scripts.attribution_benchmark import hidden_campaign

    group = Group(id="G1", name="One", campaign_ids=["C1", "C2"])
    base = AttackKnowledgeBase(
        version="t",
        groups={"G1": group},
        techniques={"T1059": Technique(id="T1059", name="x")},
        campaigns={
            "C1": Campaign(id="C1", name="Op One", group_ids=["G1"]),
            "C2": Campaign(id="C2", name="Op Two", group_ids=["G1"]),
        },
    )
    with hidden_campaign(base, "C1") as hidden:
        assert hidden and "C1" not in base.campaigns and group.campaign_ids == ["C2"]
    assert set(base.campaigns) == {"C1", "C2"} and group.campaign_ids == ["C1", "C2"]
    with hidden_campaign(base, "") as hidden:
        assert not hidden


class _Store:
    def __init__(self, hits):
        self.hits = hits
        self.asked = []

    def query_techniques(self, texts, *, n_results=10, fields=None):
        self.asked.append(n_results)
        return [list(self.hits) for _ in texts]

    def query_group_behaviors(self, texts, *, n_results=20):
        return [list(self.hits) for _ in texts]


def _hit(text: str, group_id: str) -> VectorHit:
    return VectorHit(
        id=text, document=text, metadata={"group_id": group_id}, distance=0.1, similarity=0.9
    )


def test_fold_store_hides_development_texts_under_any_group():
    from scripts.attribution_benchmark import FoldStore

    copied = "the adversary dumped credentials from lsass memory"
    hidden = frozenset({normalize_procedure_text(copied)} | {f"other {i}" for i in range(100)})
    store = _Store([_hit(copied, "G2"), _hit("unrelated behaviour text here", "G2")])
    proxy = FoldStore(
        store,
        group_id="G1",
        held_out_techniques=frozenset(),
        held_out_texts=hidden,
        any_group_texts=hidden,
    )
    behaviours = proxy.query_group_behaviors(["q"])[0]
    assert [hit.document for hit in behaviours] == ["unrelated behaviour text here"]
    techniques = proxy.query_techniques(["q"], n_results=10)[0]
    assert [hit.document for hit in techniques] == ["unrelated behaviour text here"]
    # the refill is capped: a report hides hundreds of texts, a query meets a few
    assert store.asked == [50]


def test_advisory_label_needs_mentions_dominance_and_a_documented_group():
    from scripts.prepare_advisory_reports import group_patterns, label

    techniques = {f"T10{i:02d}": Technique(id=f"T10{i:02d}", name="x") for i in range(10)}
    base = AttackKnowledgeBase(
        version="t",
        groups={
            "G1017": Group(
                id="G1017",
                name="Volt Typhoon",
                aliases=["BRONZE SILHOUETTE"],
                technique_ids=list(techniques),
            ),
            "G0007": Group(id="G0007", name="APT28", technique_ids=list(techniques)),
            "G9999": Group(id="G9999", name="Thin Group", technique_ids=["T1000"]),
        },
        techniques=techniques,
    )
    patterns = group_patterns(base)
    assert (
        label(
            "Volt Typhoon did it. " * 8 + "BRONZE SILHOUETTE too. " * 2 + "APT28.", base, patterns
        )[0]
        == "G1017"
    )
    assert label("Volt Typhoon. " * 9, base, patterns)[0] is None  # too few mentions
    assert label("Volt Typhoon. " * 10 + "APT28. " * 4, base, patterns)[0] is None  # not dominant
    assert label("Thin Group. " * 20, base, patterns)[0] is None  # too few documented techniques
    assert label("volt typhoon. " * 20, base, patterns)[0] is None  # case-sensitive


def test_advisory_statements_stop_at_mitigations_and_drop_tags_and_hashes():
    from scripts.prepare_advisory_reports import statements

    text = (
        "Technical Details\n\n"
        "The actors used PowerShell [T1059.001] to download the second stage from a server.\n\n"
        "The file hash was 0123456789abcdef0123456789abcdef0123 on the affected host machine.\n\n"
        "Too short here.\n\n"
        "Mitigations\n\n"
        "Organizations should disable PowerShell where it is not needed by any user."
    )
    found = [item["text"] for item in statements(text)]
    assert found == ["The actors used PowerShell to download the second stage from a server."]


def test_hidden_procedures_by_citation_and_by_shared_run():
    from scripts.prepare_advisory_reports import hidden_procedures, runs, tokens

    report = "the actors copied the ntds dit file with ntdsutil and staged it in a temp folder"
    examples = [
        ("Group used X.(Citation: CISA AA24-038A)", set()),
        (
            "Group copied the ntds dit file with ntdsutil and staged it in a temp folder.",
            runs(
                tokens(
                    "Group copied the ntds dit file with ntdsutil and staged it in a temp folder."
                )
            ),
        ),
        ("Group used Y.(Citation: Other)", set()),
    ]
    found, cited, copied = hidden_procedures(examples, ["CISA AA24-038A"], runs(tokens(report)))
    assert cited == 1 and copied == 1 and len(found) == 2
    assert all("Other" not in text for text in found)


def test_real_condition_parse():
    from scripts.fit_attribution_combiner import RealCondition

    assert RealCondition.parse("names:30") == RealCondition("names", 30)
    assert RealCondition.parse("strict:all") == RealCondition("strict", None)
    assert RealCondition.parse("strict:all").label == "strict:all"


def _records(rng, count, *, signal, source, groups, condition="names:30"):
    out = []
    index = FEATURE_NAMES.index(signal)
    for i in range(count):
        features = rng.normal(size=(5, len(FEATURE_NAMES)))
        truth = int(rng.integers(0, 5))
        features[truth, index] += 2.5
        group_ids = [f"X{i}_{j}" for j in range(5)]
        group_ids[truth] = groups[i % len(groups)]
        lst = CombinerList(
            tuple(group_ids),
            features,
            {name: 0.5 for name in NULL_FEATURE_NAMES[1:]},
            group_ids[truth],
        )
        record = {"group_id": group_ids[truth], "variant": 0, **lst.to_dict()}
        if source:
            record.update({"source": source, "condition": condition})
        out.append(record)
    return out


def test_cross_validation_weighs_real_lists_only_within_the_fold_margin():
    from scripts.fit_attribution_combiner import cross_validate

    rng = np.random.default_rng(11)
    groups = [f"G{i}" for i in range(10)]
    folds = _records(rng, 400, signal="rrf_rel", source=None, groups=groups)
    # real lists decided by another feature: weighing them helps the real lists, costs the folds
    real = _records(rng, 120, signal=FEATURE_NAMES[-1], source="real:vendor", groups=groups)

    def run(margin: float) -> dict:
        return cross_validate(
            folds + real,
            folds=5,
            l2_grid=[1.0],
            null_option=False,
            seed=97,
            log=lambda _: None,
            real_weights=[0, 3],
            fold_top1_margin=margin,
        )

    lenient = run(1.0)
    grid = lenient["grid"]
    assert grid["w3_l21.0"]["combiner"]["real"]["mrr"] > grid["w0_l21.0"]["combiner"]["real"]["mrr"]
    assert (
        grid["w3_l21.0"]["combiner"]["clean"]["top1"]
        < grid["w0_l21.0"]["combiner"]["clean"]["top1"]
    )
    assert lenient["chosen_real_weight"] == 3
    # the registered margin keeps the folds: the weight that costs them is not eligible
    strict = run(0.01)
    assert strict["chosen_real_weight"] == 0 and strict["selection"]["eligible"] == ["w0_l21.0"]
    # without real lists the old rule (best mixed fold MRR) and weight 0
    plain = cross_validate(
        folds, folds=5, l2_grid=[1.0], null_option=False, seed=97, log=lambda _: None
    )
    assert plain["chosen_real_weight"] == 0 and "selection" not in plain


def test_fit_script_refuses_test_sets_as_training_lists(tmp_path):
    from scripts.fit_attribution_combiner import main

    common = [
        "--output",
        str(tmp_path / "c.json"),
        "--report",
        str(tmp_path / "r.json"),
        "--lists-cache",
        str(tmp_path / "l.jsonl.gz"),
        "--real-cache",
        str(tmp_path / "real.jsonl.gz"),
    ]
    with pytest.raises(SystemExit):
        main([*common, "--real-reports", "cti_hal=data/benchmarks/cti_hal/reports.jsonl"])
    with pytest.raises(SystemExit):
        main([*common, "--real-reports", "mine=data/benchmarks/elastic/reports.jsonl"])
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "c.json"), "--real-reports", "vendor=x.jsonl"])


def test_list_generation_resumes_from_its_checkpoint_without_regenerating():
    from types import SimpleNamespace

    from scripts.fit_attribution_combiner import generate_lists, generate_unknown_lists

    class Bench:
        def __init__(self, folds):
            self.folds = folds

        def build_folds(self, knowledge_base, **_):
            if self.folds is None:
                raise AssertionError("a resumed seed was generated again")
            return self.folds

    engine = SimpleNamespace(kb=None, store="store", lexical_index="lexical")
    saved = {
        "records": [{"seed": 11, "group_id": "G1", "variant": 0}],
        "pool": {"T1": [0.5]},
        "stats": {"seeds": {"11": {"folds": 1}}, "pool_techniques": 1},
    }
    records, stats = generate_lists(
        engine,
        Bench(None),
        seeds=[11],
        variants=5,
        top_n=20,
        fold_args={},
        limit=None,
        log=lambda _: None,
        resume=saved,
    )
    assert records == saved["records"] and stats["seeds"] == {"11": {"folds": 1}}
    assert engine.store == "store" and engine.lexical_index == "lexical"
    # the removed-actor arm skips the folds its checkpoint already processed
    folds = [SimpleNamespace(group_id="G1"), SimpleNamespace(group_id="G2")]
    unknown = generate_unknown_lists(
        None,
        None,
        None,
        Bench(folds),
        seed=11,
        top_n=20,
        fold_args={},
        limit=None,
        log=lambda _: None,
        resume={"records": [{"group_id": "G1", "variant": -1}], "processed": 2},
    )
    assert unknown == [{"group_id": "G1", "variant": -1}]
