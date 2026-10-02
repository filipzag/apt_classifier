"""Tests for ``scripts/compare_runs.py`` on artifacts built by the real benchmark code.

Mapping artifacts come from ``scripts/benchmark_mapping.run`` over a fake vector store and a
stand-in acceptance model, so re-thresholding is checked against the runtime gate itself: a run
re-thresholded offline must equal the same run made at that threshold. No Chroma, model or LLM.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Technique
from apt_classifier.evaluation import TechniqueEvaluationCase
from apt_classifier.retrieve import LexicalTechniqueIndex
from apt_classifier.vectorstore import VectorHit
from scripts.benchmark_mapping import case_set_hash, run
from scripts.compare_runs import (
    NotComparableError,
    compare_mapping_identity,
    emulate_case,
    main,
    paired_bootstrap,
    pr_curve,
    rethreshold_artifact,
    summarize_ledger,
)

# made-up keywords so no regex cue or structured signal fires; similarity per keyword
KEYWORDS = {
    "zephyr": ("T1059.001", 0.9),
    "quokka": ("T1105", 0.62),
    "marmot": ("T1003.001", 0.55),
}


def _kb() -> AttackKnowledgeBase:
    return AttackKnowledgeBase(
        version="t",
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1105": Technique(id="T1105", name="Ingress Tool Transfer"),
            "T1003.001": Technique(
                id="T1003.001", name="LSASS Memory", is_subtechnique=True, parent_id="T1003"
            ),
        },
    )


class _Store:
    def query_techniques(self, texts, *, n_results, **kwargs):
        out = []
        for text in texts:
            hits = [
                VectorHit(word, word, {"technique_id": tid, "field": "description"}, 1 - sim, sim)
                for word, (tid, sim) in KEYWORDS.items()
                if word in text.lower()
            ]
            out.append(sorted(hits, key=lambda hit: -hit.similarity))
        return out

    def query_group_profile(self, texts, *, document_type, n_results):
        return []

    def query_group_behaviors(self, texts, *, n_results):
        return [[] for _ in texts]


class _Model:
    """Stand-in acceptance model: p rises with the dense score (0.05 without a dense hit)."""

    metadata: dict = {}

    def __init__(self, threshold: float):
        self.threshold = threshold

    def probability(self, features):
        return 0.05 + 0.9 * float(features.get("dense_score", 0.0))


SETTINGS = {
    "similarity_threshold": 0.45,
    "retrieval_top_k": 30,
    "technique_candidate_pool": 12,
    "technique_parent_policy": "ancestor_compatible",
    "technique_acceptance_threshold": 0.75,
    "technique_relative_margin": 0.08,
    "technique_mapping_preset": "balanced",
    "embedding_model": "fake",
    "embedding_model_revision": "rev",
}


def _case(dataset: str, index: int, text: str, labels: list[str]) -> TechniqueEvaluationCase:
    return TechniqueEvaluationCase.from_dict(
        {
            "case_id": f"{dataset}-test-{index:06d}",
            "text": text,
            "expected_technique_ids": labels,
            "source": f"tumeteor/mitre-ttp-mapping:{dataset}",
        }
    )


CASES = [
    _case("tram", 1, "The operator staged zephyr payloads on the server", ["T1059.001"]),
    _case(
        "tram", 2, "The operator moved quokka marmot archives between hosts", ["T1105", "T1003.001"]
    ),
    _case("tram", 3, "The operator ran quokka transfers quietly overnight", ["T1059.001"]),
    # explicit cue with lexical support: accepted at every threshold
    _case("expert", 4, "PowerShell ran an encoded command on the host", ["T1059.001"]),
    # benign context: never accepted on similarity alone
    _case("expert", 5, "Routine maintenance of zephyr systems happened today", []),
    _case("expert", 6, "The operator copied marmot data to a share", ["T1105"]),
]


def _artifact(threshold: float | None, *, model: bool = True) -> dict:
    from types import SimpleNamespace

    kb = _kb()
    block = run(
        CASES,
        mode="hybrid",
        settings=SimpleNamespace(**SETTINGS),
        kb=kb,
        store=_Store(),
        lexical_index=LexicalTechniqueIndex.from_kb(kb),
        acceptance_model=_Model(threshold) if model else None,
        bootstrap_samples=20,
        max_candidate_rows=40,
    )
    artifact = {
        "case_set_hash": case_set_hash(CASES),
        "case_diagnostics": {"seed": 7},
        "bootstrap_samples": 20,
        "results": {"hybrid": block},
    }
    return json.loads(json.dumps(artifact))  # as read back from disk


def _accepted(artifact: dict) -> dict[str, list[str]]:
    return {
        case["case_id"]: case["accepted_technique_ids"]
        for case in artifact["results"]["hybrid"]["cases"]
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("target", [0.5, 0.7, 0.9])
def test_rethreshold_equals_a_run_at_that_threshold(target):
    base, expected = _artifact(0.6), _artifact(target)
    assert _accepted(base) != _accepted(expected)  # the threshold matters on these cases
    output, checks = rethreshold_artifact(base, target, kb=_kb())
    assert checks["hybrid"]["mismatches"] == 0
    assert _accepted(output) == _accepted(expected)
    left, right = output["results"]["hybrid"], expected["results"]["hybrid"]
    for a, b in zip(left["cases"], right["cases"], strict=True):
        assert [row["state"] for row in a["candidates"]] == [
            row["state"] for row in b["candidates"]
        ]
        assert a["expected_accepted"] == b["expected_accepted"]
    assert left["pooled"]["metrics"] == right["pooled"]["metrics"]
    assert (
        left["pooled"]["bootstrap_confidence_intervals"]
        == right["pooled"]["bootstrap_confidence_intervals"]
    )
    for dataset in ("tram", "expert"):
        assert left["datasets"][dataset]["metrics"] == right["datasets"][dataset]["metrics"]
        assert left["datasets"][dataset]["summary"] == right["datasets"][dataset]["summary"]
    assert left["fingerprint_inputs"]["acceptance_model"]["threshold"] == target
    assert left["rethresholded"]["original_threshold"] == 0.6


def test_rethreshold_at_the_runs_own_threshold_changes_nothing():
    base = _artifact(0.6)
    output, _ = rethreshold_artifact(base, 0.6, kb=_kb())
    left, right = output["results"]["hybrid"], base["results"]["hybrid"]
    assert _accepted(output) == _accepted(base)
    assert left["pooled"] == right["pooled"] and left["datasets"] == right["datasets"]


def test_rethreshold_refuses_runs_it_cannot_reproduce(tmp_path: Path, capsys):
    with pytest.raises(NotComparableError, match="no stored acceptance probabilities"):
        rethreshold_artifact(_artifact(None, model=False), 0.5, kb=_kb())
    reranked = _artifact(0.6)
    reranked["results"]["hybrid"]["fingerprint_inputs"]["reranker"] = {"model": "x"}
    with pytest.raises(NotComparableError, match="reranker"):
        rethreshold_artifact(reranked, 0.5, kb=_kb())
    # a stored accepted set the rows cannot reproduce (e.g. truncated candidate rows)
    tampered = _artifact(0.6)
    tampered["results"]["hybrid"]["cases"][0]["accepted_technique_ids"] = ["T1105"]
    kb_path = _kb().save(tmp_path / "kb.json")
    source = _write(tmp_path / "tampered.json", tampered)
    output = tmp_path / "out.json"
    args = ["mapping", str(source), "--rethreshold", "0.5", "--output", str(output)]
    assert main([*args, "--kb", str(kb_path)]) == 2
    assert not output.exists()
    assert main([*args, "--kb", str(kb_path), "--allow-mismatch"]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["artifact_hash"] and written["rethresholded"]["threshold"] == 0.5
    capsys.readouterr()


def test_rethreshold_updates_the_run_identity_the_ledger_records(tmp_path: Path, capsys):
    """Regression: the top-level acceptance_model block (what run_ledger copies) must carry the
    applied threshold, not the source run's, and the ledger must show the re-threshold."""
    from scripts.run_ledger import main as ledger_main

    base = _artifact(0.6)
    base["acceptance_model"] = {
        "status": "active",
        "path": "data/models/acceptance/acceptance_T1.json",
        "threshold": 0.6,
        "exemplar_fingerprint": None,
        "sha256": "abc",
    }
    source = _write(tmp_path / "R31.json", base)
    kb_path = _kb().save(tmp_path / "kb.json")
    target = tmp_path / "P40.json"
    args = ["mapping", str(source), "--rethreshold", "0.5", "--output", str(target)]
    assert main([*args, "--kb", str(kb_path)]) == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    identity = written["acceptance_model"]
    assert identity["threshold"] == 0.5 and identity["rethresholded_from"] == 0.6
    assert identity["path"] == base["acceptance_model"]["path"]  # the same weights file
    ledger = tmp_path / "ledger.jsonl"
    append = ["append", "--ledger", str(ledger), "--run-id", "P40", "--artifact", str(target)]
    assert ledger_main(append, environ={}) == 0
    described = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])["artifacts"][0]
    assert described["acceptance_model"]["threshold"] == 0.5
    assert described["rethresholded"]["threshold"] == 0.5
    assert described["rethresholded"]["source_artifact"] == str(source)
    capsys.readouterr()


def test_rethreshold_refuses_a_run_whose_model_was_refused_per_call(tmp_path: Path):
    """Cases run without the model (rule-based gate) say nothing about a threshold; the ledger
    records which modes ran without it."""
    from scripts.run_ledger import describe_artifact

    artifact = _artifact(0.6)
    artifact["results"]["hybrid"]["acceptance_model_disabled"] = {
        "cases": 6,
        "reasons": {"acceptance model retrieval contract not met: retrieval_mode": 6},
    }
    with pytest.raises(NotComparableError, match="refused in 6 cases"):
        rethreshold_artifact(artifact, 0.5, kb=_kb())
    described = describe_artifact(_write(tmp_path / "run.json", artifact))
    assert described["acceptance_model_refused"] == {"hybrid": 6}
    assert (
        describe_artifact(_write(tmp_path / "ok.json", _artifact(0.6)))["acceptance_model_refused"]
        is None
    )


def test_pr_curve_self_checks_and_matches_the_run():
    block = _artifact(0.6)["results"]["hybrid"]
    curve = pr_curve(block)
    assert curve["available"] and curve["self_check"]["mismatches"] == 0
    accepted = [point["accepted"] for point in curve["points"]]
    assert accepted == sorted(accepted, reverse=True)  # higher thresholds accept less
    own = next(point for point in curve["points"] if point["run_threshold"])
    assert own["threshold"] == 0.6
    assert own["precision"] == pytest.approx(block["pooled"]["metrics"]["micro_precision"])
    assert own["recall"] == pytest.approx(block["pooled"]["metrics"]["micro_recall"])
    # the bands partition the labelled cases' acceptances at the lowest threshold (case 5 has
    # no labels, and its only candidate is benign, so it never adds one)
    assert sum(point["band_accepted"] for point in curve["points"]) == accepted[0]
    assert not pr_curve(_artifact(None, model=False)["results"]["hybrid"])["available"]


def test_paired_bootstrap_reproduces_the_artifact_deltas():
    base, current = _artifact(0.7), _artifact(0.5)
    result = paired_bootstrap(
        base["results"]["hybrid"], current["results"]["hybrid"], samples=200, seed=7
    )
    assert result["matches_artifact"] is True
    for name in ("micro_precision", "micro_recall", "micro_f1"):
        expected = (
            current["results"]["hybrid"]["pooled"]["metrics"][name]
            - base["results"]["hybrid"]["pooled"]["metrics"][name]
        )
        assert result[name]["delta"] == pytest.approx(expected)
        low, high = result[name]["ci95"]
        assert low <= high
    again = paired_bootstrap(
        base["results"]["hybrid"], current["results"]["hybrid"], samples=200, seed=7
    )
    assert again == result  # seeded


def test_mapping_expect_identical(tmp_path: Path, capsys):
    base = _artifact(0.6)
    first = _write(tmp_path / "a.json", base)
    second = _write(tmp_path / "b.json", copy.deepcopy(base))
    assert main(["mapping", str(first), str(second), "--expect-identical"]) == 0
    # newer code adds a candidate field: reported, not a difference
    newer = copy.deepcopy(base)
    for case in newer["results"]["hybrid"]["cases"]:
        for row in case["candidates"]:
            row["new_field"] = 1
    report = compare_mapping_identity(base, newer, modes=["hybrid"])
    assert report["identical"]
    assert report["modes"]["hybrid"]["candidate_fields_only_in_current"] == ["new_field"]
    changed = copy.deepcopy(base)
    changed["results"]["hybrid"]["cases"][1]["candidates"][0]["acceptance_probability"] += 1e-12
    third = _write(tmp_path / "c.json", changed)
    out = tmp_path / "report.json"
    assert (
        main(["mapping", str(first), str(third), "--expect-identical", "--output", str(out)]) == 1
    )
    identity = json.loads(out.read_text(encoding="utf-8"))["identity"]["modes"]["hybrid"]
    assert identity["differing_cases"] == 1
    assert identity["examples"][0]["fields"] == ["candidates[0].acceptance_probability"]
    other = _write(tmp_path / "d.json", _artifact(0.5))
    assert main(["mapping", str(first), str(other), "--paired-bootstrap", "50", "--pr-curve"]) == 0
    missing = _write(tmp_path / "e.json", {"results": {"lexical": {}}})
    assert main(["mapping", str(first), str(missing)]) == 2
    capsys.readouterr()


def test_emulated_materialization_drops_a_parent_that_only_its_child_supports():
    child = {
        "technique_id": "T1059.001",
        "parent_technique_id": "T1059",
        "state": "auto_accepted",
        "state_reason": "",
        "hierarchy_parent": False,
        "dense_score": 0.9,
        "dense_rank": 1,
        "acceptance_probability": 0.9,
        "benign": False,
        "sparse": False,
        "cue_supported": False,
    }
    # a cue-supported parent with no dense or lexical evidence of its own
    parent = {
        "technique_id": "T1059",
        "parent_technique_id": None,
        "state": "auto_accepted",
        "state_reason": "",
        "hierarchy_parent": False,
        "acceptance_probability": 0.1,
        "benign": False,
        "sparse": False,
        "cue_supported": True,
    }
    hierarchy = {
        "technique_id": "T1003",
        "state": "needs_review",
        "hierarchy_parent": True,
        "acceptance_probability": None,
    }
    accepted, states = emulate_case({"candidates": [child, parent, hierarchy]}, 0.5)
    assert accepted == ["T1059.001"] and states == [
        "auto_accepted",
        "auto_accepted",
        "needs_review",
    ]
    accepted, _ = emulate_case(
        {"candidates": [{**child, "acceptance_probability": 0.4}, parent]}, 0.5
    )
    assert accepted == []  # the child left for review is still visible, so the parent stays out


def _attribution(rows, unknown=(), metrics=None, parameters=None):
    return {
        "known": rows,
        "unknown": list(unknown),
        "metrics": metrics or {},
        "parameters": parameters or {},
        "pipeline_fingerprint": "pf",
    }


def _known(group, rank, stage, accepted, gold, report=None):
    return {
        "group_id": group,
        "report_id": report,
        "rank": rank,
        "stage": stage,
        "leading": group if rank == 1 else "G9999",
        "accepted_techniques": accepted,
        "held_out_techniques": gold,
        "technique_recall": len(set(accepted) & set(gold)) / len(gold) if gold else None,
    }


def test_attribution_names_technique_pr_and_combiner_shift(tmp_path: Path, capsys):
    named = "known_group_hypothesis"
    base_rows = [
        _known("G1", 1, named, ["T1", "T2"], ["T1", "T2", "T3"]),
        _known("G2", 2, "known_group_shortlist", ["T4", "T5"], ["T4"]),
    ]
    current_rows = [
        _known("G1", 1, named, ["T1", "T2", "T6", "T7", "T8"], ["T1", "T2", "T3"]),
        _known("G2", 2, named, ["T4", "T5", "T9", "T10"], ["T4"]),  # a wrong name
    ]
    unknown = [{"group_id": "G3", "stage": named, "leading": "G1"}]  # removed actor named
    base = _attribution(base_rows, metrics={"top1": 0.5, "rrf_top1": 0.5})
    current = _attribution(current_rows, unknown, metrics={"top1": 0.4, "rrf_top1": 0.5})
    a, b = _write(tmp_path / "a.json", base), _write(tmp_path / "b.json", current)
    out = tmp_path / "attr.json"
    args = ["attribution", str(a), str(b), "--names", "--technique-pr", "--combiner-shift"]
    assert main([*args, "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["names"]["base"]["wrong"] == []
    wrong = report["names"]["current"]["wrong"]
    assert {(item["section"], item["group_id"], item["leading"]) for item in wrong} == {
        ("known", "G2", "G9999"),
        ("unknown", "G3", "G1"),
    }
    assert report["names"]["newly_named"] == ["known:G2||0", "unknown:G3||0"]
    pr = report["technique_pr"]
    assert pr["base"]["technique_micro_precision"] == pytest.approx(3 / 4)
    assert pr["current"]["technique_micro_recall"] == pytest.approx(3 / 4)
    assert pr["current"]["median_accepted_techniques"] == 4.5
    shift = report["combiner_shift"]
    assert shift["triggered"] and any(reason.startswith("(a)") for reason in shift["reasons"])
    assert any(reason.startswith("(c)") for reason in shift["reasons"])  # top-1 fell, RRF held
    assert main(["attribution", str(a), str(a), "--expect-identical"]) == 0
    assert main(["attribution", str(a), str(b), "--expect-identical"]) == 1
    capsys.readouterr()


def _evidenceforge(accepted_red_herring: bool, f1: float) -> dict:
    cases = [
        {
            "case_id": "c1",
            "accepted_technique_ids": ["T1033"],
            "ranked_technique_ids": ["T1033"],
            "label_quality": "intent_non_exhaustive",
            "model_input_hash": "h1",
        },
        {
            "case_id": "c2",
            "accepted_technique_ids": ["T1105"] if accepted_red_herring else [],
            "ranked_technique_ids": ["T1105"],
            "label_quality": "red_herring",
            "model_input_hash": "h2",
        },
    ]
    return {
        "results": {"narrative": {"metrics": {"micro_f1": f1, "hit_at_1": 1.0}, "cases": cases}}
    }


def test_evidenceforge_compare_counts_red_herrings_and_differences(tmp_path: Path, capsys):
    a = _write(tmp_path / "a.json", _evidenceforge(False, 0.5))
    b = _write(tmp_path / "b.json", _evidenceforge(True, 0.4))
    out = tmp_path / "ef.json"
    assert main(["evidenceforge", str(a), str(a), "--expect-identical"]) == 0
    assert main(["evidenceforge", str(a), str(b), "--expect-identical", "--output", str(out)]) == 1
    mode = json.loads(out.read_text(encoding="utf-8"))["modes"]["narrative"]
    assert mode["red_herrings_accepted"] == {"base": 0, "current": 1}
    assert mode["metrics"]["micro_f1"]["delta"] == pytest.approx(-0.1)
    assert mode["differing_cases"] == 1 and mode["model_input_changed"] == 0
    capsys.readouterr()


def test_summary_composes_the_ledger(tmp_path: Path, capsys):
    lines = [
        {
            "run_id": "R01",
            "verdict": "recorded",
            "git_commit": "abc",
            "llm_cost_usd": 0.0,
            "chroma": {"clean": True},
            "artifacts": [{"path": "x.json", "artifact_hash": "h"}],
        },
        {
            "run_id": "R03",
            "verdict": "pass",
            "git_commit": "abc",
            "llm_cost_usd": None,
            "chroma": {"clean": False},
        },
    ]
    summary = summarize_ledger(lines)
    assert summary["verdicts"] == {"pass": 1, "recorded": 1}
    assert summary["chroma_unclean_runs"] == ["R03"] and summary["llm_cost_usd_total"] == 0.0
    assert summary["runs"][0]["artifacts"] == [{"path": "x.json", "artifact_hash": "h"}]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert main(["summary", "--ledger", str(ledger), "--output", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["git_commits"] == ["abc"]
    capsys.readouterr()


def test_mapping_identity_reports_order_dependent_metrics_without_failing():
    """Parent-aware metrics depend on set iteration order (PYTHONHASHSEED): the same case rows
    can score differently, so a difference there is listed but is not an identity failure."""

    def artifact(parent_f1: float, micro_f1: float = 0.5) -> dict:
        row = {"case_id": "c1", "accepted_technique_ids": ["T1059"], "candidates": []}
        metrics = {"micro_f1": micro_f1, "parent_aware_f1": parent_f1}
        return {
            "case_set_hash": "h",
            "results": {"hybrid": {"cases": [row], "pooled": {"metrics": metrics}}},
        }

    report = compare_mapping_identity(artifact(0.45), artifact(0.44), modes=["hybrid"])
    mode = report["modes"]["hybrid"]
    assert report["identical"] is True
    assert mode["metric_differences"] == {}
    assert mode["order_dependent_metric_differences"] == {"pooled.metrics": ["parent_aware_f1"]}
    changed = compare_mapping_identity(
        artifact(0.45), artifact(0.45, micro_f1=0.6), modes=["hybrid"]
    )
    assert changed["identical"] is False
    assert changed["modes"]["hybrid"]["metric_differences"] == {"pooled.metrics": ["micro_f1"]}
