"""Store posts as combiner training lists (prereg store_training_lists_2026-11-09): the readings,
the posts allowed in, the list records, the store lists kept out of the fold metrics and the
time-split weight choice."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apt_classifier.combiner import FEATURE_NAMES, NULL_FEATURE_NAMES, CombinerList
from tests.test_real_report_combiner import _records


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(report_id: str, published: str, **extra) -> dict:
    return {
        "report_id": report_id,
        "group_id": "G0050",
        "published": published,
        "statements": [{"text": "The adversary ran a loader on the host."}],
        "hidden_texts": [],
        **extra,
    }


def test_store_reading_masks_store_names_at_behaviour_only():
    from scripts.fit_attribution_combiner import DEFAULT_STORE_CONDITIONS, RealCondition

    behaviour = RealCondition.parse("behavior+store:all")
    assert behaviour == RealCondition("behavior", None, mask_store_names=True)
    assert behaviour.label == "behavior+store:all"
    assert RealCondition.parse(behaviour.label) == behaviour
    assert RealCondition.parse("strict:all").mask_store_names is False
    assert [c.label for c in DEFAULT_STORE_CONDITIONS] == ["strict:all", "behavior+store:all"]
    for bad in ("strict+store:all", "behavior+names:all"):
        with pytest.raises(ValueError):
            RealCondition.parse(bad)


def test_load_real_folds_drops_attribution_masks_store_names_and_splits_open_set(monkeypatch):
    import scripts.report_anonymizer as report_anonymizer
    from scripts.fit_attribution_combiner import RealCondition, load_real_folds

    made: list[str] = []

    class Dropper:
        def __init__(self, kb, level):
            made.append(level)

    calls: list[dict] = []
    folds = [
        SimpleNamespace(group_id="G1", report_id="known"),
        SimpleNamespace(group_id=None, report_id="open"),
        SimpleNamespace(group_id="G9", report_id="outside the knowledge base"),
    ]

    def anonymize(selected, kb, *, level, extra_names):
        calls.append({"level": level, "extra_names": extra_names})
        return selected, [], {}

    bench = SimpleNamespace(
        load_report_folds=lambda path, **kwargs: list(folds), anonymize_report_folds=anonymize
    )
    kb = SimpleNamespace(groups={"G1": object()})
    monkeypatch.setattr(report_anonymizer, "ReportAnonymizer", Dropper)
    behaviour = RealCondition.parse("behavior+store:all")
    with pytest.raises(ValueError):
        load_real_folds(bench, kb, Path("x"), behaviour)
    known = load_real_folds(bench, kb, Path("x"), behaviour, extra_names=("StoreName",))
    assert [f.report_id for f in known] == ["known"]
    opened = load_real_folds(
        bench, kb, Path("x"), behaviour, extra_names=("StoreName",), open_set=True
    )
    assert [f.report_id for f in opened] == ["open"]
    # the store names are masked only at the reading that asks for it
    load_real_folds(bench, kb, Path("x"), RealCondition("strict", None), extra_names=("S",))
    load_real_folds(bench, kb, Path("x"), RealCondition("names", 30))
    assert [c["extra_names"] for c in calls] == [("StoreName",), ("StoreName",), (), ()]
    # every level but names drops the report's own attribution prose (behaviour too)
    assert made == ["behavior", "behavior", "strict"]


def test_store_rows_refuse_window_posts_and_unhidden_sets(tmp_path):
    from scripts.fit_attribution_combiner import load_store_rows

    good = _write(tmp_path / "good.jsonl", [_row("a", "2024-12-31"), _row("b", "2019-03-01")])
    assert [row["report_id"] for row in load_store_rows(good)] == ["a", "b"]
    for name, rows in (
        ("window", [_row("a", "2024-06-01"), _row("b", "2025-01-01")]),
        ("undated", [_row("a", "")]),
        ("unhidden", [{k: v for k, v in _row("a", "2023-01-01").items() if k != "hidden_texts"}]),
    ):
        with pytest.raises(ValueError):
            load_store_rows(_write(tmp_path / f"{name}.jsonl", rows))


def _lst(truth: str | None) -> CombinerList:
    return CombinerList(
        ("G1", "G2"),
        np.zeros((2, len(FEATURE_NAMES))),
        {name: 0.5 for name in NULL_FEATURE_NAMES[1:]},
        truth,
    )


def test_store_lists_carry_arm_date_and_open_actor_and_resume(monkeypatch, tmp_path):
    import scripts.fit_attribution_combiner as fit

    path = _write(
        tmp_path / "train.jsonl",
        [
            _row("k1", "2023-05-01"),
            _row("k2", "2024-02-01"),
            {**_row("o1", "2022-07-01"), "group_id": None, "actor": "Water Labbu"},
        ],
    )
    known = [SimpleNamespace(group_id="G1", report_id=rid) for rid in ("k1", "k2")]
    opened = [SimpleNamespace(group_id=None, report_id="o1")]

    def folds(bench, kb, p, condition, *, extra_names=(), open_set=False):
        assert extra_names == ("StoreName",)
        return list(opened if open_set else known)

    class Cases:
        def __init__(self, engine, kb, bench):
            self.runs: list[str] = []

        def run(self, fold, *, top_n, truth):
            assert truth == fold.group_id
            return _lst(truth), {}

        def restore(self):
            pass

    engines: list[str] = []
    monkeypatch.setattr(fit, "load_real_folds", folds)
    monkeypatch.setattr(fit, "_FullEngineCases", Cases)
    monkeypatch.setattr(fit, "_release_gpu", lambda: None)
    monkeypatch.setattr(
        fit,
        "_removed_actor_engine",
        lambda settings, kb, store, bench, fold: (engines.append(fold.report_id), None),
    )
    monkeypatch.setattr(fit, "_removed_actor_list", lambda *args: _lst(None))
    conditions = list(fit.DEFAULT_STORE_CONDITIONS)
    common = {
        "sets": {"train": path},
        "conditions": conditions,
        "top_n": 20,
        "log": lambda _: None,
        "extra_names": ("StoreName",),
    }
    saved: list[int] = []
    records = fit.generate_store_lists(
        object(),
        None,
        None,
        None,
        None,
        **common,
        unknown=True,
        done=frozenset({("k1", "strict:all", "removed")}),
        on_done=lambda so_far: saved.append(len(so_far)),
        checkpoint_every=4,
    )
    by_key = {(r["report_id"], r["condition"], r["arm"]): r for r in records}
    assert ("k1", "strict:all", "removed") not in by_key  # resumed: not generated again
    assert len(records) == 2 * 3 + 3  # known + open per reading, removed-actor minus the done one
    known_list = by_key[("k2", "behavior+store:all", "known")]
    assert known_list["source"] == "store:train" and known_list["published"] == "2024-02-01"
    assert known_list["variant"] == 1 and known_list["truth"] == "G1"
    open_list = by_key[("o1", "strict:all", "open")]
    assert open_list["group_id"] == "open:Water Labbu" and open_list["variant"] == -1
    assert open_list["truth"] is None
    removed = by_key[("k1", "behavior+store:all", "removed")]
    assert removed["variant"] == -1 and "actor removed" in removed["leak_control"]
    # one reduced engine per post serves every reading it still needs
    assert engines == ["k1", "k2"]
    assert saved and saved[-1] == len(records)


def _store(records: list[dict], *, published: str, arm: str = "known", condition: str) -> list:
    return [
        {**r, "source": "store:train", "arm": arm, "published": published, "condition": condition}
        for r in records
    ]


def test_store_lists_stay_out_of_the_fold_and_real_metrics():
    from scripts.fit_attribution_combiner import _cv_metrics

    rng = np.random.default_rng(5)
    groups = [f"G{i}" for i in range(10)]
    folds = _records(rng, 200, signal="rrf_rel", source=None, groups=groups)
    store = _store(
        _records(rng, 60, signal=FEATURE_NAMES[-1], source=None, groups=groups),
        published="2023-01-01",
        condition="strict:all",
    )
    split_of = {g: i % 5 for i, g in enumerate(groups)}
    alone = _cv_metrics(folds, split_of, 5, 1.0, False)
    beside = _cv_metrics(folds + store, split_of, 5, 1.0, False)
    # at store weight 0 the store lists change nothing the fold metrics read
    assert beside["combiner"]["clean"] == alone["combiner"]["clean"]
    assert beside["combiner"]["store"]["count"] == 60
    assert beside["combiner"]["store strict:all"]["count"] == 60
    assert "real" not in beside["combiner"]
    weighted = _cv_metrics(folds + store, split_of, 5, 1.0, False, store_weight=3)
    assert weighted["store_weight"] == 3
    assert weighted["combiner"]["store"]["mrr"] > beside["combiner"]["store"]["mrr"]


def _time_split_records(rng, *, signal: str) -> list[dict]:
    groups = [f"G{i}" for i in range(10)]
    records = _records(rng, 300, signal="rrf_rel", source=None, groups=groups)
    for condition in ("strict:all", "behavior+store:all"):
        records += _store(
            _records(rng, 80, signal=signal, source=None, groups=groups),
            published="2023-03-01",
            condition=condition,
        )
        records += _store(
            _records(rng, 40, signal=signal, source=None, groups=groups),
            published="2024-03-01",
            condition=condition,
        )
    return records


def test_time_split_picks_a_store_weight_only_when_later_posts_gain():
    from scripts.fit_attribution_combiner import select_store_weight

    def run(records, margin):
        return select_store_weight(
            records,
            folds=5,
            l2_grid=[1.0],
            null_option=False,
            seed=97,
            log=lambda _: None,
            real_weight=10,
            store_weights=[3],
            choice_from="2024-01-01",
            guard_margin=margin,
        )

    rng = np.random.default_rng(7)
    # store posts decided by a feature the folds ignore: weighing the early posts helps the later
    helped = run(_time_split_records(rng, signal=FEATURE_NAMES[-1]), 1.0)
    later = helped["grid"]["s3_l21.0"]["time_split"]["behavior+store:all"]
    baseline = helped["grid"]["s0_l21.0"]["time_split"]["behavior+store:all"]
    assert later["count"] == baseline["count"] == 40
    assert later["first"] > baseline["first"]
    assert helped["chosen_store_weight"] == 3 and helped["chosen_real_weight"] == 10
    assert helped["store_known_posts_scored"] == {"strict:all": 40, "behavior+store:all": 40}
    # a guard of no fold loss: a weight that costs the folds is not eligible
    strict = run(_time_split_records(np.random.default_rng(7), signal=FEATURE_NAMES[-1]), 0.0)
    clean = {key: cell["combiner"]["clean"]["top1"] for key, cell in strict["grid"].items()}
    if clean["s3_l21.0"] < clean["s0_l21.0"]:
        assert strict["selection"]["eligible"] == ["s0_l21.0"]
        assert strict["chosen_store_weight"] == 0
    # no store post before the cut: every weight trains the same model, the tie keeps weight 0
    rng = np.random.default_rng(9)
    late_only = [
        r
        for r in _time_split_records(rng, signal=FEATURE_NAMES[-1])
        if r.get("published") != "2023-03-01"
    ]
    tied = run(late_only, 1.0)
    assert tied["store_lists_before"] == 0
    assert tied["grid"]["s3_l21.0"]["time_split"] == tied["grid"]["s0_l21.0"]["time_split"]
    assert tied["chosen_store_weight"] == 0
    with pytest.raises(ValueError):
        run([r for r in late_only if "arm" not in r], 1.0)  # no later post to score
    with pytest.raises(ValueError):
        run(late_only, -0.01)  # a negative margin could exclude weight 0 itself


def test_fit_script_store_arguments_are_checked_before_anything_runs(tmp_path, capsys):
    from scripts.fit_attribution_combiner import main

    good = _write(tmp_path / "train.jsonl", [_row("a", "2024-05-01")])
    window = _write(tmp_path / "window.jsonl", [_row("a", "2025-02-01")])
    common = [
        "--output",
        str(tmp_path / "c.json"),
        "--report",
        str(tmp_path / "r.json"),
        "--lists-cache",
        str(tmp_path / "l.jsonl.gz"),
    ]
    cache = ["--store-cache", str(tmp_path / "s.jsonl.gz"), "--real-weights", "10"]
    cases = [
        ([*cache, "--store-reports", "cti_hal=x.jsonl"], "test sets"),
        ([*cache, "--store-reports", "t=data/benchmarks/elastic/reports.jsonl"], "test sets"),
        (["--store-reports", f"t={good}"], "--store-cache"),
        ([*cache, "--real-weights", "0,10", "--store-reports", f"t={good}"], "one fixed"),
        ([*cache, "--store-reports", f"t={window}"], "before 2025-01-01"),
        ([*cache, "--store-conditions", "strict+store:all"], "behavior+store"),
    ]
    for argv, message in cases:
        with pytest.raises(SystemExit):
            main([*common, *argv])
        assert message in capsys.readouterr().err
