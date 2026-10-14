"""Paragraph rows read whole or sentence by sentence (prereg ``cribl_sentence_split_2026-10-14.json``).

The public test's ``expert`` and ``derived_procedures`` rows are report paragraphs (median 62 and 49
words) labelled with every technique they describe. The family head read each row whole. Here each
paragraph is split the way the product splits free text without the LLM
(``apt_classifier.evidence.local_extract.local_observations``: per line, ``split_sentences``,
sentences of 3+ words, repeats dropped), the model reads every sentence, and the answers are merged
per paragraph. Grading stays per paragraph against the paragraph's labels; sentences carry no labels.

Merging: a family's paragraph score is its highest probability in any of the paragraph's sentences
(the 20 kept per sentence), with the most probable technique of that family in that sentence. Both
readings give the same number of answers per paragraph:
- ``k`` answers (1 to 5): the ``k`` best families of the paragraph read whole, or of the merged list;
- one answer per sentence: the distinct top families of the sentences (``n`` per paragraph), against
  the paragraph read whole with its ``n`` best families.

  split:    (project .venv)  -> data/audits/cribl_ft/cases_test_sentences.jsonl
  predict:  (.venv-cribl)    --checkpoint <dir> --name E2 -> <name>_test_sentences.jsonl
  score:    (either venv)    E2 M1 -> per dataset and k: labelled families found (recall), answers
            that are labelled (precision), a labelled family found (hit), every labelled family
            found; exact technique recall/precision; split minus whole with paired 95% CIs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from common import bootstrap_mean_differences, read_jsonl, write_jsonl  # noqa: E402

AUDITS = HERE.parents[1] / "data" / "audits" / "cribl_ft"
DATASETS = ("expert", "derived_procedures")
SENTENCES = "cases_test_sentences.jsonl"
KS = (1, 2, 3, 4, 5)


def family_of(technique: str) -> str:
    return technique.split(".")[0]


def split(args: argparse.Namespace) -> None:
    from apt_classifier.evidence.local_extract import local_observations

    rows, counts = [], {}
    for case in read_jsonl(AUDITS / "cases_test.jsonl"):
        if case["dataset"] not in DATASETS:
            continue
        sentences = [item.text for item in local_observations(case["text"])] or [case["text"]]
        counts.setdefault(case["dataset"], []).append(len(sentences))
        for number, sentence in enumerate(sentences, start=1):
            rows.append(
                {
                    "case_id": f"{case['case_id']}#s{number}",
                    "paragraph_id": case["case_id"],
                    "dataset": case["dataset"],
                    "text": sentence,
                    "gold_ids": case["gold_ids"],  # the paragraph's labels; not used for scoring
                }
            )
    write_jsonl(AUDITS / SENTENCES, rows)
    for name, sizes in counts.items():
        print(
            f"{name}: {len(sizes)} paragraphs, {sum(sizes)} sentences, median {sorted(sizes)[len(sizes) // 2]}"
        )


def predict(args: argparse.Namespace) -> None:
    from decision_model import with_oom_retry
    from family_head import load_checkpoint, load_techniques, run_split

    cases = read_jsonl(AUDITS / SENTENCES)
    _, encoder, head, info = load_checkpoint(args.checkpoint)
    techniques = load_techniques(args.checkpoint, encoder.device)
    path = AUDITS / f"{args.name}_test_sentences.jsonl"
    with_oom_retry(
        lambda: run_split(
            encoder, head, info["families"], cases, path, batch=args.batch, techniques=techniques
        ),
        note=path.name,
    )
    print(f"{path}: {len(cases)} sentences")


def best_techniques(row: Mapping[str, Any]) -> dict[str, str]:
    best: dict[str, str] = {}
    for technique, _ in row.get("technique_top") or ():
        best.setdefault(family_of(technique), technique)
    return best


def whole_ranking(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(family, technique) by probability for a paragraph read whole."""
    best = best_techniques(row)
    return [(family, best.get(family, family)) for family, _ in row["top"]]


def merged_ranking(sentences: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """(family, technique) by the family's highest probability in any sentence."""
    score: dict[str, tuple[float, str]] = {}
    for row in sentences:
        if row.get("status") != "ok":
            continue
        best = best_techniques(row)
        for family, p in row["top"]:
            if family not in score or p > score[family][0]:
                score[family] = (p, best.get(family, family))
    ranked = sorted(score.items(), key=lambda item: -item[1][0])
    return [(family, technique) for family, (_, technique) in ranked]


def per_sentence_answers(sentences: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """The distinct top family of every sentence, with that sentence's best technique for it."""
    found: dict[str, str] = {}
    for row in sentences:
        if row.get("status") == "ok":
            family = row["top"][0][0]
            found.setdefault(family, best_techniques(row).get(family, family))
    return list(found.items())


def grade(answers: Sequence[tuple[str, str]], gold_ids: Sequence[str]) -> dict[str, float]:
    families = {family for family, _ in answers}
    techniques = {technique for _, technique in answers}
    gold_families = {family_of(item) for item in gold_ids}
    gold = set(gold_ids)
    return {
        "answers": len(answers),
        "found": len(families & gold_families),
        "labelled": len(gold_families),
        "correct": len(families & gold_families),
        "recall": len(families & gold_families) / len(gold_families),
        "hit": float(bool(families & gold_families)),
        "all": float(gold_families <= families),
        "exact_found": len(techniques & gold),
        "exact_labelled": len(gold),
        "exact_recall": len(techniques & gold) / len(gold),
    }


def summary(grades: Sequence[Mapping[str, float]]) -> dict[str, float]:
    n = len(grades)
    answers = sum(g["answers"] for g in grades)
    return {
        "paragraphs": n,
        "answers_per_paragraph": round(answers / n, 3),
        "family_recall": round(
            sum(g["found"] for g in grades) / sum(g["labelled"] for g in grades), 4
        ),
        "family_precision": round(sum(g["correct"] for g in grades) / answers, 4),
        "hit": round(sum(g["hit"] for g in grades) / n, 4),
        "all_found": round(sum(g["all"] for g in grades) / n, 4),
        "exact_recall": round(
            sum(g["exact_found"] for g in grades) / sum(g["exact_labelled"] for g in grades), 4
        ),
        "exact_precision": round(sum(g["exact_found"] for g in grades) / answers, 4),
    }


def compare(whole: Sequence[Mapping[str, float]], split_: Sequence[Mapping[str, float]]) -> dict:
    """Split minus whole, per-paragraph family and exact recall (paired bootstrap)."""
    columns = {
        "family_recall": [s["recall"] - w["recall"] for w, s in zip(whole, split_, strict=True)],
        "exact_recall": [
            s["exact_recall"] - w["exact_recall"] for w, s in zip(whole, split_, strict=True)
        ],
    }
    return {
        "delta": {name: round(sum(v) / len(v), 4) for name, v in columns.items()},
        "ci95": bootstrap_mean_differences(columns),
    }


def score(args: argparse.Namespace) -> None:
    report: dict[str, Any] = {}
    for name in args.names:
        whole = {row["case_id"]: row for row in read_jsonl(AUDITS / f"{name}_test.jsonl")}
        sentences: dict[str, list[dict[str, Any]]] = {}
        for row in read_jsonl(AUDITS / f"{name}_test_sentences.jsonl"):
            sentences.setdefault(row["case_id"].rsplit("#s", 1)[0], []).append(row)
        report[name] = {}
        for dataset in DATASETS:
            ids = [cid for cid in sentences if whole[cid]["dataset"] == dataset]
            out: dict[str, Any] = {}
            for k in KS:
                w = [grade(whole_ranking(whole[cid])[:k], whole[cid]["gold_ids"]) for cid in ids]
                s = [
                    grade(merged_ranking(sentences[cid])[:k], whole[cid]["gold_ids"]) for cid in ids
                ]
                out[f"k{k}"] = {
                    "whole": summary(w),
                    "split": summary(s),
                    "split_minus_whole": compare(w, s),
                }
            w, s = [], []
            for cid in ids:
                answers = per_sentence_answers(sentences[cid])
                s.append(grade(answers, whole[cid]["gold_ids"]))
                w.append(grade(whole_ranking(whole[cid])[: len(answers)], whole[cid]["gold_ids"]))
            out["one_per_sentence"] = {
                "whole": summary(w),
                "split": summary(s),
                "split_minus_whole": compare(w, s),
            }
            report[name][dataset] = out
            for key, block in out.items():
                d = block["split_minus_whole"]
                print(
                    f"{name} {dataset} {key}: answers {block['whole']['answers_per_paragraph']} | "
                    f"family recall whole {block['whole']['family_recall']} split "
                    f"{block['split']['family_recall']} | precision {block['whole']['family_precision']}"
                    f" / {block['split']['family_precision']} | hit {block['whole']['hit']} / "
                    f"{block['split']['hit']} | all found {block['whole']['all_found']} / "
                    f"{block['split']['all_found']} | exact recall {block['whole']['exact_recall']} / "
                    f"{block['split']['exact_recall']} | split-whole {d['delta']} {d['ci95']}"
                )
    (AUDITS / "sentence_split_test.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    commands.add_parser("split").set_defaults(func=split)
    run = commands.add_parser("predict")
    run.set_defaults(func=predict)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--batch", type=int, default=8)
    check = commands.add_parser("score")
    check.set_defaults(func=score)
    check.add_argument("names", nargs="+")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
