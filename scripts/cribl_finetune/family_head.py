"""Technique families without retrieval: cribl-decision reads one sentence and scores every ATT&CK
family (``.venv-cribl``; prereg ``docs/preregistration/cribl_family_head_2026-10-13.json``).

The prompt holds only the sentence (the C1 state) and a family question, no candidate options
(:func:`common.render_family_prompt`). The hidden state at the decision slot goes through a cosine
head with one row per family of ``catalog.json`` (active parent techniques, ``export_cases.py
--catalog-only``). Each row starts as the mean decision-slot state of the family's own ATT&CK texts
(the parent's and its sub-techniques' label and cut description, read by the starting adapter), so
the untrained head (``step_0``) is a nearest-description classifier. Training updates the LoRA and
the head with cross-entropy against the gold families (uniform over a case's gold families).

  train:     python family_head.py train --arm F1 --init cribl
             -> data/models/cribl_ft/<arm>/step_<n>/ (adapter + family_head.pt), every half epoch
                scored on public dev and AnnoCTR dev; data/audits/cribl_ft/<arm>_train_log.json
  evaluate:  python family_head.py evaluate --checkpoint <dir> --name F1 --split cti_hal
             [--compare B3=B3_cti_hal.jsonl ...]
             -> data/audits/cribl_ft/<name>_<split>.jsonl / .json

A checkpoint is chosen on the mean family top-1 over the two dev sets (ties: the earlier) and never
on CTI-HAL or a test split.

With ``--technique-head`` a second cosine head scores every listed technique (rows start from each
technique's own ATT&CK text) and is read as P(technique | family), a softmax within the family's
parent and sub-techniques (``hierarchy.py``). The exact answer is the most probable technique by
P(family) x P(technique | family) among the ``--families-considered`` most probable families.
Training adds the within-family cross-entropy for rows whose labels are exact (not the
``--family-only-datasets``, whose links are parent-level by convention); ``--exact-selection``
names the dev sets scored on exact top-1 instead of family top-1 in the selection score (prereg
``cribl_technique_head_2026-10-14.json``).

With ``--family-loss yesno`` every family gets its own yes/no probability, sigmoid(logit), instead of
a share of one answer (softmax), trained with :func:`losses.yes_no` (a missing label is weak evidence
of "no", ``--negative-weight``), so a sentence that describes several techniques can score several
families high; ``multi_answer.py`` turns either kind into up to three answers per sentence. The
checkpoint records the loss, and prediction reads the probabilities the same way (prereg
``cribl_yesno_2026-10-14.json``). ``--multi-label-weight`` multiplies the loss of every training row
labelled with two or more families, the 4% of rows that show a text with several techniques
(prereg ``cribl_multilabel_2026-10-14.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from common import (
    FLAT_K,
    c1_state,
    expected_calibration_error,
    family_targets,
    forced_row,
    hits,
    paired_delta,
    parent_of,
    read_jsonl,
    reference_rows,
    render_family_prompt,
    score_rows,
)
from decision_model import (
    BASES,
    DEFAULT_BASE,
    LORA_R,
    is_oom,
    load_model,
    wait_after_oom,
    with_oom_retry,
)
from hierarchy import (
    joint_log_probs,
    pick,
    technique_index,
    technique_targets,
    within_family_log_probs,
)
from losses import rival_margin, yes_no
from multi_answer import budget_threshold, multi_block
from train import schedule

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"
MODELS = REPO_ROOT / "data" / "models" / "cribl_ft"
HEAD_FILE = "family_head.pt"
TECHNIQUE_HEAD_FILE = "technique_head.pt"
DEV_FILES = ("cases_dev.jsonl", "cases_annoctr_dev.jsonl")
TRAIN_FILES = ("cases_train.jsonl", "cases_annoctr_train.jsonl")
SPLITS = (
    "dev",
    "annoctr_dev",
    "test",
    "annoctr_test",
    "cti_hal",
    "cti_rationale_test",
    "cti_rationale_validation",
)
# every split also has a parent-relabelled copy (relabel_parents.py: cases_<split>_parents.jsonl)
EVAL_SPLITS = (*SPLITS, *(f"{split}_parents" for split in SPLITS))
SELECTION_RULES = ("legacy", "parent")
INITIAL_SCALE = 16.0
TOP_KEPT = 20
MAX_OOM_SKIPS = 50
REPORT_BUDGET = 1.3  # answers per sentence for the multi-answer dev numbers in the training log


def load_catalog(path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]], str]:
    """(families in catalogue order, family -> its techniques' option texts, ATT&CK version)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    texts: dict[str, list[dict[str, str]]] = {}
    for item in data["techniques"]:
        texts.setdefault(item["family"], []).append(item)
    families = [family for family, items in texts.items() if items[0]["id"] == family]
    return families, texts, str(data["attack_version"])


class FamilyHead(torch.nn.Module):
    """Scaled cosine between the centred decision-slot state and one row per family, plus a bias."""

    def __init__(self, prototypes: torch.Tensor, center: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("center", center.float().clone())
        self.weight = torch.nn.Parameter(prototypes.float() - center.float())
        self.bias = torch.nn.Parameter(torch.zeros(prototypes.shape[0]))
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(INITIAL_SCALE)))
        self.independent = False  # True: a yes/no sigmoid per family (``--family-loss yesno``)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        state = functional.normalize(hidden.float() - self.center, dim=-1)
        rows = functional.normalize(self.weight, dim=-1)
        return self.log_scale.exp() * (state @ rows.T) + self.bias


class Encoder:
    """The decision-slot hidden state (float32) of the family prompt. Batches are left-padded with
    an attention mask, so the last position is every prompt's decision slot (states match the
    one-at-a-time ones to bf16 noise, cosine > 0.9997)."""

    def __init__(
        self, model: Any, tokenizer: Any, max_tokens: int = 1024, template: str = "qwen"
    ) -> None:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.backbone = base.model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.template = template  # the base model's chat format (decision_model.BaseSpec)
        # the first GPU parameter (a lookup table may live on the CPU on a small GPU)
        self.device = next(p.device for p in model.parameters() if p.device.type == "cuda")

    def token_ids(self, sentence: str) -> list[int]:
        prompt = render_family_prompt(c1_state(sentence), self.template)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens (> {self.max_tokens})")
        return ids

    def hidden_batch(self, id_lists: list[list[int]]) -> torch.Tensor:
        """(len(id_lists), hidden) states at the decision slot of each prompt."""
        width = max(len(ids) for ids in id_lists)
        pad = self.tokenizer.pad_token_id
        input_ids = [[pad] * (width - len(ids)) + ids for ids in id_lists]
        mask = [[0] * (width - len(ids)) + [1] * len(ids) for ids in id_lists]
        output = self.backbone(
            input_ids=torch.tensor(input_ids, device=self.device),
            attention_mask=torch.tensor(mask, device=self.device),
            use_cache=False,
        )
        return output.last_hidden_state[:, -1].float()


def initial_head(
    encoder: Encoder,
    families: list[str],
    texts: dict[str, list[dict[str, str]]],
    batch: int = 16,
) -> FamilyHead:
    """Rows = mean decision-slot state of each family's technique texts; centre = their mean."""
    with torch.inference_mode():
        rows = []
        for family in families:
            ids = [encoder.token_ids(f"{item['label']}: {item['what']}") for item in texts[family]]
            states = torch.cat(
                [encoder.hidden_batch(ids[i : i + batch]) for i in range(0, len(ids), batch)]
            )
            rows.append(states.mean(dim=0))
        prototypes = torch.stack(rows)
    prototypes = prototypes.clone()  # outside inference mode: a trainable tensor
    return FamilyHead(prototypes, prototypes.mean(dim=0)).to(encoder.device)


def yes_no_offset(
    encoder: Encoder, head: FamilyHead, sentences: list[str], batch: int = 16
) -> float:
    """Minus the median second-highest logit over training sentences: a shared starting bias under
    which a typical sentence has only its top family above P = 0.5 (with no bias, the untrained
    cosine head puts dozens of families above 0.9)."""
    seconds = []
    with torch.inference_mode():
        for start in range(0, len(sentences), batch):
            ids = [encoder.token_ids(text) for text in sentences[start : start + batch]]
            logits = head(encoder.hidden_batch(ids)) - head.bias
            seconds.extend(torch.topk(logits, 2, dim=-1).values[:, 1].tolist())
    return -sorted(seconds)[len(seconds) // 2]


@dataclass
class Techniques:
    """The technique head (a :class:`FamilyHead` with one row per technique), the technique ids,
    the family index of each (on the head's device) and how many families the answer may come
    from."""

    head: FamilyHead
    ids: list[str]
    group: torch.Tensor
    considered: int


def initial_technique_head(
    encoder: Encoder,
    families: list[str],
    texts: dict[str, list[dict[str, str]]],
    batch: int = 16,
) -> FamilyHead:
    """Rows = the decision-slot state of each technique's own text; centre = their mean."""
    items = [item for family in families for item in texts[family]]
    with torch.inference_mode():
        ids = [encoder.token_ids(f"{item['label']}: {item['what']}") for item in items]
        states = torch.cat(
            [encoder.hidden_batch(ids[i : i + batch]) for i in range(0, len(ids), batch)]
        )
    prototypes = states.clone()  # outside inference mode: a trainable tensor
    return FamilyHead(prototypes, prototypes.mean(dim=0)).to(encoder.device)


def save_checkpoint(
    model: Any,
    head: FamilyHead,
    directory: Path,
    meta: dict[str, Any],
    techniques: Techniques | None = None,
) -> None:
    model.save_pretrained(str(directory))
    torch.save({"state": head.state_dict(), **meta}, directory / HEAD_FILE)
    if techniques is not None:
        torch.save(
            {
                "state": techniques.head.state_dict(),
                "techniques": techniques.ids,
                "group": techniques.group.tolist(),
                "families_considered": techniques.considered,
            },
            directory / TECHNIQUE_HEAD_FILE,
        )


def load_techniques(directory: Path, device: Any) -> Techniques | None:
    """The checkpoint's technique head, or ``None`` for a family-only checkpoint."""
    path = directory / TECHNIQUE_HEAD_FILE
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    count, hidden = saved["state"]["weight"].shape
    head = FamilyHead(torch.zeros(count, hidden), torch.zeros(hidden))
    head.load_state_dict(saved["state"])
    return Techniques(
        head.to(device).eval(),
        list(saved["techniques"]),
        torch.tensor(saved["group"], device=device),
        int(saved["families_considered"]),
    )


def load_checkpoint(directory: Path) -> tuple[Any, Encoder, FamilyHead, dict[str, Any]]:
    saved = torch.load(directory / HEAD_FILE, map_location="cpu", weights_only=False)
    base = saved.get("base", DEFAULT_BASE)  # checkpoints before the registry are Qwen's
    model, tokenizer, info = load_model(str(directory), base=base)
    count = len(saved["families"])
    hidden = saved["state"]["center"].shape[0]
    head = FamilyHead(torch.zeros(count, hidden), torch.zeros(hidden))
    head.load_state_dict(saved["state"])
    head.independent = saved.get("family_loss") == "yesno"
    encoder = Encoder(model, tokenizer, template=info["prompt_template"])
    model.eval()
    return (
        model,
        encoder,
        head.to(encoder.device).eval(),
        {**info, **{key: value for key, value in saved.items() if key != "state"}},
    )


def predict_batch(
    encoder: Encoder,
    head: FamilyHead,
    families: list[str],
    cases: list[dict[str, Any]],
    techniques: Techniques | None = None,
) -> list[dict[str, Any]]:
    """One row per case; a prompt over the token limit is an error row (a miss)."""
    rows, golds, ids, scored = [], [], [], []
    for case in cases:
        gold = family_targets(case["gold_ids"], families)
        row = {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "gold_ids": case["gold_ids"],
            "gold_families": [families[index] for index in gold],
        }
        try:
            ids.append(encoder.token_ids(case["text"]))
            scored.append(len(rows))
        except ValueError as error:
            row = {**row, "status": "error", "error": str(error), "pick": None, "p_pick": None}
        rows.append(row)
        golds.append(gold)
    if ids:
        hidden = encoder.hidden_batch(ids)
        logits = head(hidden)
        if head.independent:
            probabilities, family_log_p = torch.sigmoid(logits), functional.logsigmoid(logits)
        else:
            probabilities, family_log_p = torch.softmax(logits, -1), torch.log_softmax(logits, -1)
        if techniques is not None:
            within = within_family_log_probs(
                techniques.head(hidden), techniques.group, len(families)
            )
            joint = joint_log_probs(family_log_p, within, techniques.group)
        for position, index in enumerate(scored):
            rows[index] = scored_row(rows[index], probabilities[position], golds[index], families)
            if techniques is not None:
                rows[index].update(
                    technique_fields(
                        joint[position], family_log_p[position], rows[index]["gold_ids"], techniques
                    )
                )
    return rows


def technique_fields(
    joint: torch.Tensor,
    family_log_p: torch.Tensor,
    gold_ids: list[str],
    techniques: Techniques,
) -> dict[str, Any]:
    """The exact answer under the registered rule, with the top family only and with every family,
    the top techniques by joint probability and the best gold technique's rank."""
    chosen = pick(joint, family_log_p, techniques.group, techniques.considered)
    ranked = torch.argsort(joint, descending=True).tolist()
    position = {index: rank for rank, index in enumerate(ranked, start=1)}
    probabilities = joint.exp()
    gold = technique_targets(gold_ids, techniques.ids)
    return {
        "technique_pick": techniques.ids[chosen],
        "technique_p": float(probabilities[chosen]),
        "technique_pick_k1": techniques.ids[pick(joint, family_log_p, techniques.group, 1)],
        "technique_pick_all": techniques.ids[ranked[0]],
        "technique_top": [
            [techniques.ids[index], round(float(probabilities[index]), 6)]
            for index in ranked[:TOP_KEPT]
        ],
        "technique_gold_rank": min((position[index] for index in gold), default=None),
    }


def scored_row(
    row: dict[str, Any], probabilities: torch.Tensor, gold: list[int], families: list[str]
) -> dict[str, Any]:
    ranked = torch.argsort(probabilities, descending=True).tolist()
    position = {index: rank for rank, index in enumerate(ranked, start=1)}
    return {
        **row,
        "status": "ok",
        "pick": families[ranked[0]],
        "p_pick": float(probabilities[ranked[0]]),
        "top": [
            [families[index], round(float(probabilities[index]), 6)] for index in ranked[:TOP_KEPT]
        ],
        "gold_rank": min((position[index] for index in gold), default=None),
        "gold_p": round(sum(float(probabilities[index]) for index in gold), 6),
    }


def run_split(
    encoder: Encoder,
    head: FamilyHead,
    families: list[str],
    cases: list[dict[str, Any]],
    path: Path,
    progress: int = 200,
    batch: int = 16,
    techniques: Techniques | None = None,
) -> list[dict[str, Any]]:
    """Predict every case not yet in ``path`` (appending, resumable); rows in case order."""
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        done = {row["case_id"]: row for row in read_jsonl(path)}
    pending = [case for case in cases if case["case_id"] not in done]
    started, fresh = time.monotonic(), 0
    with torch.inference_mode(), path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            for row in predict_batch(encoder, head, families, chunk, techniques):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                done[row["case_id"]] = row
                fresh += 1
                if progress and fresh % progress == 0:
                    rate = (time.monotonic() - started) / fresh
                    print(f"  {len(done)}/{len(cases)} cases, {rate:.3f} s/case", flush=True)
            handle.flush()
    return [done[case["case_id"]] for case in cases]


def family_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Family top-1/3/5, MRR and the calibration of P(pick) against a family hit; an error or a
    gold family outside the catalogue is a miss."""
    count = len(rows)
    ranks = [row.get("gold_rank") for row in rows]

    def within(k: int) -> float | None:
        return round(sum(1 for rank in ranks if rank and rank <= k) / count, 6) if count else None

    pairs = [
        (float(row["p_pick"]), parent_of(row["pick"]) in {parent_of(g) for g in row["gold_ids"]})
        for row in rows
        if row.get("pick") is not None
    ]
    return {
        "n": count,
        "top1": within(1),
        "top3": within(3),
        "top5": within(5),
        "mrr": round(sum(1 / rank for rank in ranks if rank) / count, 6) if count else None,
        "ece": expected_calibration_error(pairs),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
    }


def technique_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact top-1 under the registered rule, with the top family only and with every family;
    exact top-3/5 over every technique; exact top-1 by gold kind (a gold sub-technique, or parent
    ids only); the calibration of P(pick) against an exact hit. Errors are misses."""
    count = len(rows)

    def exact(row: dict[str, Any], key: str = "technique_pick") -> bool:
        return row.get(key) is not None and row[key] in row["gold_ids"]

    def rate(subset: list[dict[str, Any]], key: str = "technique_pick") -> float | None:
        return round(sum(exact(row, key) for row in subset) / len(subset), 6) if subset else None

    def within(k: int) -> float | None:
        ranks = [row.get("technique_gold_rank") for row in rows]
        return round(sum(1 for rank in ranks if rank and rank <= k) / count, 6) if count else None

    sub_technique = [row for row in rows if any("." in str(g) for g in row["gold_ids"])]
    parent_only = [row for row in rows if not any("." in str(g) for g in row["gold_ids"])]
    pairs = [
        (float(row["technique_p"]), exact(row))
        for row in rows
        if row.get("technique_pick") is not None
    ]
    # the answered technique's parent among the gold ids' parents: granularity forgiven
    parent_level = sum(hits(row.get("technique_pick"), row["gold_ids"])[1] for row in rows)
    return {
        "n": count,
        "top1": rate(rows),
        "parent_level": round(parent_level / count, 6) if count else None,
        "top1_top_family_only": rate(rows, "technique_pick_k1"),
        "top1_every_family": rate(rows, "technique_pick_all"),
        "top3": within(3),
        "top5": within(5),
        "gold_sub_technique": {"n": len(sub_technique), "top1": rate(sub_technique)},
        "gold_parent_only": {"n": len(parent_only), "top1": rate(parent_only)},
        "ece": expected_calibration_error(pairs),
    }


def technique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows whose ``pick`` is the exact answer (for :func:`common.score_rows` / ``paired_delta``)."""
    return [
        {**row, "pick": row.get("technique_pick"), "p_pick": row.get("technique_p")} for row in rows
    ]


def family_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        datasets.setdefault(str(row.get("dataset") or "all"), []).append(row)
    return {
        "pooled": family_block(rows),
        "datasets": {name: family_block(subset) for name, subset in sorted(datasets.items())},
    }


def references(
    cases: list[dict[str, Any]], compare: dict[str, Path]
) -> dict[str, list[dict[str, Any]]]:
    """Pick rows to compare against (their parent-aware score is family top-1): pipeline rank-1,
    Jev C1 as reported and under never-abstain where stored, and retrieve-then-pick arms' C1
    prediction files under never-abstain (amendment 4 of the parent prereg), or another family-head
    run's predictions as they are."""
    found = {"pipeline_rank1": reference_rows(cases, "pipeline")}
    jev = reference_rows(cases, "jev_c1")
    if jev:
        found["jev_c1"] = jev
        found["jev_c1_forced"] = reference_rows(cases, "jev_c1_forced")
    order = {
        case["case_id"]: [item["id"] for item in case["candidates"][:FLAT_K]] for case in cases
    }
    for name, path in compare.items():
        stored = {row["case_id"]: row for row in read_jsonl(path)}
        missing = [case["case_id"] for case in cases if case["case_id"] not in stored]
        if missing:
            raise SystemExit(f"{path} lacks {len(missing)} cases (first {missing[0]})")
        if any("top" in row for row in stored.values()):  # another family-head run: its picks
            found[name] = [stored[case["case_id"]] for case in cases]
        else:  # a C1 arm: never-abstain over its stored distribution
            found[name] = [
                forced_row(stored[case["case_id"]], order[case["case_id"]]) for case in cases
            ]
    return found


def report(
    cases: list[dict[str, Any]], rows: list[dict[str, Any]], compare: dict[str, Path]
) -> dict[str, Any]:
    out: dict[str, Any] = {"arm": family_metrics(rows), "references": {}, "vs": {}}
    found = references(cases, compare)
    for name, reference in found.items():
        scored = score_rows(reference)
        out["references"][name] = {
            "family_top1": scored["pooled"]["parent_aware"],
            "datasets": {key: value["parent_aware"] for key, value in scored["datasets"].items()},
        }
        delta = paired_delta(reference, rows)
        out["vs"][name] = {
            "n": delta["pooled"]["n"],
            "family_top1_delta": delta["pooled"]["delta"]["parent_aware"],
            "ci95": delta["pooled"]["ci95"]["parent_aware"] if delta["pooled"]["ci95"] else None,
            "datasets": {
                key: {
                    "delta": value["delta"]["parent_aware"],
                    "ci95": (value["ci95"] or {}).get("parent_aware"),
                }
                for key, value in delta["datasets"].items()
            },
        }
    if any("technique_pick" in row for row in rows):
        out.update(exact_report(rows, found))
    return out


def exact_report(
    rows: list[dict[str, Any]], found: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Exact top-1 against every reference that names techniques: the pipeline, Jev, the C1 arms
    and other technique-head runs (a family-only run names families and is left out)."""
    datasets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        datasets.setdefault(str(row.get("dataset") or "all"), []).append(row)
    mine = technique_rows(rows)
    parent_scores = score_rows(mine)
    out: dict[str, Any] = {
        "arm_exact": {
            "pooled": technique_block(rows),
            "datasets": {
                name: technique_block(subset) for name, subset in sorted(datasets.items())
            },
        },
        # the answered technique scored at parent level (sub-technique vs parent of the same
        # family forgiven; prereg cribl_recipe_base_2026-10-15.json)
        "arm_parent_level": {
            "pooled": parent_scores["pooled"]["parent_aware"],
            "datasets": {
                name: value["parent_aware"] for name, value in parent_scores["datasets"].items()
            },
        },
        "references_exact": {},
        "references_parent_level": {},
        "vs_exact": {},
    }
    for name, reference in found.items():
        if any("technique_pick" in row for row in reference):
            reference = technique_rows(reference)
        elif any("top" in row for row in reference):
            continue
        scored = score_rows(reference)["pooled"]
        out["references_exact"][name] = scored["exact"]
        out["references_parent_level"][name] = scored["parent_aware"]
        delta = paired_delta(reference, mine)
        ci = delta["pooled"]["ci95"] or {}
        out["vs_exact"][name] = {
            "n": delta["pooled"]["n"],
            "exact_delta": delta["pooled"]["delta"]["exact"],
            "ci95": ci.get("exact"),
            "parent_level_delta": delta["pooled"]["delta"]["parent_aware"],
            "parent_level_ci95": ci.get("parent_aware"),
            "datasets": {
                key: {
                    "delta": value["delta"]["exact"],
                    "ci95": (value["ci95"] or {}).get("exact"),
                    "parent_level_delta": value["delta"]["parent_aware"],
                    "parent_level_ci95": (value["ci95"] or {}).get("parent_aware"),
                }
                for key, value in delta["datasets"].items()
            },
        }
    return out


def train(args: argparse.Namespace) -> None:
    log_path = AUDITS / f"{args.arm}_train_log.json"
    models_dir = MODELS / args.arm
    if log_path.exists() or models_dir.exists():
        raise SystemExit(f"arm {args.arm!r} already has outputs ({log_path}, {models_dir})")
    families, texts, version = load_catalog(args.catalog)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows, skipped = [], 0
    for name in args.train_files:
        for row in read_jsonl(args.cases_dir / name):
            if family_targets(row["gold_ids"], families):
                rows.append(row)
            else:
                skipped += 1
    if args.limit_train:
        rng.shuffle(rows)
        rows = rows[: args.limit_train]
    dev = {name: read_jsonl(args.cases_dir / name) for name in args.dev_files}
    if args.dev_limit:
        dev = {name: cases[: args.dev_limit] for name, cases in dev.items()}

    model, tokenizer, info = load_model(
        args.init,
        trainable=True,
        base=args.base,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    encoder = Encoder(model, tokenizer, args.max_tokens, template=info["prompt_template"])
    if args.accum % args.batch:
        raise SystemExit("--accum must be a multiple of --batch")
    yesno = args.family_loss == "yesno"
    if yesno and args.margin:
        raise SystemExit("--margin is defined on the softmax family distribution only")
    model.eval()
    head = initial_head(encoder, families, texts, args.batch)
    if yesno:
        head.independent = True
        sample = [row["text"] for row in random.Random(args.seed).sample(rows, min(64, len(rows)))]
        offset = yes_no_offset(encoder, head, sample, args.batch)
        with torch.no_grad():
            head.bias.fill_(offset)
    techniques = None
    if args.technique_head:
        ids, group = technique_index(families, texts)
        techniques = Techniques(
            initial_technique_head(encoder, families, texts, args.batch),
            ids,
            torch.tensor(group, device=encoder.device),
            args.families_considered,
        )
    model.train()
    lora = [p for p in model.parameters() if p.requires_grad]
    heads = [*head.parameters(), *(techniques.head.parameters() if techniques else ())]
    optimizer = torch.optim.AdamW(
        [{"params": lora, "lr": args.lr}, {"params": heads, "lr": args.head_lr}],
        weight_decay=0.0,
    )
    peak = {id(group): group["lr"] for group in optimizer.param_groups}
    examples_total = int(len(rows) * args.epochs)
    total_steps = math.ceil(examples_total / args.accum)
    warmup = max(1, int(total_steps * args.warmup))
    every = len(rows) / args.checkpoints_per_epoch  # checkpoint k at round(k * every)
    models_dir.mkdir(parents=True)
    meta = {
        "families": families,
        "attack_version": version,
        "init": args.init,
        "base": args.base,
        "lora": info.get("lora"),
        "selection": args.selection,
        "margin": args.margin,
        "family_loss": args.family_loss,
        "negative_weight": args.negative_weight if yesno else None,
        "multi_label_weight": args.multi_label_weight,
        "yes_no_offset": round(float(head.bias[0]), 4) if yesno else None,
    }
    log: dict[str, Any] = {
        "arm": args.arm,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "args": {key: str(value) for key, value in vars(args).items() if key != "func"},
        "model": info,
        "families": len(families),
        "train_rows": len(rows),
        "train_rows_by_dataset": {
            name: sum(1 for row in rows if row["dataset"] == name)
            for name in sorted({row["dataset"] for row in rows})
        },
        "skipped_no_catalog_family": skipped,
        "train_rows_with_several_families": sum(
            1 for row in rows if len(family_targets(row["gold_ids"], families)) > 1
        ),
        "examples_total": examples_total,
        "optimizer_steps": total_steps,
        "checkpoints": [],
        "skipped_long": 0,
        "oom_skipped": 0,
    }
    if techniques:
        log["techniques"] = len(techniques.ids)
        log["train_rows_with_exact_targets"] = sum(
            1
            for row in rows
            if row["dataset"] not in args.family_only_datasets
            and technique_targets(row["gold_ids"], techniques.ids)
        )

    def save_log() -> None:
        log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

    def checkpoint(seen: int) -> None:
        directory = models_dir / f"step_{seen}"
        save_checkpoint(model, head, directory, meta, techniques)
        model.eval()
        head.eval()
        started = time.monotonic()
        scores, exact, multi = {}, {}, {}
        for name, cases in dev.items():
            split = name.removeprefix("cases_").removesuffix(".jsonl")
            path = AUDITS / f"{args.arm}_step{seen}_{split}.jsonl"
            predicted = with_oom_retry(
                lambda cases=cases, path=path: run_split(
                    encoder, head, families, cases, path, 0, args.batch, techniques
                ),
                note=f"dev scoring {split}",
            )
            scores[split] = family_block(predicted)
            if techniques:
                exact[split] = technique_block(predicted)
                multi[split] = multi_block(predicted, budget_threshold(predicted, REPORT_BUDGET))
        model.train()
        head.train()
        if args.selection == "parent":
            # parent-level accuracy of the answered technique (the family answer when there is
            # no technique head), averaged over the dev files
            selected_on = [
                exact[split]["parent_level"] if techniques else scores[split]["top1"]
                for split in scores
            ]
        else:
            selected_on = [
                exact[split]["top1"] if split in args.exact_selection else scores[split]["top1"]
                for split in scores
            ]
        log["checkpoints"].append(
            {
                "examples_seen": seen,
                "epoch": round(seen / len(rows), 3),
                "dir": str(directory),
                "dev": scores,
                **({"dev_exact": exact, "dev_multi": multi} if techniques else {}),
                "selection_score": round(sum(selected_on) / len(selected_on), 6),
                "scale": round(float(head.log_scale.exp()), 4),
                "dev_eval_s": round(time.monotonic() - started, 1),
            }
        )
        best = max(
            log["checkpoints"], key=lambda item: (item["selection_score"], -item["examples_seen"])
        )
        log["selected"] = best["dir"]
        save_log()
        print(
            f"checkpoint {seen}: "
            + " ".join(f"{split} top1 {s['top1']} top5 {s['top5']}" for split, s in scores.items())
            + "".join(
                f" {split} exact {s['top1']} parent {s['parent_level']}"
                for split, s in exact.items()
            )
            + "".join(
                f" {split} up-to-3 family {m['family_hit']} exact {m['exact_hit']}"
                for split, m in multi.items()
            ),
            flush=True,
        )

    checkpoint(0)  # the untrained head: nearest family description
    started = time.monotonic()
    seen = step = 0
    running: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    while seen < examples_total:
        order = list(rows)
        rng.shuffle(order)
        for start in range(0, len(order), args.batch):
            if seen >= examples_total:
                break
            chunk = order[start : start + args.batch][: examples_total - seen]
            ids, golds, exact_golds = [], [], []
            for row in chunk:
                try:
                    ids.append(encoder.token_ids(row["text"]))
                    golds.append(family_targets(row["gold_ids"], families))
                    exact_golds.append(
                        technique_targets(row["gold_ids"], techniques.ids)
                        if techniques and row["dataset"] not in args.family_only_datasets
                        else []
                    )
                except ValueError:
                    log["skipped_long"] += 1
            if ids:
                try:
                    hidden = encoder.hidden_batch(ids)
                    logits = head(hidden)
                    log_p = None if yesno else torch.log_softmax(logits, dim=-1)
                    within = (
                        within_family_log_probs(
                            techniques.head(hidden), techniques.group, len(families)
                        )
                        if techniques
                        else None
                    )
                    losses = torch.stack(
                        [
                            (args.multi_label_weight if len(gold) > 1 else 1.0)
                            * (
                                (
                                    yes_no(logits[i], gold, args.negative_weight)
                                    if yesno
                                    else -log_p[i, gold].mean()
                                )
                                + (
                                    args.margin[0] * rival_margin(log_p[i], gold, args.margin[1])
                                    if args.margin
                                    else 0.0
                                )
                                + (
                                    -args.technique_weight * within[i, exact_golds[i]].mean()
                                    if exact_golds[i]
                                    else 0.0
                                )
                            )
                            for i, gold in enumerate(golds)
                        ]
                    )
                    (losses.sum() / args.accum).backward()  # per-example losses, as with batch 1
                    running.extend(float(value) for value in losses.detach())
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    if not is_oom(error) or log["oom_skipped"] >= MAX_OOM_SKIPS:
                        raise
                    log_p = logits = losses = hidden = within = None
                    log["oom_skipped"] += 1
                    optimizer.zero_grad(set_to_none=True)
                    wait_after_oom(f"training examples {seen}+")
            before, seen = seen, seen + len(chunk)
            if seen // args.accum > before // args.accum or seen == examples_total:
                for group in optimizer.param_groups:
                    group["lr"] = peak[id(group)] * schedule(step, total_steps, warmup)
                torch.nn.utils.clip_grad_norm_([*lora, *heads], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 and running:
                    elapsed = time.monotonic() - started
                    mean = sum(running) / len(running)
                    log.setdefault("loss", []).append(
                        {
                            "step": step,
                            "examples_seen": seen,
                            "loss": round(mean, 5),
                            "elapsed_s": round(elapsed, 1),
                        }
                    )
                    running.clear()
                    print(
                        f"step {step}/{total_steps} loss {mean:.4f} {elapsed / seen:.3f} s/example, "
                        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                        flush=True,
                    )
            # the last boundary can fall a few examples before the end (7,761 rows: 23,280 of
            # 23,283), which used to save two near-identical final checkpoints
            boundary = round((math.floor(before / every) + 1) * every)
            if seen == examples_total or (before < boundary <= seen < examples_total):
                log["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log["train_s"] = round(time.monotonic() - started, 1)
                checkpoint(seen)
    log["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    save_log()
    print(f"selected on dev: {log['selected']}")


def evaluate(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.cases_dir / f"cases_{args.split}.jsonl")
    compare = {}
    for item in args.compare:
        name, _, path = item.partition("=")
        compare[name] = Path(path) if Path(path).is_absolute() else AUDITS / path
    model, encoder, head, info = load_checkpoint(args.checkpoint)
    families = info["families"]
    techniques = load_techniques(args.checkpoint, encoder.device)
    stem = f"{args.name}_{args.split}"
    started = time.monotonic()
    rows = with_oom_retry(
        lambda: run_split(
            encoder,
            head,
            families,
            cases,
            AUDITS / f"{stem}.jsonl",
            batch=args.batch,
            techniques=techniques,
        ),
        note=stem,
    )
    out = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "model": {key: value for key, value in info.items() if key != "families"},
        "families": len(families),
        "runtime_s": round(time.monotonic() - started, 1),
        **report(cases, rows, compare),
    }
    (AUDITS / f"{stem}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    arm = out["arm"]["pooled"]
    line = (
        f"{stem}: family top1 {arm['top1']} top3 {arm['top3']} top5 {arm['top5']} ECE {arm['ece']}"
    )
    for name, value in out["references"].items():
        delta = out["vs"][name]
        line += (
            f" | {name} {value['family_top1']} (delta {delta['family_top1_delta']} {delta['ci95']})"
        )
    print(line)
    if "arm_exact" in out:
        exact = out["arm_exact"]["pooled"]
        line = (
            f"{stem}: exact top1 {exact['top1']} (top family only {exact['top1_top_family_only']},"
            f" every family {exact['top1_every_family']}) top3 {exact['top3']} top5 {exact['top5']}"
            f" ECE {exact['ece']}"
        )
        for name, value in out["references_exact"].items():
            delta = out["vs_exact"][name]
            line += f" | {name} {value} (delta {delta['exact_delta']} {delta['ci95']})"
        print(line)
        line = f"{stem}: parent level {out['arm_parent_level']['pooled']}"
        for name, value in out["references_parent_level"].items():
            delta = out["vs_exact"][name]
            line += (
                f" | {name} {value} (delta {delta['parent_level_delta']}"
                f" {delta['parent_level_ci95']})"
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    fit = commands.add_parser("train")
    fit.set_defaults(func=train)
    fit.add_argument("--arm", required=True)
    fit.add_argument("--init", default="cribl", help="cribl | fresh | <adapter checkpoint dir>")
    fit.add_argument(
        "--base",
        choices=sorted(BASES),
        default=DEFAULT_BASE,
        help="the base model (decision_model.BASES); Cribl's adapter needs the default",
    )
    fit.add_argument("--lora-r", type=int, default=LORA_R, help="LoRA rank (--init fresh)")
    fit.add_argument("--lora-alpha", type=int, help="LoRA alpha (default 2 x rank)")
    fit.add_argument(
        "--selection",
        choices=SELECTION_RULES,
        default="legacy",
        help="legacy: family top-1, or exact top-1 on the --exact-selection splits; parent: the "
        "answered technique's parent-level accuracy on every dev file",
    )
    fit.add_argument("--catalog", type=Path, default=AUDITS / "catalog.json")
    fit.add_argument("--cases-dir", type=Path, default=AUDITS)
    fit.add_argument("--train-files", nargs="+", default=list(TRAIN_FILES))
    fit.add_argument("--dev-files", nargs="+", default=list(DEV_FILES))
    fit.add_argument("--epochs", type=float, default=3.0)
    fit.add_argument("--lr", type=float, default=1e-4, help="LoRA")
    fit.add_argument("--head-lr", type=float, default=1e-3)
    fit.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
    fit.add_argument("--batch", type=int, default=16, help="examples per forward pass")
    fit.add_argument(
        "--margin",
        nargs=2,
        type=float,
        metavar=("LAMBDA", "M"),
        help="add LAMBDA * max(0, M - (log P(gold) - log P(strongest wrong family))) to the loss",
    )
    fit.add_argument(
        "--technique-head",
        action="store_true",
        help="add the exact-technique head: P(technique) = P(family) x P(technique | family)",
    )
    fit.add_argument(
        "--families-considered",
        type=int,
        default=3,
        help="the exact answer comes from this many most probable families",
    )
    fit.add_argument(
        "--family-only-datasets",
        nargs="*",
        default=["annoctr"],
        help="datasets whose labels are parent-level by convention: no technique loss",
    )
    fit.add_argument("--technique-weight", type=float, default=1.0)
    fit.add_argument(
        "--multi-label-weight",
        type=float,
        default=1.0,
        help="loss weight of a training row labelled with two or more families (prereg "
        "cribl_multilabel_2026-10-14.json)",
    )
    fit.add_argument(
        "--family-loss",
        choices=("softmax", "yesno"),
        default="softmax",
        help="softmax: one answer shared out; yesno: each family's own sigmoid (losses.yes_no)",
    )
    fit.add_argument(
        "--negative-weight",
        type=float,
        default=0.1,
        help="yesno: weight of the 'no' term for every family a sentence is not labelled with",
    )
    fit.add_argument(
        "--exact-selection",
        nargs="*",
        default=[],
        help="dev splits scored on exact top-1 (not family top-1) in the selection score",
    )
    fit.add_argument("--warmup", type=float, default=0.03)
    fit.add_argument("--checkpoints-per-epoch", type=int, default=2)
    fit.add_argument("--max-tokens", type=int, default=1024)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--limit-train", type=int, help="pilot only")
    fit.add_argument("--dev-limit", type=int, help="pilot only")
    score = commands.add_parser("evaluate")
    score.set_defaults(func=evaluate)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--name", required=True)
    score.add_argument("--split", required=True, choices=EVAL_SPLITS)
    score.add_argument("--cases-dir", type=Path, default=AUDITS)
    score.add_argument("--compare", nargs="*", default=[], help="NAME=<C1 predictions .jsonl>")
    score.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
