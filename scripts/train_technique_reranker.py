from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings
from apt_classifier.evaluation import TechniqueEvaluationCase, pipeline_fingerprint
from apt_classifier.reranker import (
    MANIFEST_FILENAME,
    RERANKER_FEATURE_SCHEMA,
    RERANKER_MANIFEST_SCHEMA_VERSION,
    artifact_fingerprint,
    technique_pair_text,
)


@dataclass(frozen=True, slots=True)
class PairExample:
    case_id: str
    observation: str
    technique_id: str
    technique_text: str
    label: float
    negative_type: str | None = None


def load_cases(paths: list[Path]) -> list[TechniqueEvaluationCase]:
    cases = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if raw.get("unscorable_technique_ids") and not raw.get("expected_technique_ids"):
                continue
            cases.append(TechniqueEvaluationCase.from_dict(raw))
    return cases


def _normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.casefold()))


def _near_duplicate(left: frozenset[str], right: frozenset[str], threshold: float) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= threshold


def filter_cross_split_duplicates(
    train_cases: Sequence[TechniqueEvaluationCase],
    dev_cases: Sequence[TechniqueEvaluationCase],
    *,
    near_duplicate_threshold: float = 0.9,
) -> tuple[list[TechniqueEvaluationCase], list[TechniqueEvaluationCase], dict[str, int]]:
    """Remove exact/near duplicates while preserving the held-out dev split.

    Dev is deduplicated internally first.  Train examples matching any retained
    dev example are then removed, so calibration data is never discarded in
    favour of training data.  Input order is preserved and the result is fully
    deterministic.
    """
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")

    def unique(cases: Sequence[TechniqueEvaluationCase]) -> tuple[list[Any], int]:
        retained: list[TechniqueEvaluationCase] = []
        token_sets: list[frozenset[str]] = []
        removed = 0
        for case in cases:
            tokens = _normalized_tokens(case.text)
            if any(
                _near_duplicate(tokens, previous, near_duplicate_threshold)
                for previous in token_sets
            ):
                removed += 1
                continue
            retained.append(case)
            token_sets.append(tokens)
        return [(case, tokens) for case, tokens in zip(retained, token_sets, strict=True)], removed

    dev_unique, dev_internal_removed = unique(dev_cases)
    train_unique, train_internal_removed = unique(train_cases)
    dev_tokens = [tokens for _, tokens in dev_unique]
    filtered_train = []
    cross_split_removed = 0
    for case, tokens in train_unique:
        if any(
            _near_duplicate(tokens, held_out, near_duplicate_threshold) for held_out in dev_tokens
        ):
            cross_split_removed += 1
        else:
            filtered_train.append(case)
    return (
        filtered_train,
        [case for case, _ in dev_unique],
        {
            "train_internal_removed": train_internal_removed,
            "dev_internal_removed": dev_internal_removed,
            "cross_split_train_removed": cross_split_removed,
        },
    )


def build_pairs(
    cases: list[TechniqueEvaluationCase],
    kb: AttackKnowledgeBase,
    *,
    negatives_per_positive: int = 3,
    seed: int = 7,
    store: Any = None,
    lexical_index: Any = None,
) -> list[PairExample]:
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    technique_ids = sorted(kb.techniques)
    technique_id_set = set(technique_ids)
    pairs: list[PairExample] = []
    for case in cases:
        expected = set(case.expected_technique_ids) & technique_id_set
        retrieval_negs = (
            _retrieval_negatives(case, expected, kb, store, lexical_index)
            if store is not None and lexical_index is not None
            else []
        )
        if expected:
            for technique_id in sorted(expected):
                pairs.append(
                    PairExample(
                        case.case_id,
                        case.text,
                        technique_id,
                        _technique_text(kb, technique_id),
                        1.0,
                    )
                )
            # Actual hybrid false positives are highest priority, followed by
            # sibling/hierarchy confounders and deterministic random fill.
            negative_ids = _unique_negatives(
                [*retrieval_negs, *_hard_negatives(case, expected, kb)], expected
            )
            if len(negative_ids) < negatives_per_positive:
                negative_ids.extend(
                    _sample_ids(
                        case.case_id,
                        technique_ids,
                        excluded=expected | {item[0] for item in negative_ids},
                        count=negatives_per_positive - len(negative_ids),
                        seed=seed,
                    )
                )
            for technique_id, negative_type in negative_ids[:negatives_per_positive]:
                pairs.append(
                    PairExample(
                        case.case_id,
                        case.text,
                        technique_id,
                        _technique_text(kb, technique_id),
                        0.0,
                        negative_type,
                    )
                )
        else:
            # Benign/zero-label observations should use the candidates the
            # production retriever would actually surface whenever possible.
            negative_ids = _unique_negatives(retrieval_negs, set())
            if len(negative_ids) < negatives_per_positive:
                negative_ids.extend(
                    _sample_ids(
                        case.case_id,
                        technique_ids,
                        excluded={item[0] for item in negative_ids},
                        count=negatives_per_positive - len(negative_ids),
                        seed=seed,
                    )
                )
            for technique_id, negative_type in negative_ids[:negatives_per_positive]:
                pairs.append(
                    PairExample(
                        case.case_id,
                        case.text,
                        technique_id,
                        _technique_text(kb, technique_id),
                        0.0,
                        f"zero_label_{negative_type}",
                    )
                )
    random.Random(seed).shuffle(pairs)
    return pairs


def _unique_negatives(
    values: Sequence[tuple[str, str]], excluded: set[str]
) -> list[tuple[str, str]]:
    output = []
    seen = set(excluded)
    for technique_id, negative_type in values:
        if technique_id not in seen:
            output.append((technique_id, negative_type))
            seen.add(technique_id)
    return output


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


MAX_CALIBRATION_GRID = 201


def _threshold_candidates(values: list[float]) -> list[float]:
    """Unique scores plus midpoints when few, otherwise a quantile grid (bounded size)."""
    unique_scores = sorted(set(values))
    candidates = sorted(
        set(
            [0.0, 1.0, *unique_scores]
            + [
                (left + right) / 2.0
                for left, right in zip(unique_scores, unique_scores[1:], strict=False)
            ]
        )
    )
    if len(candidates) <= 2 * MAX_CALIBRATION_GRID:
        return candidates
    import numpy as np

    grid = np.quantile(np.asarray(values), np.linspace(0.0, 1.0, MAX_CALIBRATION_GRID))
    return sorted({0.0, 1.0, *(float(value) for value in grid)})


PROMOTE_PRECISION_FLOOR = 0.85
DEMOTE_PRECISION_FLOOR = 0.90


def calibrate_thresholds(
    scores: Sequence[float],
    labels: Sequence[float],
    *,
    promote_precision_floor: float = PROMOTE_PRECISION_FLOOR,
    demote_precision_floor: float = DEMOTE_PRECISION_FLOOR,
) -> dict[str, Any]:
    """Select a promote/demote *band* from held-out dev scores.

    The reranker is only allowed to intervene where it is reliable, so the two thresholds
    are chosen by precision floors rather than by a single accuracy-style objective:

    - ``promote_threshold`` is the lowest threshold at which the precision of "score >= t"
      on dev positives is at least ``promote_precision_floor`` (maximum promotion recall
      under the floor);
    - ``demote_threshold`` is the highest threshold at which the fraction of dev negatives
      among "score <= t" is at least ``demote_precision_floor``;
    - the band is kept open (``demote < promote``); if the floors cannot be met the
      corresponding side is disabled (promote 1.0 / demote 0.0) so the reranker abstains.

    The earlier objective (balanced F1, then coverage) collapsed to a single threshold
    because full coverage always won the tie-break, which let the reranker decide every
    candidate. Counts use binary search over sorted score arrays (fast for ~5k pairs).
    """
    if len(scores) != len(labels) or not scores:
        raise ValueError("calibration scores and labels must be non-empty and have equal length")
    values = [float(score) for score in scores]
    binary = [int(float(label) >= 0.5) for label in labels]
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in values):
        raise ValueError("calibration scores must be finite values in [0, 1]")
    if len(set(binary)) != 2:
        raise ValueError("calibration requires both positive and negative dev examples")

    import numpy as np

    positives = np.sort(np.asarray([v for v, b in zip(values, binary, strict=True) if b == 1]))
    negatives = np.sort(np.asarray([v for v, b in zip(values, binary, strict=True) if b == 0]))
    total = len(values)
    candidates = _threshold_candidates(values)
    grid = np.asarray(candidates)
    pos_ge = len(positives) - np.searchsorted(positives, grid, side="left")
    neg_ge = len(negatives) - np.searchsorted(negatives, grid, side="left")
    pos_le = np.searchsorted(positives, grid, side="right")
    neg_le = np.searchsorted(negatives, grid, side="right")

    promote_index: int | None = None
    for index in range(len(candidates)):
        decided = int(pos_ge[index] + neg_ge[index])
        if (
            decided
            and pos_ge[index] / decided >= promote_precision_floor
            and candidates[index] > 0.0
        ):
            promote_index = index
            break
    demote_index: int | None = None
    for index in range(len(candidates) - 1, -1, -1):
        decided = int(pos_le[index] + neg_le[index])
        if (
            decided
            and neg_le[index] / decided >= demote_precision_floor
            and candidates[index] < 1.0
        ):
            demote_index = index
            break
    promote = candidates[promote_index] if promote_index is not None else 1.0
    demote = candidates[demote_index] if demote_index is not None else 0.0
    if demote >= promote:
        # Keep an open band: the demote side yields to the promote side.
        lower = [index for index in range(len(candidates)) if candidates[index] < promote]
        demote_index = max(
            (
                index
                for index in lower
                if (pos_le[index] + neg_le[index])
                and neg_le[index] / (pos_le[index] + neg_le[index]) >= demote_precision_floor
            ),
            default=None,
        )
        demote = candidates[demote_index] if demote_index is not None else 0.0

    def _at(array, index):
        return int(array[index]) if index is not None else 0

    promote_tp = _at(pos_ge, promote_index)
    promote_fp = _at(neg_ge, promote_index)
    promote_fn = len(positives) - promote_tp
    demote_tn = _at(neg_le, demote_index)
    demote_fp = _at(pos_le, demote_index)
    demote_fn = len(negatives) - demote_tn
    positive_f1 = _f1(promote_tp, promote_fp, promote_fn)
    negative_f1 = _f1(demote_tn, demote_fp, demote_fn)
    coverage = (promote_tp + promote_fp + demote_tn + demote_fp) / total
    return {
        "method": "precision_floor_band_dev",
        "promote_threshold": float(promote),
        "demote_threshold": float(demote),
        "promote_precision_floor": promote_precision_floor,
        "demote_precision_floor": demote_precision_floor,
        "dev_pair_count": total,
        "held_out_metrics": {
            "balanced_f1": (positive_f1 + negative_f1) / 2.0,
            "positive_f1": positive_f1,
            "negative_f1": negative_f1,
            "promote_precision": promote_tp / (promote_tp + promote_fp)
            if promote_tp + promote_fp
            else 0.0,
            "demote_precision": demote_tn / (demote_tn + demote_fp)
            if demote_tn + demote_fp
            else 0.0,
            "decision_coverage": coverage,
            "promote_true_positive": promote_tp,
            "promote_false_positive": promote_fp,
            "promote_false_negative": promote_fn,
            "demote_true_negative": demote_tn,
            "demote_false_positive": demote_fp,
            "demote_false_negative": demote_fn,
        },
    }


def _pairs_hash(pairs: Sequence[PairExample]) -> str:
    records = sorted(
        (asdict(pair) for pair in pairs),
        key=lambda value: (
            value["case_id"],
            value["observation"],
            value["technique_id"],
            value["label"],
            value["negative_type"] or "",
        ),
    )
    return pipeline_fingerprint({"pairs": records})


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    material = dict(manifest)
    material["artifact_fingerprint"] = artifact_fingerprint(material)
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(material, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest.clear()
    manifest.update(material)


def train(
    pairs: list[PairExample],
    *,
    dev_pairs: list[PairExample],
    attack_version: str,
    base_model: str,
    revision: str,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    if not revision:
        raise ValueError("A pinned base-model revision is required")
    if attack_version.strip().lower() in {"", "latest", "unknown"}:
        raise ValueError("A pinned ATT&CK version is required")
    if not pairs or not dev_pairs:
        raise ValueError("Non-empty train and held-out dev pairs are required")

    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    random.seed(seed)
    torch.manual_seed(seed)
    model = CrossEncoder(
        base_model,
        num_labels=1,
        max_length=RERANKER_FEATURE_SCHEMA["max_length"],
        revision=revision,
        trust_remote_code=False,
    )
    examples = [
        InputExample(texts=[pair.observation, pair.technique_text], label=pair.label)
        for pair in pairs
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, generator=generator)
    output.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=max(round(len(loader) * epochs * 0.1), 1),
        optimizer_params={"lr": learning_rate},
        output_path=str(output),
        use_amp=torch.cuda.is_available(),
    )
    # Persist the weights explicitly: the library's save callback only fires on evaluator
    # events, and an empty output directory after a completed fit is otherwise silent.
    model.save(str(output))
    dev_scores = model.predict(
        [(pair.observation, pair.technique_text) for pair in dev_pairs],
        batch_size=batch_size,
        show_progress_bar=False,
    )
    calibration = calibrate_thresholds(
        [float(score) for score in dev_scores], [pair.label for pair in dev_pairs]
    )
    calibration["dev_data_hash"] = _pairs_hash(dev_pairs)
    manifest = {
        "schema_version": RERANKER_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "apt_classifier_technique_reranker",
        "attack_version": attack_version,
        "base_model": base_model,
        "base_model_revision": revision,
        "feature_schema": RERANKER_FEATURE_SCHEMA,
        "training_data_hash": _pairs_hash(pairs),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "pair_count": len(pairs),
            "positive_count": sum(pair.label == 1.0 for pair in pairs),
            "negative_count": sum(pair.label == 0.0 for pair in pairs),
        },
        "calibration": calibration,
    }
    _write_manifest(output, manifest)
    return manifest


def _hard_negatives(
    case: TechniqueEvaluationCase,
    expected: set[str],
    kb: AttackKnowledgeBase,
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    seen = set(expected)
    expected_items = [kb.techniques[technique_id] for technique_id in expected]
    parents = {item.parent_id for item in expected_items if item.parent_id}
    tactics = {tactic for item in expected_items for tactic in item.tactics}
    for technique_id, technique in sorted(kb.techniques.items()):
        if technique_id in seen:
            continue
        negative_type = None
        if technique.parent_id and technique.parent_id in parents:
            negative_type = "sibling"
        elif technique_id in parents or technique_id.split(".", 1)[0] in expected:
            negative_type = "hierarchy"
        elif tactics & set(technique.tactics):
            negative_type = "same_tactic"
        if negative_type:
            output.append((technique_id, negative_type))
            seen.add(technique_id)
    digest = int(hashlib.sha256(case.case_id.encode()).hexdigest()[:16], 16)
    random.Random(digest).shuffle(output)
    return output


def _retrieval_negatives(
    case: TechniqueEvaluationCase,
    expected: set[str],
    kb: AttackKnowledgeBase,
    store: Any,
    lexical_index: Any,
    *,
    top_k: int = 20,
) -> list[tuple[str, str]]:
    """Mine false candidates from the production hybrid retrieval pool."""
    from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
    from apt_classifier.retrieve import retrieve_techniques

    ledger = EvidenceLedger(
        observations=[Observation(text=case.text, provenance=Provenance(source="training"))]
    )
    try:
        result = retrieve_techniques(
            ledger,
            store,
            kb,
            threshold=0.30,
            top_k=top_k,
            retrieval_mode="hybrid",
            candidate_pool=top_k,
            lexical_index=lexical_index,
            acceptance_threshold=0.50,
            relative_margin=0.05,
        )
    except Exception:
        return []

    negatives: list[tuple[str, str]] = []
    seen = set(expected)
    for candidate in result.candidates:
        technique_id = candidate.technique_id
        if technique_id not in seen:
            negatives.append((technique_id, "retrieval_mined"))
            seen.add(technique_id)
    return negatives[:top_k]


def _sample_ids(
    case_id: str,
    technique_ids: list[str],
    *,
    excluded: set[str],
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    candidates = [technique_id for technique_id in technique_ids if technique_id not in excluded]
    digest = int(hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()[:16], 16)
    random.Random(digest).shuffle(candidates)
    return [(technique_id, "random") for technique_id in candidates[:count]]


def _technique_text(kb: AttackKnowledgeBase, technique_id: str) -> str:
    # Single shared definition with inference (apt_classifier.reranker.technique_pair_text).
    return technique_pair_text(kb, technique_id)


def _save_pairs_cache(path: Path, pairs: list[PairExample], dev_pairs: list[PairExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [asdict(pair) for pair in pairs],
        "dev": [asdict(pair) for pair in dev_pairs],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_pairs_cache(path: Path | None) -> tuple[list[PairExample], list[PairExample]] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        [PairExample(**item) for item in payload["train"]],
        [PairExample(**item) for item in payload["dev"]],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a local ATT&CK technique cross-encoder")
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--dev", type=Path, nargs="+", required=True, help="Held-out dev cases")
    parser.add_argument("--kb", type=Path, default=Path("data/kb/attack_kb.json"))
    parser.add_argument("--output", type=Path, default=Path("data/models/technique-reranker"))
    parser.add_argument("--base-model")
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pairs-cache",
        type=Path,
        help="JSON file to save mined train/dev pairs to (or load from when it exists), so a "
        "failed training run does not repeat retrieval-negative mining",
    )
    parser.add_argument(
        "--use-retrieval-negatives",
        action="store_true",
        help="Mine negatives from actual retrieval results (requires ChromaDB)",
    )
    args = parser.parse_args()
    settings = Settings()
    kb = AttackKnowledgeBase.load(args.kb)
    train_cases, dev_cases, duplicate_stats = filter_cross_split_duplicates(
        load_cases(args.train),
        load_cases(args.dev),
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    if not train_cases or not dev_cases:
        raise ValueError("Duplicate filtering left an empty train or dev split")

    store = None
    lexical_index = None
    if args.use_retrieval_negatives:
        from apt_classifier.embeddings import SentenceTransformerEmbedder
        from apt_classifier.retrieve import LexicalTechniqueIndex
        from apt_classifier.vectorstore import ChromaVectorStore

        embedder = SentenceTransformerEmbedder.from_settings(settings)
        store = ChromaVectorStore(settings=settings, embedder=embedder)
        lexical_index = LexicalTechniqueIndex.from_kb(kb)
    cached = _load_pairs_cache(args.pairs_cache)
    if cached is not None:
        pairs, dev_pairs = cached
        print(f"loaded {len(pairs)} train / {len(dev_pairs)} dev pairs from {args.pairs_cache}")
    else:
        pairs = build_pairs(
            train_cases,
            kb,
            negatives_per_positive=args.negatives,
            seed=args.seed,
            store=store,
            lexical_index=lexical_index,
        )
        dev_pairs = build_pairs(
            dev_cases,
            kb,
            negatives_per_positive=args.negatives,
            seed=args.seed,
            store=store,
            lexical_index=lexical_index,
        )
        if args.pairs_cache is not None:
            _save_pairs_cache(args.pairs_cache, pairs, dev_pairs)
            print(f"saved mined pairs to {args.pairs_cache}")
    summary: dict[str, Any] = {
        "train_cases": len(train_cases),
        "dev_cases": len(dev_cases),
        "train_pairs": len(pairs),
        "dev_pairs": len(dev_pairs),
        "positive": sum(pair.label == 1.0 for pair in pairs),
        "negative": sum(pair.label == 0.0 for pair in pairs),
        "negative_types": {},
        "duplicate_filtering": duplicate_stats,
    }
    for pair in pairs:
        if pair.negative_type:
            summary["negative_types"][pair.negative_type] = (
                summary["negative_types"].get(pair.negative_type, 0) + 1
            )
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    base_model = args.base_model or settings.embedding_model
    if args.revision:
        revision = args.revision
    elif args.base_model:
        raise ValueError("--revision is required when --base-model is supplied")
    else:
        revision = settings.embedding_model_revision
    manifest = train(
        pairs,
        dev_pairs=dev_pairs,
        attack_version=kb.attack_version,
        base_model=base_model,
        revision=revision,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps({"summary": summary, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
