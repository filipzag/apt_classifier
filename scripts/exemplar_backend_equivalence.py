"""Equivalence of the exemplar backends: the npz library against its Chroma copy.

Levels (pass criteria are bitwise; any difference fails the level):

``L1``  ``search=exact``: every ``ExemplarVotes`` field of the Chroma store equals the npz's,
        bit for bit, on every query set.
``L2``  ``search=hnsw`` recall: each query's exact window (``2k+16``) is inside Chroma's top
        ``ceil(window * overfetch)`` (``recall_misses == 0``), and again at ``--wide-k`` (256 by
        default, window 528) on ``--wide-k-sets``; Chroma's raw recall@window is reported.
``L3``  ``search=hnsw`` votes: bit for bit equal to the npz's (any exception is listed).
``L4``  the pipeline: arms ``npz``, ``npz_repeat`` (the control), ``chroma_exact`` and
        ``chroma_hnsw`` through ``retrieve_techniques`` (``scripts/benchmark_mapping.run``) with the
        configured acceptance model, cues on and off, on ``--l4-sets``: candidates, fused ranks,
        states, acceptance probabilities (bitwise), accepted sets and metrics equal the npz arm's;
        the model is ``active`` with the library's fingerprint in every arm (loaded under the
        runtime retrieval contract, and scoring every case: not refused per call, with stored
        probabilities) and the Chroma fallback counter stays 0.

Query sets: ``self`` (the exemplar vectors themselves), ``public:dev``, ``public:test``,
``cti_hal`` (statements) and ``evidenceforge:<name>`` for bundles given with
``--evidenceforge-bundle NAME DIR SCENARIO`` (structured records are queried with their values,
as at runtime). The clinic bundle and the Elastic posts are refused: they are run once, last.

Every text is embedded once and cached in ``--vector-cache`` (an ``.npz`` keyed by text hash;
never commit it, CTI-HAL has no licence). A caching embedder is swapped into the vector store, so
the dense and exemplar paths of every arm see identical vectors and the arms differ only in the
exemplar backend. Read-only towards Chroma: ``get_collection``, ``count``, ``get`` and ``query``.

    PYTHONPATH=. .venv/bin/python scripts/exemplar_backend_equivalence.py --collection NAME \\
        --sets self public:dev public:test cti_hal --levels L1 L2 L3 L4 \\
        --vector-cache data/audits/chroma_exemplars/cache/equivalence_vectors.npz \\
        --output data/audits/chroma_exemplars/runs/R11_equivalence.json

Exit 0 when every requested level passes, 1 otherwise, 2 when refused.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.exemplars import (  # noqa: E402
    ExemplarIndex,
    ExemplarVotes,
    exemplar_query_text,
    window_size,
)

LEVELS = ("L1", "L2", "L3", "L4")
DEFAULT_SETS = ("self", "public:dev", "public:test", "cti_hal")
DEFAULT_L4_SETS = ("public:dev", "public:test", "cti_hal")
REFUSED_MARKERS = ("clinic", "elastic")
VOTE_FIELDS = (
    "top_similarity",
    "votes",
    "family",
    "max_similarity",
    "counts",
    "ranks",
    "best_text",
    "best_source",
    "neighbours",
    "excluded_duplicates",
    "k",
)
# rows of benchmark_mapping.run compared between arms (the fingerprints differ by design)
COMPARED_CASE_KEYS = (
    "case_id",
    "ranked_technique_ids",
    "accepted_technique_ids",
    "expected_in_pool",
    "expected_accepted",
    "candidates",
)


# ----- bitwise comparison -----
def _bits(value: Any) -> Any:
    """A float's IEEE bytes (so -0.0, NaN and 1 ulp differences are visible), recursively."""
    if isinstance(value, bool) or value is None or isinstance(value, int | str):
        return value
    if isinstance(value, float | np.floating):
        return np.float64(value).tobytes().hex()
    if isinstance(value, dict):
        return [(key, _bits(item)) for key, item in value.items()]  # insertion order counts
    if isinstance(value, list | tuple):
        return [_bits(item) for item in value]
    return repr(value)


def vote_differences(reference: ExemplarVotes, candidate: ExemplarVotes) -> list[str]:
    """Names of the ``ExemplarVotes`` fields that are not bit-identical."""
    return [
        name
        for name in VOTE_FIELDS
        if _bits(getattr(reference, name)) != _bits(getattr(candidate, name))
    ]


def compare_vote_lists(
    reference: Sequence[ExemplarVotes],
    candidate: Sequence[ExemplarVotes],
    *,
    labels: Sequence[str] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        return {"queries": len(reference), "length_mismatch": len(candidate), "differences": -1}
    differing = []
    for position, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        fields = vote_differences(left, right)
        if fields:
            differing.append({"query": labels[position] if labels else position, "fields": fields})
    return {
        "queries": len(reference),
        "differences": len(differing),
        "examples": differing[:limit],
    }


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    array = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95))}


def batched_votes(
    source: Any,
    vectors: np.ndarray,
    *,
    batch_size: int,
    timings: list[float] | None = None,
    **arguments: Any,
) -> list[ExemplarVotes]:
    """Votes in fixed batches (the similarity gemm depends on the batch, so every arm must be
    queried with the same batches)."""
    votes: list[ExemplarVotes] = []
    for start in range(0, len(vectors), max(1, int(batch_size))):
        started = time.perf_counter()
        votes.extend(source.query_vectors(vectors[start : start + batch_size], **arguments))
        if timings is not None:
            timings.append(time.perf_counter() - started)
    return votes


def vote_level(
    reference: Any,
    candidate: Any,
    vectors: np.ndarray,
    *,
    batch_size: int,
    labels: Sequence[str] | None = None,
    limit: int = 20,
    **arguments: Any,
) -> dict[str, Any]:
    """L1 / L3: the candidate backend's votes against the reference's, bit for bit."""
    reference_times: list[float] = []
    candidate_times: list[float] = []
    left = batched_votes(
        reference, vectors, batch_size=batch_size, timings=reference_times, **arguments
    )
    right = batched_votes(
        candidate, vectors, batch_size=batch_size, timings=candidate_times, **arguments
    )
    summary = compare_vote_lists(left, right, labels=labels, limit=limit)
    summary["passed"] = summary["differences"] == 0
    summary["seconds_per_batch"] = {
        "reference": _percentiles(reference_times),
        "candidate": _percentiles(candidate_times),
    }
    return summary


def recall_level(
    index: ExemplarIndex,
    store: Any,
    vectors: np.ndarray,
    *,
    k: int,
    batch_size: int,
) -> dict[str, Any]:
    """L2: is each query's exact window inside Chroma's candidates, and Chroma's raw
    recall@window (its own top-window against the exact window)."""
    window = window_size(len(index), k)
    contained = 0
    raw: list[float] = []
    n_results = 0
    for start in range(0, len(vectors), max(1, int(batch_size))):
        chunk = vectors[start : start + batch_size]
        exact = index.exact_windows(chunk, k=k)
        found = store.candidate_rows(chunk, k=k)
        for exact_rows, chroma_rows in zip(exact, found, strict=True):
            n_results = max(n_results, len(chroma_rows))
            contained += bool(np.isin(exact_rows, chroma_rows).all())
            raw.append(len(np.intersect1d(exact_rows, chroma_rows[:window])) / max(1, window))
    queries = len(raw)
    return {
        "k": int(k),
        "window": window,
        "n_results": n_results,
        "queries": queries,
        "recall_misses": queries - contained,
        "raw_recall_at_window_mean": float(np.mean(raw)) if raw else None,
        "raw_recall_at_window_min": float(np.min(raw)) if raw else None,
        "passed": queries == contained,
    }


def compare_mapping_runs(
    base: dict[str, Any], other: dict[str, Any], *, limit: int = 20
) -> dict[str, Any]:
    """L4: two ``benchmark_mapping.run`` blocks case by case (candidate rows, accepted sets)
    and metric by metric; probabilities are compared as stored (unrounded floats)."""
    base_cases = {row["case_id"]: row for row in base.get("cases", [])}
    other_cases = {row["case_id"]: row for row in other.get("cases", [])}
    differing = []
    for case_id in sorted(set(base_cases) | set(other_cases)):
        left, right = base_cases.get(case_id), other_cases.get(case_id)
        if left is None or right is None:
            differing.append({"case_id": case_id, "fields": ["missing"]})
            continue
        fields = [
            key for key in COMPARED_CASE_KEYS if _bits(left.get(key)) != _bits(right.get(key))
        ]
        if fields:
            differing.append({"case_id": case_id, "fields": fields})
    metrics_equal = _bits((base.get("pooled") or {}).get("metrics")) == _bits(
        (other.get("pooled") or {}).get("metrics")
    ) and all(
        _bits((block or {}).get("metrics"))
        == _bits(((other.get("datasets") or {}).get(name) or {}).get("metrics"))
        for name, block in (base.get("datasets") or {}).items()
    )
    return {
        "cases": len(base_cases),
        "differences": len(differing),
        "examples": differing[:limit],
        "metrics_equal": metrics_equal,
        "passed": not differing and metrics_equal,
    }


# ----- cached embedding -----
class CachedEmbedder:
    """Embeds each distinct text once and returns the same floats every time it is asked.

    Vectors are kept as float32 and returned as lists (what the real embedder returns), keyed by
    the SHA-256 of the text, and persisted to ``path``. The real embedder is built only when a
    text is missing from the cache."""

    def __init__(self, factory: Callable[[], Any] | None, path: Path | None = None) -> None:
        self._factory = factory
        self._inner: Any = None
        self.path = path
        self._vectors: dict[str, np.ndarray] = {}
        self.loaded = 0
        self.computed = 0
        if path is not None and Path(path).is_file():
            with np.load(path, allow_pickle=False) as payload:
                for key, vector in zip(payload["keys"], payload["vectors"], strict=True):
                    self._vectors[str(key)] = np.asarray(vector, dtype=np.float32)
            self.loaded = len(self._vectors)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str], batch_size: int | None = None, **_: Any) -> list:
        values = [str(text) for text in texts]
        missing = list(
            dict.fromkeys(text for text in values if self.key(text) not in self._vectors)
        )
        if missing:
            if self._factory is None:
                raise RuntimeError(f"{len(missing)} texts are not in the vector cache")
            if self._inner is None:
                self._inner = self._factory()
            fresh = np.asarray(self._inner.embed(missing), dtype=np.float32)
            for text, vector in zip(missing, fresh, strict=True):
                self._vectors[self.key(text)] = vector
            self.computed += len(missing)
        return [self._vectors[self.key(text)].tolist() for text in values]

    def vectors(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.embed(texts), dtype=np.float32)

    def save(self) -> None:
        if self.path is None or not self._vectors:
            return
        keys = sorted(self._vectors)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(self.path).with_suffix(".tmp.npz")
        np.savez(
            temporary,
            keys=np.asarray(keys),
            vectors=np.stack([self._vectors[key] for key in keys]),
        )
        temporary.replace(self.path)


# ----- query sets -----
def refused_set(name: str, bundle: Sequence[str] | None = None) -> str | None:
    text = " ".join([name, *(bundle or ())]).casefold()
    for marker in REFUSED_MARKERS:
        if marker in text:
            return f"{name}: the {marker} data are evaluated once, last, never here"
    return None


def evidenceforge_cases(bundle_dir: Path, scenario: Path, *, window_seconds: int = 120) -> list:
    from scripts.evidenceforge_eval import (
        _load_native_records,
        build_narrative_case_set,
        build_source_case_set,
        build_technique_mapping,
        load_ground_truth,
    )

    events, intent_status = load_ground_truth(Path(bundle_dir) / "GROUND_TRUTH.json")
    intents = build_technique_mapping(Path(scenario))
    windows_records, ecar_records = _load_native_records(Path(bundle_dir))
    cases = list(build_narrative_case_set(events, intents, intent_status).cases)
    for source, records in (("windows", windows_records), ("ecar", ecar_records)):
        cases.extend(
            build_source_case_set(
                events,
                intents,
                records,
                source=source,
                intent_status=intent_status,
                window_seconds=window_seconds,
            ).cases
        )
    return cases


def load_set_cases(name: str, *, kb: Any, bundles: dict[str, tuple[Path, Path]]) -> list:
    from apt_classifier.vectorstore import procedure_sentence_texts
    from scripts.benchmark_mapping import DEFAULT_DATASETS, DEFAULT_ROOT, load_cases

    indexed = procedure_sentence_texts(kb)
    if name in ("public:dev", "public:test"):
        cases, _ = load_cases(
            DEFAULT_ROOT,
            datasets=set(DEFAULT_DATASETS),
            split=name.split(":", 1)[1],
            indexed_texts=indexed,
        )
        return cases
    if name == "cti_hal":
        cases, _ = load_cases(
            REPO_ROOT / "data" / "benchmarks" / "cti_hal",
            datasets={"cti_hal"},
            split="test",
            indexed_texts=indexed,
        )
        return cases
    if name.startswith("evidenceforge:"):
        bundle_dir, scenario = bundles[name.split(":", 1)[1]]
        return evidenceforge_cases(bundle_dir, scenario)
    raise ValueError(f"unknown query set {name!r}")


# ----- the run -----
def _store_settings(settings: Any, collection: str, search: str) -> Any:
    return settings.model_copy(
        update={
            "technique_exemplar_backend": "chroma",
            "technique_exemplar_collection": collection,
            "technique_exemplar_search": search,
        }
    )


def run_vote_levels(
    npz: ExemplarIndex,
    exact_store: Any,
    hnsw_store: Any,
    vectors_by_set: dict[str, np.ndarray],
    *,
    levels: Sequence[str],
    parents: dict[str, str | None],
    k: int,
    tau: float,
    duplicate_similarity: float,
    batch_size: int,
    wide_k: int | None = None,
    wide_k_sets: Sequence[str] = (),
    limit: int = 20,
) -> dict[str, Any]:
    """L1-L3 over the query sets (the testable core; no Chroma writes, no embedding)."""
    arguments = {
        "parent_of": parents,
        "k": k,
        "tau": tau,
        "duplicate_similarity": duplicate_similarity,
    }
    output: dict[str, Any] = {}
    for name, vectors in vectors_by_set.items():
        if "L1" in levels:
            output.setdefault("L1", {})[name] = vote_level(
                npz, exact_store, vectors, batch_size=batch_size, limit=limit, **arguments
            )
        if "L2" in levels:
            entry = {
                "default_k": recall_level(npz, hnsw_store, vectors, k=k, batch_size=batch_size)
            }
            if wide_k and name in wide_k_sets:
                entry["wide_k"] = recall_level(
                    npz, hnsw_store, vectors, k=wide_k, batch_size=batch_size
                )
            entry["passed"] = all(item["passed"] for item in entry.values())
            output.setdefault("L2", {})[name] = entry
        if "L3" in levels:
            output.setdefault("L3", {})[name] = vote_level(
                npz, hnsw_store, vectors, batch_size=batch_size, limit=limit, **arguments
            )
    return output


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    from apt_classifier.acceptance import load_acceptance_model
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.embeddings import SentenceTransformerEmbedder
    from apt_classifier.evaluation import write_artifact
    from apt_classifier.exemplar_store import (
        clear_memo,
        configured_collection_name,
        load_chroma_exemplar_store,
        query_totals,
    )
    from apt_classifier.exemplars import parent_map
    from apt_classifier.retrieve import (
        LexicalTechniqueIndex,
        lexical_fields,
        retrieval_contract_for,
    )
    from apt_classifier.vectorstore import ChromaVectorStore
    from scripts import benchmark_mapping
    from scripts.run_tooling import enforce_output_guard, git_state, guard_output_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", help="default: derived from the npz and HNSW settings")
    parser.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument(
        "--l4-sets", nargs="+", choices=DEFAULT_L4_SETS, default=list(DEFAULT_L4_SETS)
    )
    parser.add_argument(
        "--evidenceforge-bundle",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "DIR", "SCENARIO"),
        help="a bundle for the set evidenceforge:NAME (development bundles only)",
    )
    parser.add_argument("--vector-cache", type=Path, help=".npz of cached query vectors")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "audits" / "exemplar_backend_equivalence.json",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--wide-k", type=int, default=256)
    parser.add_argument("--wide-k-sets", nargs="+", default=["public:dev"])
    parser.add_argument("--max-differences", type=int, default=20)
    parser.add_argument("--device", help="embedding device for texts missing from the cache")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --output")
    parser.add_argument("--allow-shared-output", action="store_true")
    args = parser.parse_args(argv)

    bundles = {
        name: (Path(path), Path(scenario)) for name, path, scenario in args.evidenceforge_bundle
    }
    for name in args.sets:
        bundle = bundles.get(name.split(":", 1)[1]) if name.startswith("evidenceforge:") else None
        problem = refused_set(name, [str(item) for item in bundle] if bundle else None)
        if problem:
            parser.error(problem)
        if name.startswith("evidenceforge:") and bundle is None:
            parser.error(
                f"{name} needs --evidenceforge-bundle {name.split(':', 1)[1]} DIR SCENARIO"
            )
        if (
            name != "self"
            and not name.startswith("evidenceforge:")
            and name
            not in (
                "public:dev",
                "public:test",
                "cti_hal",
            )
        ):
            parser.error(f"unknown query set {name!r}")
    enforce_output_guard(parser, args, [args.output], repo_root=REPO_ROOT)
    if args.vector_cache is not None:
        problems = guard_output_paths(
            [args.vector_cache],
            repo_root=REPO_ROOT,
            allow_shared=args.allow_shared_output,
            overwrite=True,  # a cache is read and extended
        )
        if problems:
            parser.error("; ".join(problems))

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    embedder = CachedEmbedder(
        lambda: SentenceTransformerEmbedder.from_settings(settings, device=args.device),
        args.vector_cache,
    )
    store = ChromaVectorStore(settings, embedder, client=client)  # type: ignore[arg-type]
    npz = ExemplarIndex.load(settings.technique_exemplar_index_path)
    if npz is None:
        parser.error(f"exemplar index not found: {settings.technique_exemplar_index_path}")
    collection = args.collection or configured_collection_name(settings)[0]
    if not collection:
        parser.error("no --collection and no npz to derive its name from")
    clear_memo()
    exact_store, exact_reason = load_chroma_exemplar_store(
        _store_settings(settings, collection, "exact"), store.client
    )
    hnsw_store, hnsw_reason = load_chroma_exemplar_store(
        _store_settings(settings, collection, "hnsw"), store.client
    )
    if exact_store is None or hnsw_store is None:
        print(f"Chroma exemplar store unusable: {exact_reason or hnsw_reason}", file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git": git_state(REPO_ROOT),
        "collection": collection,
        "npz": npz.signature(),
        "stores": {"exact": exact_store.signature(), "hnsw": hnsw_store.signature()},
        "sets": list(args.sets),
        "levels": {},
    }
    case_sets = {
        name: load_set_cases(name, kb=kb, bundles=bundles)
        for name in dict.fromkeys([*args.sets, *(args.l4_sets if "L4" in args.levels else [])])
        if name != "self"
    }
    vectors_by_set: dict[str, np.ndarray] = {}
    for name in args.sets:
        if name == "self":
            vectors_by_set[name] = np.asarray(npz.embeddings, dtype=np.float32)
        else:
            texts = [exemplar_query_text(str(case.text)) for case in case_sets[name]]
            vectors_by_set[name] = embedder.vectors(texts)
    embedder.save()
    k = int(settings.technique_exemplar_k)
    report["levels"].update(
        run_vote_levels(
            npz,
            exact_store,
            hnsw_store,
            vectors_by_set,
            levels=args.levels,
            parents=parent_map(kb),
            k=k,
            tau=float(settings.technique_exemplar_tau),
            duplicate_similarity=float(settings.technique_exemplar_duplicate_similarity),
            batch_size=args.batch_size,
            wide_k=args.wide_k,
            wide_k_sets=args.wide_k_sets,
            limit=args.max_differences,
        )
    )
    if "L4" in args.levels:
        lexical_index = LexicalTechniqueIndex.from_kb(kb, fields=lexical_fields(settings))
        arms = {
            "npz": npz,
            "npz_repeat": npz,
            "chroma_exact": exact_store,
            "chroma_hnsw": hnsw_store,
        }

        def arm_model(source: Any) -> tuple[Any, str]:
            # the loader call AttributionEngine makes: refused unless the settings meet the
            # model's retrieval contract
            return load_acceptance_model(
                settings,
                kb,
                source,
                contract=retrieval_contract_for(settings, source, store=store),
            )

        level: dict[str, Any] = {"arms": {}}
        for arm, source in arms.items():
            model, status = arm_model(source)
            level["arms"][arm] = {
                "acceptance_model_status": status,
                "acceptance_exemplar_fingerprint": getattr(model, "exemplar_fingerprint", None),
                "model_active": status == "active"
                and getattr(model, "exemplar_fingerprint", None) == npz.fingerprint,
                # filled per run: whether the model really scored inside retrieve_techniques
                # (not refused per call) and how many candidate rows carry its probability
                "model_scored": True,
                "model_refused_cases": 0,
                "probability_rows": 0,
            }
        for name in args.l4_sets:
            for cues in (True, False):
                key = f"{name}{'' if cues else ':no_cues'}"
                runs: dict[str, Any] = {}
                for arm, source in arms.items():
                    model, _ = arm_model(source)
                    before = query_totals()
                    runs[arm] = benchmark_mapping.run(
                        case_sets[name],
                        mode="hybrid",
                        settings=settings,
                        kb=kb,
                        store=store,
                        lexical_index=lexical_index,
                        cues_enabled=cues,
                        acceptance_model=model,
                        exemplar_index=source,
                        bootstrap_samples=1,
                        max_candidate_rows=10_000,
                    )
                    after = query_totals()
                    entry = level["arms"][arm]
                    refused = runs[arm].get("acceptance_model_disabled") or {}
                    entry["model_scored"] = bool(
                        entry["model_scored"] and runs[arm]["acceptance_model_active"]
                    )
                    entry["model_refused_cases"] += int(refused.get("cases", 0))
                    if refused:
                        entry.setdefault("model_refused_reasons", {}).update(
                            refused.get("reasons") or {}
                        )
                    entry["probability_rows"] += sum(
                        1
                        for case in runs[arm]["cases"]
                        for row in case["candidates"]
                        if row.get("acceptance_probability") is not None
                    )
                    level["arms"][arm].setdefault("search_fallbacks", 0)
                    level["arms"][arm]["search_fallbacks"] += (
                        after["search_fallbacks"] - before["search_fallbacks"]
                    )
                    level["arms"][arm].setdefault("recall_misses", 0)
                    level["arms"][arm]["recall_misses"] += (
                        after["recall_misses"] - before["recall_misses"]
                    )
                    level["arms"][arm].setdefault("runtime_seconds", {})[key] = runs[arm][
                        "runtime_seconds"
                    ]
                level[key] = {
                    arm: compare_mapping_runs(runs["npz"], runs[arm], limit=args.max_differences)
                    for arm in ("npz_repeat", "chroma_exact", "chroma_hnsw")
                }
                embedder.save()
        # all arms falling back to the rule-based gate would match each other: the model must
        # have loaded and actually scored in every arm
        level["passed"] = all(
            entry["model_active"]
            and entry["model_scored"]
            and entry["probability_rows"] > 0
            and entry.get("search_fallbacks", 0) == 0
            for entry in level["arms"].values()
        ) and all(
            comparison["passed"]
            for key, block in level.items()
            if key not in ("arms", "passed")
            for comparison in block.values()
        )
        report["levels"]["L4"] = level
    passed = {
        name: (
            block["passed"] if name == "L4" else all(entry["passed"] for entry in block.values())
        )
        for name, block in report["levels"].items()
    }
    report["passed"] = passed
    report["ok"] = all(passed.values())
    report["vector_cache"] = {
        "path": str(args.vector_cache) if args.vector_cache else None,
        "loaded": embedder.loaded,
        "computed": embedder.computed,
    }
    report["query_totals"] = query_totals()
    embedder.save()
    destination = write_artifact(args.output, report)
    for name, ok in passed.items():
        print(f"  {name}: {'pass' if ok else 'FAIL'}")
    print(f"wrote {destination}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
