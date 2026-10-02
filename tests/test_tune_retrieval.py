"""The dev-only retrieval tuning harness (scripts/tune_retrieval.py) on fakes only.

A tiny world: a hand-made knowledge base, a bag-of-words embedder, a fake Chroma client holding
``attack_techniques`` (one sentence hidden from its "ANN" search, so ANN and exact differ) and a
synced exemplar collection, and a few dozen dev cases. No server, no GPU, no network.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import apt_classifier.config as config_module
from apt_classifier import exemplar_store
from apt_classifier.acceptance import (
    AcceptanceModel,
    choose_threshold_runtime,
    feature_vector,
)
from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
from apt_classifier.evaluation import TechniqueEvaluationCase
from apt_classifier.exemplar_store import exemplar_collection_name
from apt_classifier.exemplars import ExemplarIndex
from apt_classifier.retrieve import (
    LexicalTechniqueIndex,
    lexical_fields,
    retrieval_contract_for,
    runtime_gate_masks,
)
from apt_classifier.vectorstore import ChromaVectorStore, technique_documents
from scripts import tune_retrieval, tuning_harness
from scripts.retrieval_cache import out_of_fold_probabilities
from scripts.sync_attack_collection_copy import create_copy, read_records
from scripts.sync_exemplars_to_chroma import create_and_fill, creation_configuration
from scripts.tune_retrieval import (
    CHECK_FIT_ROW_SHARE,
    DEFAULT_PREREG,
    RECORDED_SETTINGS,
    REFERENCE_CONFIG,
    Harness,
    Refused,
    ann_recall_report,
    checked_prereg,
    choose_operating_point,
    fixed_setting_problems,
    fold_probabilities,
    p3_check,
    served_config,
)
from scripts.tuning_harness import (
    SERVING_FIELDS,
    CacheError,
    CacheMiss,
    DevCache,
    DevCase,
    DevOnlyError,
    build_cache,
    build_rowset,
    cache_key,
    case_texts,
    check_dev_artifact,
    check_dev_datasets,
    dev_path_problem,
    key_material,
    load_dev_cases,
    output_problems,
    rowset_differences,
)
from scripts.tuning_selection import (
    EXEMPLAR_LIBRARY_PLACEHOLDER,
    KNOBS,
    Bar,
    Candidate,
    CaseCounts,
    Family,
    PreregError,
    Replication,
    Variant,
    amendment_problems,
    bar_check,
    env_assignments,
    fill_exemplar_library,
    fit_arguments,
    full_config,
    greedy_path,
    is_library_fingerprint,
    load_prereg,
    micro_f1,
    paired_cluster_bootstrap,
    prereg_git_problems,
    replicate_step_back,
    tie_break_pick,
    write_env,
)
from tests.chroma_fakes import FakeClient

MODEL = "test-model"
REVISION = "r1"
DIM = 48
_WORD = re.compile(r"[a-z0-9]+")


# ----- the fake world -----
def _word_vector(word: str) -> np.ndarray:
    return np.random.default_rng(zlib.crc32(word.encode())).normal(size=DIM)


class BagEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words are similar."""

    def embed(self, texts, batch_size=None, **_):
        output = []
        for text in texts:
            words = _WORD.findall(str(text).lower())
            vector = (
                np.sum([_word_vector(word) for word in words], axis=0) if words else np.ones(DIM)
            )
            vector = vector / np.linalg.norm(vector)
            output.append(vector.astype(np.float32).tolist())
        return output


TECHNIQUES = {
    "T1059": (
        "Command and Scripting Interpreter",
        "Adversaries may abuse command and script interpreters to execute commands.",
        None,
        ["Windows", "Linux"],
    ),
    "T1059.001": (
        "PowerShell",
        "Adversaries may abuse PowerShell commands and scripts for execution.",
        "T1059",
        ["Windows"],
    ),
    "T1059.003": (
        "Windows Command Shell",
        "Adversaries may abuse the Windows command shell cmd for execution.",
        "T1059",
        ["Windows"],
    ),
    "T1003": (
        "OS Credential Dumping",
        "Adversaries may attempt to dump credentials to obtain account login and credential material.",
        None,
        ["Windows", "Linux"],
    ),
    "T1003.001": (
        "LSASS Memory",
        "Adversaries may attempt to access credential material stored in the process memory of the LSASS.",
        "T1003",
        ["Windows"],
    ),
    "T1105": (
        "Ingress Tool Transfer",
        "Adversaries may transfer tools or other files from an external system into a compromised environment.",
        None,
        ["Windows", "Linux", "macOS"],
    ),
    "T1566": (
        "Phishing",
        "Adversaries may send phishing messages to gain access to victim systems.",
        None,
        ["Windows", "Linux", "macOS"],
    ),
    "T1566.001": (
        "Spearphishing Attachment",
        "Adversaries may send spearphishing emails with a malicious attachment.",
        "T1566",
        ["Windows", "Linux", "macOS"],
    ),
    "T1053": (
        "Scheduled Task/Job",
        "Adversaries may abuse task scheduling functionality to facilitate recurring execution of malicious code.",
        None,
        ["Windows", "Linux"],
    ),
    "T1082": (
        "System Information Discovery",
        "An adversary may attempt to get detailed information about the operating system and hardware.",
        None,
        ["Windows", "Linux", "macOS"],
    ),
    "T1070": (
        "Indicator Removal",
        "Adversaries may delete or modify artifacts generated within systems to remove evidence.",
        None,
        ["Windows", "Linux"],
    ),
    "T1070.004": (
        "File Deletion",
        "Adversaries may delete files left behind by the actions of their intrusion activity.",
        "T1070",
        ["Windows", "Linux"],
    ),
    "T1543.002": (
        "Systemd Service",
        "Adversaries may create or modify systemd services to repeatedly execute malicious payloads.",
        None,
        ["Linux"],
    ),
}
PHRASES = {
    "T1059.001": [
        "ran powershell commands and scripts for execution",
        "launched an encoded powershell script to execute the payload",
    ],
    "T1059.003": [
        "used the windows command shell cmd to execute commands",
        "ran batch commands through the cmd shell",
    ],
    "T1003.001": [
        "dumped credential material from lsass process memory",
        "read lsass memory to obtain credentials",
    ],
    "T1105": [
        "transferred tools from an external system into the compromised environment",
        "downloaded additional files and tools from a remote server",
    ],
    "T1566.001": [
        "sent spearphishing emails with a malicious attachment",
        "delivered a malicious document attachment by email",
    ],
    "T1053": [
        "created a scheduled task for recurring execution of malicious code",
        "abused task scheduling to run the payload",
    ],
    "T1082": [
        "collected detailed information about the operating system and hardware",
        "gathered system information about the host",
    ],
    "T1070.004": [
        "deleted files left behind by the intrusion",
        "removed the dropped files to delete evidence",
    ],
    "T1543.002": ["created a systemd service to repeatedly execute the payload on linux"],
}
FILLER = [
    "quickly",
    "again",
    "later",
    "during",
    "campaign",
    "network",
    "victim",
    "operation",
    "stealthily",
    "hosts",
]
SYNTHETIC = {"T1059.001": ["The attacker ran an encoded PowerShell command to launch a payload."]}


def _knowledge_base() -> AttackKnowledgeBase:
    techniques = {
        tid: Technique(
            id=tid,
            name=name,
            description=description,
            platforms=list(platforms),
            is_subtechnique=parent is not None,
            parent_id=parent,
        )
        for tid, (name, description, parent, platforms) in TECHNIQUES.items()
    }
    groups = {
        "G0001": Group(
            id="G0001",
            name="Alpha",
            technique_ids=["T1059.001", "T1003.001", "T1105", "T1082"],
            technique_behaviors={
                "T1059.001": ["Alpha used PowerShell scripts to download and execute payloads."],
                "T1003.001": ["Alpha dumped LSASS process memory with procdump."],
            },
        ),
        "G0002": Group(
            id="G0002",
            name="Beta",
            technique_ids=["T1566.001", "T1105", "T1053", "T1082"],
            technique_behaviors={"T1053": ["Beta created scheduled tasks for persistence."]},
        ),
        "G0003": Group(
            id="G0003",
            name="Gamma",
            technique_ids=["T1059.003", "T1070.004", "T1082", "T1105"],
            technique_behaviors={"T1070.004": ["Gamma deleted its tools after use."]},
        ),
    }
    return AttackKnowledgeBase(version="19.2", techniques=techniques, groups=groups)


def _sentence(rng, tids, *, long: bool) -> str:
    def clause(tid):
        phrase = PHRASES[tid][int(rng.integers(len(PHRASES[tid])))]
        extra = " ".join(rng.choice(FILLER, size=int(rng.integers(1, 4))))
        return f"{phrase} {extra}"

    if long:
        parts = [
            f"The actor {clause(tid)} across the windows network of the victim." for tid in tids
        ]
        parts.append(
            "The operators then waited for several days before continuing the operation quietly."
        )
        return " ".join(parts)
    text = "The actor " + " and ".join(clause(tid) for tid in tids)
    if rng.random() < 0.3:
        text += " on windows hosts"
    return text + "."


def _write_cases(root: Path) -> None:
    rng = np.random.default_rng(11)
    tids = sorted(PHRASES)
    root.mkdir(parents=True, exist_ok=True)
    for dataset, count in (("tram", 18), ("expert", 8), ("derived_procedures", 16)):
        lines = []
        for number in range(count):
            chosen = sorted(set(rng.choice(tids, size=int(rng.integers(1, 3)), replace=False)))
            long = dataset != "tram" and number % 4 == 0
            text = _sentence(rng, chosen, long=long)
            if number == 3:
                text = "Routine maintenance: the admin " + PHRASES[chosen[0]][0] + "."
            if number == 5:
                text += " It ran powershell -enc with a hidden window."
            case_id = f"{dataset}-dev-{number + 1:06d}"
            lines.append(
                json.dumps(
                    {
                        "case_id": case_id,
                        "document_id": case_id,
                        "expected_technique_ids": chosen,
                        "original_technique_ids": chosen,
                        "label_quality": "public_dataset",
                        "source": f"tumeteor/mitre-ttp-mapping:{dataset}",
                        "split": "dev",
                        "text": text,
                    }
                )
            )
        (root / f"{dataset}_dev.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _exemplar_rows():
    rng = np.random.default_rng(5)
    rows = []
    for tid, phrases in sorted(PHRASES.items()):
        for phrase in phrases:
            for repeat in range(3):
                words = " ".join(rng.choice(FILLER, size=3))
                labels = [tid] + (["T1105"] if repeat == 2 and tid != "T1105" else [])
                source = ("tram", "expert", "derived_procedures")[repeat]
                rows.append((f"Attackers {phrase} {words} {repeat}.", labels, source))
    return rows


@dataclass
class World:
    root: Path
    settings: Any
    kb: AttackKnowledgeBase
    index: ExemplarIndex
    client: FakeClient
    cases: list[DevCase]
    cache: DevCache
    hidden: str
    writes_during_build: list


def _make_world(root: Path) -> World:
    exemplar_store.clear_memo()
    kb = _knowledge_base()
    index = ExemplarIndex.build(
        _exemplar_rows(),
        BagEmbedder(),
        embedding_model=MODEL,
        embedding_revision=REVISION,
        attack_version="19.2",
        extra_metadata={
            "split": "train",
            "source": "tumeteor/mitre-ttp-mapping",
            "datasets": ["derived_procedures", "expert", "tram"],
            "license": "CC-BY-4.0",
        },
    )
    npz = index.save(root / "exemplars.npz")
    settings = config_module.Settings(
        _env_file=None,
        data_dir=root / "data",
        technique_exemplar_index_path=npz,
        technique_exemplar_backend="npz",
        embedding_model=MODEL,
        embedding_model_revision=REVISION,
        acceptance_model_path=root / "no_model.json",
        acceptance_model_fallback_path=None,
    )
    kb.save(settings.kb_path)
    client = FakeClient()
    probe = ChromaVectorStore(settings, BagEmbedder(), client=client)
    collection = client.seed(
        "attack_techniques",
        metadata=probe._technique_collection_metadata(),
        configuration={"hnsw": {"space": "cosine"}},
    )
    documents = technique_documents(kb, SYNTHETIC)
    collection.add(
        ids=[item.id for item in documents],
        embeddings=BagEmbedder().embed([item.document for item in documents]),
        documents=[item.document for item in documents],
        metadatas=[dict(item.metadata) for item in documents],
    )
    hidden = next(item.id for item in documents if item.id.startswith("T1105:description"))
    collection.hidden_ids = {hidden}  # an ANN recall loss: exact search still finds it
    create_and_fill(
        client,
        index,
        exemplar_collection_name(index.fingerprint, 32, 400),
        configuration=creation_configuration(32, 400, 8192),
        source_npz_sha256="abc",
        batch_size=64,
    )
    _write_cases(root / "public")
    cases, _ = load_dev_cases(root / "public", ("tram", "expert", "derived_procedures"), kb)
    client.calls.clear()
    cache = build_cache(
        root / "cache",
        settings=settings,
        kb=kb,
        cases=cases,
        client=client,
        embedder_factory=BagEmbedder,
        exemplar_source=index,
        ann_depths=(90, 150),
        synthetic_ann_depths=(90,),
        log=lambda message: None,
    )
    return World(root, settings, kb, index, client, cases, cache, hidden, list(client.writes()))


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    return _make_world(tmp_path_factory.mktemp("world"))


@pytest.fixture
def harness(world) -> Harness:
    return Harness(world.cache, world.settings, world.kb, world.index, use_rows_cache=False)


@pytest.fixture
def fake_settings(world, monkeypatch):
    """Commands read the world's settings and may write under tmp (the guard's root)."""
    monkeypatch.setattr(config_module, "Settings", lambda *args, **kwargs: world.settings)


# ----- dev-only refusal -----
def test_dev_only_guard_refuses_other_splits_and_sets(tmp_path):
    for path in (
        "data/benchmarks/cti_hal/reports.jsonl",
        "/home/user/evidenceforge_heldout",
        "data/evidenceforge/heldout-clinic/scenario.yaml",
        "data/benchmarks/elastic/reports.jsonl",
        "data/benchmarks/public/normalized/tram_test.jsonl",
        "data/benchmarks/public/normalized/expert_train.jsonl",
        "runs/R01_B0_public_test.json",
        "data/audits/public_heldout_current.json",
    ):
        assert dev_path_problem(path), path
    for path in ("data/benchmarks/public/normalized", "runs/R09_B0_dev.json", "tram_dev.jsonl"):
        assert dev_path_problem(path) is None, path
    with pytest.raises(DevOnlyError):
        check_dev_datasets(["tram", "procedures"])
    with pytest.raises(DevOnlyError):
        check_dev_datasets([])
    with pytest.raises(DevOnlyError):
        check_dev_artifact({"split": "test", "datasets": ["tram"]}, "runs/R09_B0_dev.json")
    with pytest.raises(DevOnlyError):
        check_dev_artifact({"split": "dev", "datasets": ["cti_hal"]}, "runs/R09_B0_dev.json")
    check_dev_artifact({"split": "dev", "datasets": ["tram", "expert"]}, "runs/R09_B0_dev.json")
    # a case file whose rows are not dev rows
    root = tmp_path / "public"
    root.mkdir()
    row = {
        "case_id": "tram-x-1",
        "text": "ran powershell",
        "expected_technique_ids": ["T1059.001"],
        "source": "x:tram",
        "split": "test",
    }
    (root / "tram_dev.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DevOnlyError):
        load_dev_cases(root, ["tram"], _knowledge_base())


def test_commands_refuse_non_dev_inputs_before_any_work(tmp_path):
    marked = tmp_path / "cti_hal"
    marked.mkdir()
    assert (
        tune_retrieval.main(["cache", "--root", str(marked), "--output", str(tmp_path / "c")]) == 2
    )
    assert tune_retrieval.main(["parity", "--cache", str(marked)]) == 2
    rows = tmp_path / "T1_test_rows.npz"
    assert (
        tune_retrieval.main(
            ["rethreshold", "--model", "m.json", "--rows", str(rows), "--output", "o.json"]
        )
        == 2
    )


def test_output_guard_never_writes_the_shared_models(tmp_path):
    shared = tuning_harness.REPO_ROOT / "data" / "models" / "acceptance" / "x.json"
    assert any("data/models" in problem for problem in output_problems([shared], overwrite=True))
    assert output_problems([tmp_path / "x.json"], overwrite=True)  # outside the repository
    inside = tuning_harness.REPO_ROOT / "data" / "audits" / "chroma_exemplars" / "new.json"
    assert output_problems([inside], overwrite=True) == []


# ----- cache key and cache integrity -----
def test_cache_key_covers_every_input():
    case = DevCase("tram-dev-000001", "tram", "ran powershell", "x:tram", ("T1059.001",))
    base = key_material(
        cases=[case],
        text_keys=["a", "b"],
        inputs={
            "embedding_model": "m",
            "embedding_revision": "r1",
            "kb_sha256": "k",
            "synthetic_sha256": "s",
            "source_fingerprint": "f",
        },
        collection={
            "name": "attack_techniques",
            "id": "1",
            "metadata": "{}",
            "count": 10,
            "embeddings_sha256": "e",
            "records_hash": "h",
        },
        copy=None,
        exemplars={"fingerprint": "fp", "count": 5, "records_hash": "rh"},
        lists=[{"kind": "dense", "source": "ann", "fields": ["name"], "depth": 90}],
        git_commit="c1",
    )
    key = cache_key(base)
    assert cache_key(copy.deepcopy(base)) == key
    reordered = key_material(
        cases=[case],
        text_keys=["b", "a"],
        inputs=base_inputs(base),
        collection=base["technique_collection"],
        copy=None,
        exemplars=base["exemplars"],
        lists=base["lists"],
        git_commit="c1",
    )
    assert cache_key(reordered) == key  # text order does not matter

    def changed(path, value):
        material = copy.deepcopy(base)
        target = material
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        return cache_key(material)

    for path, value in (
        (("texts_sha256",), "other"),
        (("embedding_revision",), "r2"),
        (("source_fingerprint",), "g"),
        (("kb_sha256",), "k2"),
        (("technique_collection", "metadata"), '{"schema_version": 5}'),
        (("technique_collection", "count"), 11),
        (("technique_collection", "embeddings_sha256"), "e2"),
        (("exemplars", "fingerprint"), "fp2"),
        (("exemplars", "records_hash"), "rh2"),
        (("git_commit",), "c2"),
        (("cases_sha256",), "x"),
        (("lists",), [{"kind": "dense", "source": "ann", "fields": ["name"], "depth": 150}]),
    ):
        assert changed(path, value) != key, path


def base_inputs(material):
    return {
        key: material[key]
        for key in (
            "embedding_model",
            "embedding_revision",
            "kb_sha256",
            "synthetic_sha256",
            "source_fingerprint",
        )
    }


def test_cache_is_integrity_checked_and_staleness_is_detected(world, tmp_path):
    assert DevCache.load(world.cache.path).key == world.cache.key
    stale = world.cache.staleness_problems(
        settings=world.settings.model_copy(update={"embedding_model_revision": "r2"}),
        exemplar_source=world.index,
    )
    assert any("embedding_revision" in problem for problem in stale)
    other = ExemplarIndex.build(
        _exemplar_rows()[:10],
        BagEmbedder(),
        embedding_model=MODEL,
        embedding_revision=REVISION,
        attack_version="19.2",
    )
    assert any(
        "exemplar library" in problem
        for problem in world.cache.staleness_problems(
            settings=world.settings, exemplar_source=other
        )
    )
    assert (
        world.cache.staleness_problems(settings=world.settings, exemplar_source=world.index) == []
    )
    damaged = tmp_path / "damaged"
    shutil.copytree(world.cache.path, damaged, ignore=shutil.ignore_patterns("rows"))
    with (damaged / "lists.npz").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(CacheError):
        DevCache.load(damaged)
    with pytest.raises(CacheError):  # a directory that already holds a cache is never reused
        build_cache(
            world.cache.path,
            settings=world.settings,
            kb=world.kb,
            cases=world.cases,
            client=world.client,
            embedder_factory=BagEmbedder,
            exemplar_source=world.index,
        )


def test_cache_build_is_read_only_and_stand_ins_never_guess(world):
    assert world.writes_during_build == []  # build_cache only reads
    store = world.cache.technique_store("ann")
    segment = case_texts(world.cache.cases[0]).segments[0]
    assert store.query_techniques([segment], n_results=90, fields=SERVING_FIELDS)
    with pytest.raises(CacheMiss):
        store.query_techniques([segment], n_results=123, fields=SERVING_FIELDS)  # depth not cached
    with pytest.raises(CacheMiss):
        store.query_techniques(["a text nobody cached"], n_results=90, fields=SERVING_FIELDS)
    with pytest.raises(CacheMiss):
        store.query_techniques([segment], n_results=90, fields=None)
    with pytest.raises(CacheMiss):
        world.cache.technique_store("copy")  # no collection copy was cached
    with pytest.raises(RuntimeError):
        world.cache.embedder.embed(["a text nobody embedded"])


# ----- P1-style parity: harness rows equal the real pipeline's rows -----
def _live_rows(world, harness, cues, client=None):
    settings = harness.settings_for(REFERENCE_CONFIG)
    return build_rowset(
        world.cache.cases,
        kb=world.kb,
        settings=settings,
        store=ChromaVectorStore(settings, world.cache.embedder, client=client or world.client),
        lexical_index=LexicalTechniqueIndex.from_kb(world.kb, fields=lexical_fields(settings)),
        exemplar_index=world.index,
        cues_enabled=cues,
        generic=harness.generic,
    )


def test_harness_rows_equal_the_real_store_rows_bitwise(world, harness):
    for cues in (True, False):
        cached = harness.rows(REFERENCE_CONFIG, cues)
        live = _live_rows(world, harness, cues)
        difference = rowset_differences(cached, live)
        assert len(cached) > 50 and cached.labels.sum() > 10
        assert difference["differing_cases"] == 0, difference["examples"]
    # the only calls on the fake were reads (get_or_create is the serving store's own lookup)
    assert {call[0] for call in world.client.writes()} <= {"get_or_create_collection"}


def test_parity_catches_a_changed_collection(tmp_path):
    changed = _make_world(tmp_path)
    harness = Harness(
        changed.cache, changed.settings, changed.kb, changed.index, use_rows_cache=False
    )
    changed.client.collections["attack_techniques"].hidden_ids = set()  # the index "changed"
    cached = harness.rows(REFERENCE_CONFIG, True)
    live = _live_rows(changed, harness, True)
    assert rowset_differences(cached, live)["differing_cases"] > 0


def test_harness_rows_equal_the_fit_extraction(world, harness):
    from scripts.fit_acceptance_model import extract_rows

    settings = harness.settings_for(REFERENCE_CONFIG)
    fitted = extract_rows(
        [
            TechniqueEvaluationCase(
                case_id=case.case_id,
                document_id=case.case_id,
                text=case.text,
                expected_technique_ids=case.expected,
                source=case.source,
            )
            for case in world.cache.cases
        ],
        settings=settings,
        kb=world.kb,
        store=world.cache.technique_store("ann"),
        lexical_index=world.cache.lexical_index(lexical_fields(settings)),
        priors=None,
        exemplar_index=world.index,
    )
    rows = harness.rows(REFERENCE_CONFIG, True)
    matrix = np.vstack([feature_vector(row) for row in fitted.rows])
    assert np.array_equal(matrix.view(np.uint64), rows.features.view(np.uint64))
    assert fitted.labels == rows.labels.tolist() and fitted.case_ids == rows.row_case_ids
    assert fitted.technique_ids == rows.technique_ids.tolist()
    blocked, forced = fitted.masks()
    harness_blocked, harness_forced = rows.gate_masks()
    assert blocked == harness_blocked.tolist() and forced == harness_forced.tolist()
    assert runtime_gate_masks(rows.flag_dicts()) == (blocked, forced)
    assert any(forced) and any(blocked)  # the fake exercises the cue and guard paths


def test_fold_probabilities_match_the_fit_code(world, harness):
    rows = harness.rows(REFERENCE_CONFIG, True)
    case_fold = harness.folds(7)
    mine, _ = fold_probabilities(rows, case_fold, l2=1.0, attack_version="19.2")
    reference = out_of_fold_probabilities(
        [dict(row) for row in rows.feature_dicts],
        rows.labels.tolist(),
        case_fold[rows.case_index].tolist(),
        attack_version="19.2",
        l2=1.0,
    )
    assert np.array_equal(mine.view(np.uint64), reference.view(np.uint64))


def test_emulated_decisions_and_ranks_equal_finalize(world, harness):
    for cues in (True, False):
        report = p3_check(harness, REFERENCE_CONFIG, cues=cues, thresholds=(0.2, 0.35, 0.5))
        assert report["passed"], report["examples"]
        assert report["decisions"] == 3 * len(world.cache.cases)


def test_variants_change_the_rows_and_ann_recall_sees_the_hidden_sentence(world, harness):
    reference = harness.rows(REFERENCE_CONFIG, True)
    for overrides in ({"dense_search": "exact"}, {"exemplar_k": 8}, {"exemplar_tau": 0.5}):
        variant = harness.rows(full_config(REFERENCE_CONFIG, overrides), True)
        assert rowset_differences(reference, variant)["differing_cases"] > 0, overrides
    report = ann_recall_report(harness)
    assert report["segment_recall"][0]["recall_mean"] < 1.0
    assert {entry["depth"] for entry in report["cases"]} == {90, 150}


def test_parity_command_passes_and_writes_nothing_to_chroma(
    world, fake_settings, monkeypatch, tmp_path
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    output = tmp_path / "parity.json"
    code = tune_retrieval.main(
        ["parity", "--cache", str(world.cache.path), "--live", "--output", str(output)],
        client=world.client,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0, report
    assert report["P1"]["cues_on"]["differing_cases"] == 0
    assert report["P2"]["passed"] and report["P2"]["queries"] > 0
    assert report["P3"]["passed"] and report["P1b"]["ran"] is False
    assert {call[0] for call in world.client.writes()} <= {"get_or_create_collection"}
    # without --live the blocking P1 is not run: never a pass
    assert (
        tune_retrieval.main(["parity", "--cache", str(world.cache.path)], client=world.client) == 1
    )


# ----- selection logic -----
def _counts(tp_rate: float, *, cases: int = 300, seeds: int = 3, fp: int = 1) -> CaseCounts:
    rng = np.random.default_rng(int(tp_rate * 1000))
    tp = (rng.random((seeds, cases)) < tp_rate).astype(np.int64)
    return CaseCounts(tp=tp, fp=np.full((seeds, cases), fp), expected=np.full(cases, 2))


def _summary(counts: CaseCounts, **changes) -> dict[str, Any]:
    f1 = float(np.mean(micro_f1(counts.tp, counts.fp, counts.expected[None, :])))
    summary = {
        "f1": f1,
        "precision": 0.55,
        "precision_floor_met": True,
        "dataset_f1": {"tram": f1, "derived_procedures": f1, "expert": f1},
        "f1_no_cues": f1,
        "hit_at_10": 0.8,
        "generic_share": 0.2,
        "accepted_per_case": 1.0,
        "per_seed": [
            {"f1": float(value)}
            for value in micro_f1(counts.tp, counts.fp, counts.expected[None, :])
        ],
    }
    summary.update(changes)
    return summary


def _candidate(name, config, tp_rate, **changes) -> Candidate:
    counts = _counts(tp_rate)
    return Candidate(name, config, _summary(counts, **changes), counts)


REFERENCE = full_config(REFERENCE_CONFIG)
CLUSTERS = np.arange(300) // 2  # pairs of near-duplicate cases


def test_bootstrap_is_paired_clustered_and_seeded():
    base = _counts(0.30)
    same = paired_cluster_bootstrap(base, base, CLUSTERS, samples=200, seed=7)
    assert same["delta_f1"] == 0.0 and same["ci95"] == [0.0, 0.0]
    better = paired_cluster_bootstrap(base, _counts(0.45), CLUSTERS, samples=200, seed=7)
    assert better["delta_f1"] > 0 and better["ci95"][0] > 0
    assert paired_cluster_bootstrap(base, _counts(0.45), CLUSTERS, samples=200, seed=7) == better
    one_cluster = paired_cluster_bootstrap(base, _counts(0.45), np.zeros(300), samples=50, seed=7)
    assert one_cluster["ci95"][0] == pytest.approx(
        one_cluster["delta_f1"]
    )  # the whole set every time
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(base, _counts(0.3, cases=200), CLUSTERS[:200], samples=10, seed=7)


def test_bar_checks_each_criterion():
    bar = Bar()
    best = _summary(
        _counts(0.3),
        f1=0.40,
        f1_no_cues=0.38,
        hit_at_10=0.80,
        generic_share=0.20,
        accepted_per_case=1.0,
        dataset_f1={"tram": 0.4, "derived_procedures": 0.4, "expert": 0.4},
    )
    good = dict(best, f1=0.410, f1_no_cues=0.38, precision=0.52)
    boot = {"ci95": [0.001, 0.02]}
    assert bar_check(good, best, boot, bar)["passed"]  # exactly +0.010 passes
    failing = {
        "delta_f1": dict(good, f1=0.4099),
        "precision_floor": dict(good, precision=0.49),
        "tram_f1_drop": dict(
            good, dataset_f1={"tram": 0.379, "derived_procedures": 0.4, "expert": 0.4}
        ),
        "delta_f1_no_cues": dict(good, f1_no_cues=0.379),
        "delta_hit_at_10": dict(good, hit_at_10=0.789),
        "generic_share_rise": dict(good, generic_share=0.221),
        "accepted_per_case_rise": dict(good, accepted_per_case=1.26),
    }
    for name, candidate in failing.items():
        verdict = bar_check(candidate, best, boot, bar)
        assert not verdict["passed"] and not verdict["checks"][name]["passed"], name
    assert not bar_check(good, best, {"ci95": [0.0, 0.02]}, bar)["passed"]  # CI must exclude 0
    assert not bar_check(dict(good, precision_floor_met=False), best, boot, bar)["passed"]
    edge = bar_check(
        dict(
            good,
            accepted_per_case=1.25,
            dataset_f1={"tram": 0.38, "derived_procedures": 0.38, "expert": 0.34},
        ),
        best,
        boot,
        bar,
    )
    assert edge["passed"] and edge["flags"]  # expert never selects, a drop > 0.05 is flagged


def test_tie_break_prefers_fewer_knobs_then_closer_values():
    two_knobs = _candidate(
        "both", full_config(REFERENCE, {"exemplar_k": 128, "exemplar_tau": 0.03}), 0.45, f1=0.500
    )
    one_far = _candidate("k256", full_config(REFERENCE, {"exemplar_k": 256}), 0.45, f1=0.497)
    one_near = _candidate("k128", full_config(REFERENCE, {"exemplar_k": 128}), 0.45, f1=0.496)
    assert tie_break_pick([two_knobs, one_far, one_near], REFERENCE, window=0.005).id == "k128"
    clear_winner = _candidate("both", two_knobs.config, 0.45, f1=0.510)
    assert tie_break_pick([clear_winner, one_near], REFERENCE, window=0.005).id == "both"
    precise = _candidate(
        "k32", full_config(REFERENCE, {"exemplar_k": 32}), 0.45, f1=0.496, precision=0.6
    )
    assert tie_break_pick([one_near, precise], REFERENCE, window=0.005).id == "k32"  # same distance


def test_greedy_path_builds_family_on_family():
    reference = _candidate("B0r", REFERENCE, 0.30)
    families = (
        Family(
            "F1",
            "E1",
            "",
            (Variant("k128", "F1", {"exemplar_k": 128}), Variant("k32", "F1", {"exemplar_k": 32})),
        ),
        Family(
            "F2",
            "E2",
            "",
            (
                Variant("pool16", "F2", {"candidate_pool": 16}),
                Variant("k128again", "F2", {"exemplar_k": 128}),
            ),
        ),
        Family("F3", "E3", "", (Variant("l2", "F3", {"l2": 3.0}),)),
    )
    rates = {"k128": 0.45, "k32": 0.31, "pool16": 0.60, "l2": 0.70}
    seen: list[tuple[str, dict]] = []

    def evaluate(variant, config):
        seen.append((variant.id, config))
        changes = {"generic_share": 0.5} if variant.id == "l2" else {}  # fails the generic bar
        return _candidate(variant.id, config, rates[variant.id], **changes)

    path, reports = greedy_path(
        families,
        evaluate,
        reference,
        bar=Bar(),
        window=0.005,
        clusters=CLUSTERS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert [candidate.id for candidate in path] == ["B0r", "k128", "pool16"]
    by_id = dict(seen)
    assert by_id["pool16"]["exemplar_k"] == 128 and by_id["pool16"]["candidate_pool"] == 16
    assert by_id["l2"]["candidate_pool"] == 16  # F3 builds on F2's adoption
    assert reports[1]["identical_to_current"] == ["k128again"]
    assert [report["adopted"] for report in reports] == [True, True, False]
    assert not reports[2]["bar"]["checks"]["generic_share_rise"]["passed"]


def test_replication_steps_back_to_the_last_configuration_that_holds():
    reference = Candidate(
        "B0r", REFERENCE, {"per_seed": [{"f1": 0.40}, {"f1": 0.40}]}, _counts(0.30, seeds=2)
    )
    first = Candidate(
        "s1",
        full_config(REFERENCE, {"exemplar_k": 128}),
        {"per_seed": [{"f1": 0.41}, {"f1": 0.406}]},
        _counts(0.40, seeds=2),
    )
    weak = Candidate(
        "s2",
        full_config(first.config, {"candidate_pool": 16}),
        {"per_seed": [{"f1": 0.43}, {"f1": 0.404}]},
        _counts(0.45, seeds=2),
    )
    rule = Replication(seeds=(41, 53), min_delta_f1_per_seed=0.005, pooled_ci_lower_above=0.0)
    selected, steps = replicate_step_back(
        [reference, first, weak],
        reference,
        rule=rule,
        clusters=CLUSTERS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert selected.id == "s1" and [step["id"] for step in steps] == ["s2", "s1"]
    assert steps[0]["seeds_passed"] is False and steps[1]["replicates"] is True
    nothing = Candidate(
        "s1", first.config, {"per_seed": [{"f1": 0.401}, {"f1": 0.43}]}, _counts(0.40, seeds=2)
    )
    selected, steps = replicate_step_back(
        [reference, nothing],
        reference,
        rule=rule,
        clusters=CLUSTERS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert selected is reference and len(steps) == 1  # worst case: T1 := B0r


def test_t1_env_lines_round_trip_through_settings(tmp_path, monkeypatch):
    config = full_config(
        REFERENCE,
        {"exemplar_k": 128, "exemplar_tau": 0.03, "exemplar_segment_votes": "max", "l2": 3.0},
    )
    lines = env_assignments(config)
    assert "TECHNIQUE_EXEMPLAR_K=128" in lines and "TECHNIQUE_INCLUDE_SYNTHETIC=false" in lines
    assert "TECHNIQUE_DENSE_OVERFETCH=0" in lines and "TECHNIQUE_PLATFORM_FROM_TEXT=true" in lines
    assert len(lines) == sum(1 for knob in KNOBS.values() if knob.env)
    path = write_env(tmp_path / "T1.env", lines)
    tokens = path.read_text(encoding="utf-8").split()
    assert tokens == lines and all(token.count("=") == 1 for token in tokens)
    for token in tokens:
        name, value = token.split("=")
        monkeypatch.setenv(name, value)
    loaded = config_module.Settings(_env_file=None)
    for name, knob in KNOBS.items():
        if knob.setting:
            assert getattr(loaded, knob.setting) == config[name], name
    assert fit_arguments(config) == ["--l2", "3.0"]
    with pytest.raises(ValueError):
        env_assignments(full_config(REFERENCE, {"dense_search": "exact"}))
    exact = env_assignments(
        full_config(REFERENCE, {"dense_search": "exact"}),
        technique_collection="attack_techniques-s4-abc-m32-efc400",
    )
    assert exact[-1] == "TECHNIQUE_COLLECTION_NAME=attack_techniques-s4-abc-m32-efc400"


def test_knob_values_are_validated():
    with pytest.raises(ValueError):
        full_config(REFERENCE, {"exemplar_k": 1.5})
    with pytest.raises(ValueError):
        full_config(REFERENCE, {"exemplar_vote_norm": "sqrt"})
    with pytest.raises(ValueError):
        full_config(REFERENCE, {"unknown_knob": 1})
    with pytest.raises(ValueError):
        full_config(REFERENCE, {"include_synthetic": "yes"})


# ----- the pre-registration -----
def test_the_prereg_file_matches_the_plan():
    prereg = load_prereg(DEFAULT_PREREG)
    assert prereg.reference == full_config(REFERENCE_CONFIG) and prereg.reference_id == "B0r"
    assert prereg.dev_datasets == ("tram", "expert", "derived_procedures")
    assert prereg.dev_cases == 1147
    # Chroma is the source: the library is the Chroma collection's, whose fingerprint the sync
    # report gave (prereg-fill wrote it before the registration commit, R10v2)
    assert prereg.exemplar_library == {
        "fingerprint": REGISTERED_LIBRARY_FINGERPRINT,
        "count": 6360,
        "split": "train",
        "backend": "chroma",
        "source_fingerprint": "0f8d2e19a59967c63b03ea71",
        "tolerance": 1e-6,
    }
    assert not prereg.exemplar_library_placeholder
    registered = prereg.payload["fixed"]["exemplar_library"]
    assert registered["collection"] == prereg.payload["fixed"]["hnsw"]["exemplar_collection"]
    assert registered["tolerance"] == 1e-6
    # the registered tolerance is a fixed setting: checked by every selection step, in T1.env
    assert prereg.fixed_settings["technique_exemplar_tolerance"] == 1e-6
    assert prereg.amendments == () and prereg.retry_allowed == 1
    assert [family.id for family in prereg.families] == [
        "E1a",
        "E1b",
        "E1c",
        "E1d",
        "E2a",
        "E2b",
        "E2c",
        "E2d",
        "E3",
    ]
    assert prereg.variant_count == 27
    sizes = {family.id: len(family.variants) for family in prereg.families}
    assert sizes == {
        "E1a": 11,
        "E1b": 3,
        "E1c": 1,
        "E1d": 2,
        "E2a": 1,
        "E2b": 2,
        "E2c": 3,
        "E2d": 2,
        "E3": 2,
    }
    grid = {
        (v.overrides["exemplar_k"], v.overrides["exemplar_tau"])
        for v in prereg.families[0].variants
    }
    assert len(grid) == 11 and (64, 0.05) not in grid
    assert prereg.selection_seeds == (7, 11, 13) and prereg.replication.seeds == (41, 53)
    assert prereg.bootstrap_samples == 1000 and prereg.bootstrap_seed == 7
    assert prereg.bar == Bar()  # the section-6 numbers
    assert (
        prereg.tie_window == 0.005
        and prereg.min_precision == 0.5
        and prereg.p40_min_precision == 0.4
    )
    assert prereg.fixed_settings["technique_rrf_k"] == 60
    assert prereg.fixed_settings["technique_exemplar_duplicate_similarity"] == 0.95
    assert (
        fixed_setting_problems(config_module.Settings(_env_file=None), prereg.fixed_settings) == []
    )
    assert {item["id"] for item in prereg.decomposition} == {
        "old_fit",
        "fix_synthetic_only",
        "fix_platform_only",
        "fix_threshold_rule_only",
    }
    payload = prereg.payload
    assert {gate["id"] for gate in payload["gates"]} >= {
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "D1",
        "G8",
        "G9",
    }
    assert len(payload["combiner_refit_trigger"]["refit_if_any"]) == 3


def _git(repo: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "harness test",
        "GIT_AUTHOR_EMAIL": "harness@example.invalid",
        "GIT_COMMITTER_NAME": "harness test",
        "GIT_COMMITTER_EMAIL": "harness@example.invalid",
    }
    completed = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        env=environment,
    )
    return completed.stdout.decode().strip()


def _filled_default_prereg(fingerprint: str = "a1" * 12) -> dict[str, Any]:
    """The pre-registration with its library placeholder filled (as prereg-fill would)."""
    payload = json.loads(DEFAULT_PREREG.read_text(encoding="utf-8"))
    payload["fixed"]["exemplar_library"]["fingerprint"] = fingerprint
    return payload


def _repo_with(
    tmp_path: Path, payload: dict[str, Any], *, commit: bool = True
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo, "init", "-q")
    target = repo / "docs" / "prereg.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if commit:
        _git(repo, "add", "docs/prereg.json")
        _git(repo, "commit", "-q", "-m", "prereg")
    return repo, target


def test_prereg_must_be_committed_and_clean(tmp_path):
    payload = _filled_default_prereg()
    repo, target = _repo_with(tmp_path, payload, commit=False)
    problems, _ = prereg_git_problems(target, repo_root=repo)
    assert any("not committed" in problem for problem in problems)
    _git(repo, "add", "docs/prereg.json")
    problems, _ = prereg_git_problems(target, repo_root=repo)
    assert any("uncommitted" in problem for problem in problems)  # staged is not committed
    _git(repo, "commit", "-q", "-m", "prereg")
    problems, identity = prereg_git_problems(target, repo_root=repo)
    assert problems == [] and identity["commit"] == _git(repo, "rev-parse", "HEAD")
    prereg, identity = checked_prereg(target, repo_root=repo)
    assert identity["variants"] == 27
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Refused):
        checked_prereg(target, repo_root=repo)
    outside = tmp_path / "loose.json"
    outside.write_text(json.dumps(payload), encoding="utf-8")
    assert prereg_git_problems(outside, repo_root=repo)[0]


# ----- end to end on the fake world -----
def _small_prereg(world: World) -> dict[str, Any]:
    payload = json.loads(DEFAULT_PREREG.read_text(encoding="utf-8"))
    payload["families"] = [
        {
            "id": "E1b",
            "group": "E1",
            "variants": [{"id": "E1b_new0", "set": {"exemplar_new_candidates": 0}}],
        },
        {"id": "E3", "group": "E3", "variants": [{"id": "E3_l2_3.0", "set": {"l2": 3.0}}]},
    ]
    payload["variant_count"] = 2
    payload["fixed"]["settings"]["embedding_model"] = MODEL
    payload["fixed"]["settings"]["embedding_model_revision"] = REVISION
    # the fake world's library is an npz (the real pre-registration names the Chroma library)
    payload["fixed"]["exemplar_library"].update(
        {
            "fingerprint": world.index.fingerprint,
            "count": len(world.index),
            "split": "train",
            "backend": world.index.backend,
            "source_fingerprint": world.index.fingerprint,
        }
    )
    payload["dev_only"]["cases"] = len(world.cache.cases)
    payload["selection"]["bar"]["generic_min_groups"] = 2
    payload["selection"]["bar"]["bootstrap"]["samples"] = 200
    return payload


def _run_parity(world: World, output: Path, *, cache: DevCache | None = None, extra=()) -> Path:
    """P1-P3 (live) of a cache; the caller has pointed Settings and REPO_ROOT at the world."""
    code = tune_retrieval.main(
        [
            "parity",
            "--cache",
            str((cache or world.cache).path),
            "--live",
            "--output",
            str(output),
            *extra,
        ],
        client=world.client,
    )
    assert code == 0, output.read_text(encoding="utf-8") if output.is_file() else code
    return output


def _sweep_arguments(cache: DevCache, prereg: Path, parity: Path | None, output: Path) -> list:
    arguments = ["sweep", "--cache", str(cache.path), "--prereg", str(prereg)]
    arguments += ["--seeds", "7", "11", "13", "--bootstrap", "200", "--output", str(output)]
    return arguments + (["--parity", str(parity)] if parity is not None else [])


@pytest.fixture(scope="module")
def swept(world, tmp_path_factory):
    """One committed small pre-registration, a passing parity run and a sweep on the world."""
    base = tmp_path_factory.mktemp("swept")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config_module, "Settings", lambda *args, **kwargs: world.settings)
        patch.setattr(tuning_harness, "REPO_ROOT", base)
        repo, prereg = _repo_with(base, _small_prereg(world))
        parity = _run_parity(world, base / "R21_parity.json")
        sweep = base / "R23_dev_sweep.json"
        code = tune_retrieval.main(
            _sweep_arguments(world.cache, prereg, parity, sweep),
            client=world.client,
            repo_root=repo,
        )
        assert code == 0
    return SimpleNamespace(base=base, repo=repo, prereg=prereg, parity=parity, sweep=sweep)


def _replicate(swept, tmp_path, *, client, name="R24.json", extra=()) -> tuple[int, Path]:
    output = tmp_path / name
    code = tune_retrieval.main(
        [
            "replicate",
            "--sweep",
            str(swept.sweep),
            "--seeds",
            "41",
            "53",
            "--output",
            str(output),
            *extra,
        ],
        client=client,
        repo_root=swept.repo,
    )
    return code, output


def test_sweep_and_replicate_end_to_end(world, fake_settings, monkeypatch, tmp_path):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    repo, prereg = _repo_with(tmp_path, _small_prereg(world))
    parity = _run_parity(world, tmp_path / "R21_parity.json")
    sweep_path = tmp_path / "R23_dev_sweep.json"
    arguments = [
        "sweep",
        "--cache",
        str(world.cache.path),
        "--prereg",
        str(prereg),
        "--parity",
        str(parity),
    ]
    # the CLI must repeat the pre-registered seeds and bootstrap size
    assert (
        tune_retrieval.main(
            [*arguments, "--seeds", "7", "--bootstrap", "200", "--output", str(sweep_path)],
            client=world.client,
            repo_root=repo,
        )
        == 2
    )
    code = tune_retrieval.main(
        [*arguments, "--seeds", "7", "11", "13", "--bootstrap", "200", "--output", str(sweep_path)],
        client=world.client,
        repo_root=repo,
    )
    assert code == 0
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    assert sweep["prereg"]["commit"] == _git(repo, "rev-parse", "HEAD")
    assert sweep["counts"]["pre_registered_variants"] == 2
    assert [report["family"] for report in sweep["families"]] == ["E1b", "E3"]
    assert sweep["e0"]["B0r"]["summary"]["seeds"] == [7, 11, 13]
    assert {item["id"] for item in sweep["e0"]["decomposition"]} == {
        "old_fit",
        "fix_synthetic_only",
        "fix_platform_only",
        "fix_threshold_rule_only",
    }
    assert sweep["e0"]["fold_leak"]["exemplars"] == len(world.index)
    assert sweep["finalist"]["e4_nested_cv"]["finalist"]["per_seed"]
    assert sweep["llm"] == {"calls": 0, "cost_usd": 0.0}
    # what the next step compares: base settings, code, the parity run and the true case set
    assert sweep["base_settings"]["settings"]["similarity_threshold"] == 0.45
    assert sweep["base_settings"]["retrieval_kwargs"]["rrf_k"] == 60
    assert set(sweep["code"]) == {"sources", "harness", "tune_retrieval", "selection"}
    assert sweep["parity"]["path"] == str(parity)
    assert sweep["dev_only"]["datasets"] == sorted({case.dataset for case in world.cache.cases})
    assert sweep["prereg"]["history"] == [sweep["prereg"]["commit"]]

    replicate_path = tmp_path / "R24_dev_replicate.json"
    env_path = tmp_path / "T1.env"
    code = tune_retrieval.main(
        [
            "replicate",
            "--sweep",
            str(sweep_path),
            "--seeds",
            "41",
            "53",
            "--output",
            str(replicate_path),
            "--write-env",
            str(env_path),
        ],
        client=world.client,
        repo_root=repo,
    )
    assert code == 0
    replicate = json.loads(replicate_path.read_text(encoding="utf-8"))
    selected = replicate["selected"]
    assert env_path.read_text(encoding="utf-8").split() == env_assignments(
        selected["config"],
        fixed=load_prereg(prereg).fixed_settings,
        exemplar_library={"backend": "npz"},
    )
    assert env_path.read_text(encoding="utf-8").split()[-1] == "TECHNIQUE_EXEMPLAR_BACKEND=npz"
    assert (tmp_path / "T1.fit_args").read_text(encoding="utf-8").split() == fit_arguments(
        selected["config"]
    )
    assert set(selected["operating_points"]) == {"seed", "T1", "T1-P40"}
    assert replicate["prereg"]["sha256"] == sweep["prereg"]["sha256"]
    # a changed pre-registration is refused
    prereg.write_text(prereg.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert (
        tune_retrieval.main(
            [
                "replicate",
                "--sweep",
                str(sweep_path),
                "--output",
                str(tmp_path / "again.json"),
                "--overwrite",
            ],
            client=world.client,
            repo_root=repo,
        )
        == 2
    )


def _selection_artifact(world, path: Path, *, settings=None, config=REFERENCE_CONFIG) -> Path:
    """A replicate artifact as far as check-fit reads it."""
    path.write_text(
        json.dumps(
            {
                "kind": "tune_retrieval_replicate",
                "selected": {"id": "B0r", "config": config, "operating_points": {"seed": 7}},
                "cache": {"path": str(world.cache.path), "key": world.cache.key},
                "prereg": {
                    "path": str(DEFAULT_PREREG),
                    "sha256": load_prereg(DEFAULT_PREREG).sha256,
                },
                "base_settings": tune_retrieval.base_settings_record(settings or world.settings),
                "code": tune_retrieval.harness_code(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _fit_artifacts(
    world, harness, tmp_path, fitted, *, contract, out_of_fold=None, name="T1", summary=None
):
    """What fit_acceptance_model.py writes for ``fitted`` rows: the model (threshold from the
    harness rows' out-of-fold p at seed 7, the contract, the row counts) and --dump-rows."""
    from scripts.fit_acceptance_model import dump_rows

    model = AcceptanceModel.fit(
        fitted.rows, fitted.labels, attack_version="19.2", contract=contract
    )
    rows = harness.rows(REFERENCE_CONFIG, True)
    oof, _ = fold_probabilities(rows, harness.folds(7), l2=1.0, attack_version="19.2")
    model.threshold = float(
        choose_operating_point(rows, oof, rule="runtime", min_precision=0.5)["threshold"]
    )
    model.metadata = {
        "split": "dev",
        "min_precision": 0.5,
        "row_count": len(fitted.rows),
        "positive_rows": int(sum(fitted.labels)),
    }
    model_path = model.save(tmp_path / f"acceptance_{name}.json")
    rows_path = dump_rows(
        tmp_path / f"{name}_rows.npz",
        fitted,
        datasets=dict(zip(rows.case_ids, rows.datasets, strict=True)),
        in_sample=[model.probability(row) for row in fitted.rows],
        out_of_fold=out_of_fold,
        folds=None,
        groups=None,
        summary=summary
        if summary is not None
        else {"threshold": model.threshold, "retrieval_contract": contract},
    )
    return model, model_path, rows_path


def _harness_contract(world, harness):
    return retrieval_contract_for(
        harness.settings_for(REFERENCE_CONFIG),
        world.index,
        store=world.cache.technique_store("ann"),
    )


def _fit_like_artifacts(world, harness, tmp_path):
    """What fit_acceptance_model.py --dump-rows writes, made from the harness rows."""
    from scripts.fit_acceptance_model import FitRows

    rows = harness.rows(REFERENCE_CONFIG, True)
    fitted = FitRows(
        rows=[dict(row) for row in rows.feature_dicts],
        labels=rows.labels.tolist(),
        case_ids=list(rows.row_case_ids),
        technique_ids=rows.technique_ids.tolist(),
        flags=rows.flag_dicts(),
        expected_counts=dict(rows.expected_counts),
    )
    oof, _ = fold_probabilities(rows, harness.folds(7), l2=1.0, attack_version="19.2")
    model, model_path, rows_path = _fit_artifacts(
        world,
        harness,
        tmp_path,
        fitted,
        contract=_harness_contract(world, harness),
        out_of_fold=oof,
    )
    selection = _selection_artifact(world, tmp_path / "R24_dev_replicate.json")
    return model, model_path, rows_path, selection, fitted, oof


def test_check_fit_and_rethreshold(world, harness, fake_settings, monkeypatch, tmp_path):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    model, model_path, rows_path, selection, fitted, oof = _fit_like_artifacts(
        world, harness, tmp_path
    )
    check = [
        "check-fit",
        "--rows",
        str(rows_path),
        "--model",
        str(model_path),
        "--sweep",
        str(selection),
    ]
    assert (
        tune_retrieval.main([*check, "--output", str(tmp_path / "check.json")], client=world.client)
        == 0
    )
    report = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
    assert report["checks"]["rows_equal_share"]["value"] == 1.0
    assert report["checks"]["weight_cosine"]["value"] > 0.999999
    # a dump whose rows differ fails the 99.5% row check
    with np.load(rows_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["features"] = arrays["features"].copy()
    arrays["features"][:10, 0] += 1.0
    changed = tmp_path / "T1_rows_changed.npz"
    with changed.open("wb") as handle:
        np.savez(handle, **arrays)
    assert (
        tune_retrieval.main(
            [
                "check-fit",
                "--rows",
                str(changed),
                "--model",
                str(model_path),
                "--sweep",
                str(selection),
            ],
            client=world.client,
        )
        == 1
    )

    output = tmp_path / "acceptance_T1_p40.json"
    assert (
        tune_retrieval.main(
            [
                "rethreshold",
                "--model",
                str(model_path),
                "--rows",
                str(rows_path),
                "--min-precision",
                "0.40",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    p40 = AcceptanceModel.load(output)
    assert np.array_equal(p40.weights.view(np.uint64), model.weights.view(np.uint64))
    blocked, forced = fitted.masks()
    expected = choose_threshold_runtime(
        oof,
        fitted.labels,
        fitted.case_ids,
        fitted.expected_counts,
        blocked=blocked,
        forced=forced,
        technique_ids=fitted.technique_ids,
        min_precision=0.40,
    )
    assert p40.threshold == expected["threshold"]
    assert p40.metadata["rethreshold"]["self_check"]["matches_source"] is True
    # never over an existing file, never from a dump without out-of-fold probabilities
    assert (
        tune_retrieval.main(
            [
                "rethreshold",
                "--model",
                str(model_path),
                "--rows",
                str(rows_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    arrays["p_out_of_fold"] = np.full(len(arrays["labels"]), np.nan)
    in_sample = tmp_path / "T1_rows_in_sample.npz"
    with in_sample.open("wb") as handle:
        np.savez(handle, **arrays)
    assert (
        tune_retrieval.main(
            [
                "rethreshold",
                "--model",
                str(model_path),
                "--rows",
                str(in_sample),
                "--output",
                str(tmp_path / "x.json"),
            ]
        )
        == 2
    )


# ----- review fixes: every step runs under the sweep's settings, library, cases and code -----
def _settings_returning(monkeypatch, settings) -> None:
    monkeypatch.setattr(config_module, "Settings", lambda *args, **kwargs: settings)


def test_recorded_settings_are_the_preregistered_fixed_settings():
    assert set(RECORDED_SETTINGS) == set(load_prereg(DEFAULT_PREREG).fixed_settings)


def test_replicate_refuses_settings_the_sweep_did_not_run_under(
    world, swept, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    drifts = (
        ({"similarity_threshold": 0.40}, "similarity_threshold is 0.4"),  # a fixed setting
        ({"technique_rrf_k": 20}, "technique_rrf_k is 20"),
        ({"technique_segment_acceptance": True}, "technique_segment_acceptance is True"),
        # not a fixed setting, but a retrieval argument: the recorded base settings catch it
        ({"behavior_similarity_threshold": 0.6}, "behavior_threshold is 0.6"),
        # the same library from another backend: the sweep's rows were not built from it
        (
            {"technique_exemplar_backend": "chroma", "technique_exemplar_search": "hnsw"},
            "the exemplar library loaded as",
        ),
    )
    for number, (update, expected) in enumerate(drifts):
        _settings_returning(monkeypatch, world.settings.model_copy(update=update))
        code, output = _replicate(swept, tmp_path, client=world.client, name=f"R24_{number}.json")
        message = capsys.readouterr().err
        assert code == 2 and not output.exists(), (update, message)
        assert "replicate refused" in message and expected in message, message
    _settings_returning(monkeypatch, world.settings)
    env = tmp_path / "T1.env"
    code, output = _replicate(swept, tmp_path, client=world.client, extra=["--write-env", str(env)])
    assert code == 0
    replicate = json.loads(output.read_text(encoding="utf-8"))
    assert replicate["base_settings"] == json.loads(swept.sweep.read_text())["base_settings"]
    lines = env.read_text(encoding="utf-8").split()
    assert "SIMILARITY_THRESHOLD=0.45" in lines and "TECHNIQUE_RRF_K=60" in lines
    assert "TECHNIQUE_SEGMENT_ACCEPTANCE=false" in lines


def test_t1_env_pins_the_fixed_settings_against_the_shell(monkeypatch):
    fixed = load_prereg(DEFAULT_PREREG).fixed_settings
    config = full_config(REFERENCE, {"exemplar_k": 128})
    lines = env_assignments(config, fixed=fixed)
    assert len(lines) == sum(1 for knob in KNOBS.values() if knob.env) + len(fixed)
    # a shell (or .env) that drifted after the sweep; env applies T1.env's tokens last
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.4")
    monkeypatch.setenv("TECHNIQUE_RRF_K", "20")
    monkeypatch.setenv("TECHNIQUE_SEGMENT_ACCEPTANCE", "true")
    for token in lines:
        name, value = token.split("=", 1)
        monkeypatch.setenv(name, value)
    loaded = config_module.Settings(_env_file=None)
    assert fixed_setting_problems(loaded, fixed) == []
    assert loaded.technique_exemplar_k == 128
    with pytest.raises(ValueError):  # a knob is never also a fixed setting
        env_assignments(config, fixed={"technique_exemplar_k": 64})


def test_check_fit_refuses_base_settings_the_selection_did_not_record(
    world, harness, fake_settings, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    _, model_path, rows_path, selection, _, _ = _fit_like_artifacts(world, harness, tmp_path)
    arguments = ["check-fit", "--rows", str(rows_path), "--model", str(model_path)]
    arguments += ["--sweep", str(selection)]
    _settings_returning(monkeypatch, world.settings.model_copy(update={"technique_rrf_k": 20}))
    assert tune_retrieval.main(arguments, client=world.client) == 2
    assert "technique_rrf_k" in capsys.readouterr().err
    _settings_returning(monkeypatch, world.settings)
    assert tune_retrieval.main(arguments, client=world.client) == 0


# ----- review fix: the sweep covers exactly the pre-registered dev set -----
def test_sweep_refuses_a_cache_that_is_not_the_preregistered_case_set(
    world, swept, fake_settings, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    cases, _ = load_dev_cases(world.root / "public", ("tram", "expert"), world.kb)  # a subset
    subset = build_cache(
        tmp_path / "cache_subset",
        settings=world.settings,
        kb=world.kb,
        cases=cases,
        client=world.client,
        embedder_factory=BagEmbedder,
        exemplar_source=world.index,
        ann_depths=(90, 150),
        synthetic_ann_depths=(90,),
        log=lambda message: None,
    )
    parity = _run_parity(world, tmp_path / "R21_subset.json", cache=subset)
    output = tmp_path / "R23_subset.json"
    code = tune_retrieval.main(
        _sweep_arguments(subset, swept.prereg, parity, output),
        client=world.client,
        repo_root=swept.repo,
    )
    message = capsys.readouterr().err
    assert code == 2 and not output.exists()
    assert "datasets" in message and "cases" in message, message
    # the full cache against a pre-registration of another case count
    payload = _small_prereg(world)
    payload["dev_only"]["cases"] = 1147
    repo, prereg = _repo_with(tmp_path / "count", payload)
    code = tune_retrieval.main(
        _sweep_arguments(world.cache, prereg, swept.parity, tmp_path / "R23_count.json"),
        client=world.client,
        repo_root=repo,
    )
    assert code == 2 and "1147" in capsys.readouterr().err


def test_bar_fails_a_dataset_criterion_it_cannot_evaluate():
    best = _summary(_counts(0.3), f1=0.40, dataset_f1={"tram": 0.4, "expert": 0.4})
    good = dict(best, f1=0.42, dataset_f1={"tram": 0.4, "expert": 0.4})
    verdict = bar_check(good, best, {"ci95": [0.001, 0.02]}, Bar())
    assert not verdict["passed"]
    assert verdict["checks"]["derived_procedures_f1_drop"]["passed"] is False
    assert verdict["checks"]["tram_f1_drop"]["passed"] is True


# ----- review fix: amendments of the pre-registration are declared, and counted -----
def test_prereg_amendments_must_be_declared(tmp_path):
    payload = _filled_default_prereg()
    repo, target = _repo_with(tmp_path, payload)
    registered = _git(repo, "rev-parse", "HEAD")
    prereg, identity = checked_prereg(target, repo_root=repo)
    assert identity["registered_commit"] == registered and identity["history"] == [registered]
    # after a result: a new variant committed without saying so
    amended = copy.deepcopy(payload)
    amended["families"][0]["variants"].append(
        {"id": "E1a_k512_tau0.05", "set": {"exemplar_k": 512, "exemplar_tau": 0.05}}
    )
    amended["variant_count"] = 28
    target.write_text(json.dumps(amended, indent=2), encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "add k512")
    with pytest.raises(Refused, match="declares 0 amendment"):
        checked_prereg(target, repo_root=repo)
    # declared: the commit it replaces and why (the retry budget is 1)
    amended["amendments"] = [{"amends": registered[:10], "reason": "retry after R23"}]
    target.write_text(json.dumps(amended, indent=2), encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "declare the amendment")
    with pytest.raises(Refused, match="declares 1 amendment"):  # two later commits, one declared
        checked_prereg(target, repo_root=repo)
    history = prereg_git_problems(target, repo_root=repo)[1]["history"]
    assert len(history) == 3 and history[0] == registered
    assert amendment_problems(history, amended["amendments"], allowed=1)
    wrong = [{"amends": "deadbeef00", "reason": "x"}, {"amends": history[1], "reason": "y"}]
    assert any(
        "names deadbeef00" in problem for problem in amendment_problems(history, wrong, allowed=2)
    )
    # squash the two into one declared amendment: accepted, and recorded in the identity
    _git(repo, "reset", "-q", "--soft", registered)
    _git(repo, "commit", "-q", "-m", "amendment: k512, declared")
    prereg, identity = checked_prereg(target, repo_root=repo)
    assert identity["history"][0] == registered and len(identity["history"]) == 2
    assert identity["amendments"] == amended["amendments"]
    # a second amendment exceeds retry.allowed
    second = copy.deepcopy(amended)
    second["amendments"].append({"amends": identity["commit"], "reason": "again"})
    target.write_text(json.dumps(second, indent=2), encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "second amendment")
    with pytest.raises(Refused, match="retry.allowed"):
        checked_prereg(target, repo_root=repo)
    # an amendment entry without a reason is malformed
    bad = dict(payload, amendments=[{"amends": registered}])
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PreregError):
        load_prereg(tmp_path / "bad.json")


def test_replicate_refuses_a_rewritten_registration_commit(
    world, swept, fake_settings, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    repo, prereg = _repo_with(tmp_path, _small_prereg(world))
    sweep = tmp_path / "R23.json"
    assert (
        tune_retrieval.main(
            _sweep_arguments(world.cache, prereg, swept.parity, sweep),
            client=world.client,
            repo_root=repo,
        )
        == 0
    )
    _git(repo, "commit", "-q", "--amend", "-m", "prereg, rewritten")  # same content, new commit
    code = tune_retrieval.main(
        ["replicate", "--sweep", str(sweep), "--output", str(tmp_path / "R24.json")],
        client=world.client,
        repo_root=repo,
    )
    assert code == 2 and "commit" in capsys.readouterr().err


# ----- review fix: the exemplar library is the pre-registered one -----
def test_sweep_refuses_an_exemplar_library_other_than_the_preregistered(
    world, swept, fake_settings, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    for name, change in (
        ("fingerprint", {"fingerprint": "0f8d2e19a59967c63b03ea71"}),
        ("count", {"count": 6360}),
        ("split", {"split": "dev"}),
        # the votes must come from the pre-registered library's backend (Chroma is the source)
        ("backend", {"backend": "chroma"}),
        ("source", {"source_fingerprint": "0f8d2e19a59967c63b03ea71"}),
    ):
        payload = _small_prereg(world)
        payload["fixed"]["exemplar_library"].update(change)
        repo, prereg = _repo_with(tmp_path / name, payload)
        output = tmp_path / f"R23_{name}.json"
        code = tune_retrieval.main(
            _sweep_arguments(world.cache, prereg, swept.parity, output),
            client=world.client,
            repo_root=repo,
        )
        message = capsys.readouterr().err
        assert code == 2 and not output.exists(), name
        assert "exemplar library" in message, message


def test_sweep_and_replicate_refuse_the_library_placeholder(
    world, swept, fake_settings, monkeypatch, tmp_path, capsys
):
    """The Chroma library's fingerprint is known only after sync --verify-only: until
    prereg-fill writes it, the pre-registration cannot run a selection."""
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    payload = _small_prereg(world)
    payload["fixed"]["exemplar_library"]["fingerprint"] = EXEMPLAR_LIBRARY_PLACEHOLDER
    repo, prereg = _repo_with(tmp_path / "placeholder", payload)
    assert load_prereg(prereg).exemplar_library_placeholder
    with pytest.raises(Refused, match="placeholder"):
        checked_prereg(prereg, repo_root=repo)
    output = tmp_path / "R23_placeholder.json"
    code = tune_retrieval.main(
        _sweep_arguments(world.cache, prereg, swept.parity, output),
        client=world.client,
        repo_root=repo,
    )
    assert code == 2 and not output.exists()
    assert "placeholder" in capsys.readouterr().err
    # a replicate of a sweep whose pre-registration now reads as a placeholder is refused too
    sweep = json.loads(swept.sweep.read_text(encoding="utf-8"))
    sweep["prereg"]["path"] = str(prereg)
    forged = tmp_path / "R23_forged.json"
    forged.write_text(json.dumps(sweep), encoding="utf-8")
    replicate = tmp_path / "R24_placeholder.json"
    code = tune_retrieval.main(
        ["replicate", "--sweep", str(forged), "--output", str(replicate)],
        client=world.client,
        repo_root=repo,
    )
    assert code == 2 and not replicate.exists()
    assert "placeholder" in capsys.readouterr().err
    # library_problems names the placeholder even for a prereg loaded without the git check
    assert any(
        "placeholder" in problem
        for problem in tune_retrieval.library_problems(
            world.index, load_prereg(prereg).exemplar_library
        )
    )


def _sync_report(fingerprint: str, *, tolerance: Any = 1e-6, **library_changes) -> dict[str, Any]:
    library = {
        "backend": "chroma",
        "fingerprint": fingerprint,
        "source_fingerprint": "0f8d2e19a59967c63b03ea71",
        "collection": "technique_exemplars-0f8d2e19a59967c63b03ea71-s1-m32-efc400",
        "count": 6360,
        "split": "train",
        # what R10 measured (2026-10-02)
        "tolerance": tolerance,
        "compared_with_source": True,
        "max_abs_diff": 7.45e-9,
        "max_norm_deviation": 6.0e-8,
        **library_changes,
    }
    return {
        "ok": True,
        "tolerance": tolerance,
        "library_fingerprint": fingerprint,
        "library": library,
    }


REGISTERED_LIBRARY_FINGERPRINT = "d8fbb0565e9addeaa3cb048c"


def _unfilled_prereg_text() -> str:
    """The committed pre-registration as it was before prereg-fill wrote the library
    fingerprint (the registered file holds the filled value)."""
    text = DEFAULT_PREREG.read_text(encoding="utf-8")
    assert text.count(json.dumps(REGISTERED_LIBRARY_FINGERPRINT)) == 1
    return text.replace(
        json.dumps(REGISTERED_LIBRARY_FINGERPRINT), json.dumps(EXEMPLAR_LIBRARY_PLACEHOLDER)
    )


def test_prereg_fill_writes_the_chroma_library_fingerprint_and_nothing_else(tmp_path, capsys):
    target = tmp_path / "prereg.json"
    original = _unfilled_prereg_text()
    target.write_text(original, encoding="utf-8")
    fingerprint = "c0ffee" * 4
    assert is_library_fingerprint(fingerprint) and not is_library_fingerprint(
        EXEMPLAR_LIBRARY_PLACEHOLDER
    )
    for report, needle in (
        ({**_sync_report(fingerprint), "ok": False}, "did not pass"),
        (_sync_report("not-a-fingerprint"), "names no library fingerprint"),
        (_sync_report(fingerprint, backend="npz"), "backend"),
        (_sync_report(fingerprint, source_fingerprint="ab" * 12), "copied from"),
        (_sync_report(fingerprint, count=6359), "6359 sentences"),
        (_sync_report(fingerprint, split="dev"), "split 'dev'"),
        (_sync_report(fingerprint, collection="technique_exemplars-other"), "collection"),
    ):
        with pytest.raises(PreregError, match=needle):
            fill_exemplar_library(target, report)
        assert target.read_text(encoding="utf-8") == original  # nothing written on refusal
    report_path = tmp_path / "R10_verify.json"
    report_path.write_text(json.dumps(_sync_report(fingerprint)), encoding="utf-8")
    code = tune_retrieval.main(
        ["prereg-fill", "--sync-report", str(report_path), "--prereg", str(target)]
    )
    assert code == 0 and capsys.readouterr().out.strip() == fingerprint
    filled = target.read_text(encoding="utf-8")
    assert filled == original.replace(
        json.dumps(EXEMPLAR_LIBRARY_PLACEHOLDER), json.dumps(fingerprint)
    )
    prereg = load_prereg(target)
    assert prereg.exemplar_library["fingerprint"] == fingerprint
    assert not prereg.exemplar_library_placeholder
    # a registered fingerprint is never overwritten (only an amendment changes it)
    assert (
        tune_retrieval.main(
            ["prereg-fill", "--sync-report", str(report_path), "--prereg", str(target)]
        )
        == 2
    )
    assert "not the placeholder" in capsys.readouterr().err


def test_the_registered_exemplar_tolerance_is_enforced(tmp_path, monkeypatch):
    """Review fix: fixed.exemplar_library.tolerance (1e-06) was never enforced. prereg-fill took
    a sync report verified at --tolerance 0.5 with components 0.3 off the npz, and nothing
    pinned or checked TECHNIQUE_EXEMPLAR_TOLERANCE. It is now a fixed setting (checked by
    every selection step, recorded, pinned in T1.env), prereg-fill holds the report to it, and
    library_problems holds the loaded Chroma library to it."""
    from tests.test_exemplar_store import _settings, _synced

    prereg = load_prereg(DEFAULT_PREREG)
    fixed = prereg.fixed_settings
    assert prereg.exemplar_library["tolerance"] == fixed["technique_exemplar_tolerance"] == 1e-6
    # 1. prereg-fill: the loosened report of the finding, and every way short of 1e-06
    target = tmp_path / "prereg.json"
    original = _unfilled_prereg_text()
    target.write_text(original, encoding="utf-8")
    fingerprint = "c0ffee" * 4
    for report, needle in (
        (_sync_report(fingerprint, tolerance=0.5, max_abs_diff=0.3), "tolerance 0.5"),
        (_sync_report(fingerprint, max_abs_diff=0.3), "max_abs_diff 0.3"),
        (_sync_report(fingerprint, max_norm_deviation=2e-6), "max_norm_deviation"),
        (_sync_report(fingerprint, max_abs_diff=None), "max_abs_diff None"),
        (_sync_report(fingerprint, compared_with_source=False), "not compared with the npz"),
        (_sync_report(fingerprint, tolerance=None), "tolerance None"),
        (_sync_report(fingerprint, tolerance=float("nan")), "tolerance nan"),
    ):
        with pytest.raises(PreregError, match=needle):
            fill_exemplar_library(target, report)
        assert target.read_text(encoding="utf-8") == original
    # a report verified at a tighter tolerance than registered is fine
    assert fill_exemplar_library(target, _sync_report(fingerprint, tolerance=1e-7)) == fingerprint
    # 2. the pre-registration itself: the library tolerance must be the fixed setting
    payload = json.loads(original)
    for change, needle in (
        ({"technique_exemplar_tolerance": 0.5}, "!= fixed.exemplar_library.tolerance"),
        ({"technique_exemplar_tolerance": None}, "must pin technique_exemplar_tolerance"),
    ):
        broken = copy.deepcopy(payload)
        broken["fixed"]["settings"].update(change)
        if change["technique_exemplar_tolerance"] is None:
            del broken["fixed"]["settings"]["technique_exemplar_tolerance"]
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(PreregError, match=needle):
            load_prereg(path)
    for bad in (-1e-6, "loose", True):
        broken = copy.deepcopy(payload)
        broken["fixed"]["exemplar_library"]["tolerance"] = bad
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(PreregError, match="tolerance"):
            load_prereg(path)
    # 3. a loosened TECHNIQUE_EXEMPLAR_TOLERANCE in the shell: refused by the fixed settings,
    # recorded in every artifact, and T1.env sets it back
    loose = config_module.Settings(_env_file=None, technique_exemplar_tolerance=0.5)
    assert any(
        "technique_exemplar_tolerance" in item for item in fixed_setting_problems(loose, fixed)
    )
    assert "technique_exemplar_tolerance" in RECORDED_SETTINGS
    default_record = tune_retrieval.base_settings_record(config_module.Settings(_env_file=None))
    loose_record = tune_retrieval.base_settings_record(loose)
    assert loose_record["settings"]["technique_exemplar_tolerance"] == 0.5
    assert any(
        "technique_exemplar_tolerance" in item
        for item in tune_retrieval.record_problems(loose_record, default_record, "the sweep")
    )
    lines = env_assignments(full_config(REFERENCE_CONFIG), fixed=fixed)
    assert "TECHNIQUE_EXEMPLAR_TOLERANCE=1e-06" in lines
    monkeypatch.setenv("TECHNIQUE_EXEMPLAR_TOLERANCE", "0.5")
    for token in lines:
        name, value = token.split("=", 1)
        monkeypatch.setenv(name, value)
    assert config_module.Settings(_env_file=None).technique_exemplar_tolerance == 1e-6
    # 4. the loaded Chroma library: verified at the registered tolerance, deviations inside it
    exemplar_store.clear_memo()
    try:
        client, index, name = _synced(normalise_vectors=True)
        path = index.save(tmp_path / "ex.npz")
        vector = index.embeddings[5].astype(np.float64)
        vector[0] += 0.4
        client.collections[name].tamper(
            "ex-00005", embedding=(vector / np.linalg.norm(vector)).astype(np.float32)
        )
        store, reason = exemplar_store.load_chroma_exemplar_store(
            _settings(name, index_path=path, technique_exemplar_tolerance=0.5), client
        )
        assert reason == "ok" and 0.1 < store.verification["max_abs_diff"] <= 0.5
        expected = {
            "fingerprint": store.fingerprint,
            "count": len(index),
            "split": "train",
            "backend": "chroma",
            "source_fingerprint": index.fingerprint,
            "tolerance": 1e-6,
        }
        problems = tune_retrieval.library_problems(store, expected)
        assert any("verified at tolerance 0.5" in item for item in problems), problems
        assert any("max_abs_diff" in item for item in problems), problems
        # the untouched library at the registered tolerance passes
        clean_client, clean_index, clean_name = _synced(normalise_vectors=True)
        clean_path = clean_index.save(tmp_path / "clean.npz")
        clean, reason = exemplar_store.load_chroma_exemplar_store(
            _settings(clean_name, index_path=clean_path, technique_exemplar_tolerance=1e-6),
            clean_client,
        )
        assert reason == "ok"
        registered = {
            **expected,
            "fingerprint": clean.fingerprint,
            "count": len(clean_index),
            "source_fingerprint": clean_index.fingerprint,
        }
        assert tune_retrieval.library_problems(clean, registered) == []
    finally:
        exemplar_store.clear_memo()


def test_p2_compares_the_harness_library_with_the_chroma_library(world, tmp_path):
    """Chroma is the source: a harness that reads the npz while Chroma re-normalised the vectors
    reads another library and fails P2; one that reads the Chroma library passes it."""
    from scripts.tune_retrieval import ClientHolder, load_exemplars, p2_check

    client = FakeClient(normalise_vectors=True)
    name = exemplar_collection_name(world.index.fingerprint, 32, 400)
    create_and_fill(
        client,
        world.index,
        name,
        configuration=creation_configuration(32, 400, 8192),
        source_npz_sha256="abc",
        batch_size=64,
    )
    settings = world.settings.model_copy(
        update={
            "technique_exemplar_backend": "chroma",
            "technique_exemplar_collection": name,
            "technique_exemplar_search": "hnsw",
        }
    )
    exemplar_store.clear_memo()
    try:
        chroma = load_exemplars(settings, ClientHolder(settings, client), "chroma")
        assert chroma.backend == "chroma" and chroma.search == "exact"  # the harness reads exactly
        assert chroma.fingerprint != world.index.fingerprint
        assert chroma.source_fingerprint == world.index.fingerprint
        on_npz = Harness(world.cache, settings, world.kb, world.index, use_rows_cache=False)
        report = p2_check(on_npz, settings, client)
        assert not report["passed"] and report["library"]["same_library"] is False
        on_chroma = Harness(world.cache, settings, world.kb, chroma, use_rows_cache=False)
        report = p2_check(on_chroma, settings, client)
        assert report["passed"] and report["differences"] == 0 and report["queries"] > 0
        assert report["store"]["search"] == "hnsw" and report["search_fallbacks"] == 0
        # the cache was built from the npz: a harness on the Chroma library finds it stale
        problems = world.cache.staleness_problems(settings=settings, exemplar_source=chroma)
        assert any("exemplar library" in problem for problem in problems)
        assert tuning_harness.exemplar_identity(chroma)["backend"] == "chroma"
    finally:
        exemplar_store.clear_memo()
    assert client.writes() == [
        call for call in client.writes() if call[0] in {"create_collection", "add"}
    ]


def test_t1_env_pins_the_exemplar_library(monkeypatch):
    config = full_config(REFERENCE_CONFIG)
    fingerprint = "ab" * 12
    library = {
        "backend": "chroma",
        "collection": "technique_exemplars-x-s1-m32-efc400",
        "fingerprint": fingerprint,
    }
    lines = env_assignments(config, fixed={"similarity_threshold": 0.45}, exemplar_library=library)
    assert lines[-3:] == [
        "TECHNIQUE_EXEMPLAR_BACKEND=chroma",
        "TECHNIQUE_EXEMPLAR_COLLECTION=technique_exemplars-x-s1-m32-efc400",
        f"TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT={fingerprint}",
    ]
    for token in lines:
        name, value = token.split("=", 1)
        monkeypatch.setenv(name, value)
    loaded = config_module.Settings(_env_file=None)
    assert loaded.technique_exemplar_library_fingerprint == fingerprint
    assert env_assignments(config, exemplar_library={"backend": "npz"})[-1] == (
        "TECHNIQUE_EXEMPLAR_BACKEND=npz"
    )
    with pytest.raises(ValueError, match="collection"):
        env_assignments(config, exemplar_library={"backend": "chroma"})
    # a Chroma library is pinned by its fingerprint (the loader refuses any other vectors)
    with pytest.raises(ValueError, match="fingerprint"):
        env_assignments(
            config, exemplar_library={"backend": "chroma", "collection": library["collection"]}
        )
    with pytest.raises(ValueError, match="not a fingerprint"):
        env_assignments(config, exemplar_library={**library, "fingerprint": "XYZ"})
    with pytest.raises(ValueError, match="backend"):
        env_assignments(config, exemplar_library={"backend": "sqlite"})
    for setting in (
        "technique_exemplar_backend",
        "technique_exemplar_collection",
        "technique_exemplar_library_fingerprint",
    ):
        with pytest.raises(ValueError, match="pinned from the exemplar library"):
            env_assignments(config, fixed={setting: "npz"}, exemplar_library=library)
    assert tune_retrieval.library_env(
        SimpleNamespace(backend="chroma", collection_name="c", fingerprint=fingerprint)
    ) == {"backend": "chroma", "collection": "c", "fingerprint": fingerprint}
    assert tune_retrieval.library_env(SimpleNamespace(backend="npz")) == {"backend": "npz"}


# ----- review fix: rethreshold ties the rows to the model and enforces its self-check -----
def _rewrite_dump(source: Path, target: Path, **changes) -> Path:
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays.update(changes)
    with target.open("wb") as handle:
        np.savez(handle, **arrays)
    return target


def test_rethreshold_refuses_rows_of_another_fit_and_a_failed_self_check(
    world, harness, fake_settings, monkeypatch, tmp_path
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    model, model_path, rows_path, _, fitted, oof = _fit_like_artifacts(world, harness, tmp_path)
    contract = _harness_contract(world, harness)

    def rethreshold(rows: Path, output: Path) -> int:
        return tune_retrieval.main(
            ["rethreshold", "--model", str(model_path), "--rows", str(rows)]
            + ["--min-precision", "0.40", "--output", str(output)]
        )

    other_contract = dict(contract, exemplar_k=256)
    for number, summary in enumerate(
        (
            {"threshold": 0.99, "retrieval_contract": contract},  # another fit's threshold
            {"threshold": model.threshold, "retrieval_contract": other_contract},
            {"threshold": model.threshold},  # no contract recorded
        )
    ):
        other = _rewrite_dump(
            rows_path, tmp_path / f"other_{number}.npz", summary=np.asarray(json.dumps(summary))
        )
        output = tmp_path / f"p40_{number}.json"
        assert rethreshold(other, output) == 2 and not output.exists(), summary
    fewer = _rewrite_dump(
        rows_path,
        tmp_path / "fewer.npz",
        **{
            name: np.asarray(value)[:-1]
            for name, value in np.load(rows_path, allow_pickle=False).items()
            if name not in ("summary", "feature_names", "expected_case_ids", "expected_counts")
        },
    )
    assert rethreshold(fewer, tmp_path / "p40_fewer.json") == 2  # row_count differs
    # the dump's probabilities do not reproduce the model's own threshold: nothing written
    blocked, forced = fitted.masks()
    flipped = 1.0 - np.asarray(oof)
    own = choose_threshold_runtime(
        flipped,
        fitted.labels,
        fitted.case_ids,
        fitted.expected_counts,
        blocked=blocked,
        forced=forced,
        technique_ids=fitted.technique_ids,
        min_precision=0.5,
    )
    assert own["threshold"] != model.threshold
    unreproduced = _rewrite_dump(rows_path, tmp_path / "flipped.npz", p_out_of_fold=flipped)
    output = tmp_path / "p40_flipped.json"
    assert rethreshold(unreproduced, output) == 1 and not output.exists()
    assert rethreshold(rows_path, tmp_path / "p40.json") == 0  # its own rows pass
    # a fit that records its dump's sha256 ties the two exactly
    for name, digest in (("wrong", "0" * 64), ("right", tuning_harness.file_sha256(rows_path))):
        linked = AcceptanceModel.load(model_path)
        linked.metadata = dict(linked.metadata, dump_rows={"sha256": digest})
        linked_path = linked.save(tmp_path / f"acceptance_{name}.json")
        code = tune_retrieval.main(
            ["rethreshold", "--model", str(linked_path), "--rows", str(rows_path)]
            + ["--output", str(tmp_path / f"p40_{name}.json")]
        )
        assert code == (2 if name == "wrong" else 0), name


# ----- review fix: sweep runs only after a passing parity run of the same cache -----
def test_sweep_requires_a_passing_parity_run_of_the_same_cache(
    world, swept, fake_settings, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    good = json.loads(swept.parity.read_text(encoding="utf-8"))

    def sweep(parity: Path | None, name: str) -> int:
        output = tmp_path / f"R23_{name}.json"
        code = tune_retrieval.main(
            _sweep_arguments(world.cache, swept.prereg, parity, output),
            client=world.client,
            repo_root=swept.repo,
        )
        assert code != 0 or output.exists()
        return code

    assert sweep(None, "none") == 2
    assert "--parity is required" in capsys.readouterr().err
    failed = copy.deepcopy(good)
    failed["P1"]["cues_on"]["differing_cases"] = 4
    failed["blocking"]["P1"] = failed["passed"] = False
    other_cache = dict(copy.deepcopy(good), cache=dict(good["cache"], key="0" * 64))
    other_settings = copy.deepcopy(good)
    other_settings["base_settings"]["retrieval_kwargs"]["rrf_k"] = 20
    other_code = copy.deepcopy(good)
    other_code["code"]["harness"] = "0" * 64
    copy_run = dict(copy.deepcopy(good), config=dict(good["config"], dense_search="copy"))
    for name, payload, expected in (
        ("failed", failed, "P1"),
        ("other_cache", other_cache, "another cache"),
        ("other_settings", other_settings, "rrf_k"),
        ("other_code", other_code, "harness code"),
        ("copy_run", copy_run, "reference"),
    ):
        path = tmp_path / f"parity_{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert sweep(path, name) == 2, name
        assert expected in capsys.readouterr().err, name


# ----- review fix: check-fit tolerates the fit's embedding path (each case alone) -----
class AloneEmbedder(BagEmbedder):
    """The fit's embedding path: each call embeds one case's few texts, and the low bits of a
    vector move with the batch it was embedded in (as BERT's padded batches do)."""

    def embed(self, texts, batch_size=None, **_):
        texts = list(texts)
        output = []
        for text, vector in zip(texts, super().embed(texts), strict=True):
            rng = np.random.default_rng(zlib.crc32(f"{len(texts)}|{text}".encode()))
            moved = np.asarray(vector, dtype=np.float32) + 1e-7 * rng.normal(size=DIM)
            output.append(moved.astype(np.float32).tolist())
        return output


def test_check_fit_passes_a_fit_that_embedded_each_case_alone(
    world, harness, fake_settings, monkeypatch, tmp_path
):
    from scripts.fit_acceptance_model import extract_rows

    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    settings = harness.settings_for(REFERENCE_CONFIG)
    store = ChromaVectorStore(settings, AloneEmbedder(), client=world.client)
    fitted = extract_rows(
        [
            TechniqueEvaluationCase(
                case_id=case.case_id,
                document_id=case.case_id,
                text=case.text,
                expected_technique_ids=case.expected,
                source=case.source,
            )
            for case in world.cache.cases
        ],
        settings=settings,
        kb=world.kb,
        store=store,
        lexical_index=LexicalTechniqueIndex.from_kb(world.kb, fields=lexical_fields(settings)),
        priors=None,
        exemplar_index=world.index,
    )
    contract = retrieval_contract_for(settings, world.index, store=store)
    _, model_path, rows_path = _fit_artifacts(world, harness, tmp_path, fitted, contract=contract)
    selection = _selection_artifact(world, tmp_path / "R24.json")
    output = tmp_path / "check.json"
    code = tune_retrieval.main(
        ["check-fit", "--rows", str(rows_path), "--model", str(model_path)]
        + ["--sweep", str(selection), "--output", str(output)],
        client=world.client,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0, report["checks"]
    # the bitwise rule this replaces would have failed: the low bits differ on most rows
    assert report["rows"]["bitwise_share"] < CHECK_FIT_ROW_SHARE
    assert 0 < report["rows"]["largest_feature_gap"] <= tune_retrieval.CHECK_FIT_FEATURE_ATOL
    assert report["checks"]["accepted_sets_share"]["passed"]
    assert report["checks"]["retrieval_contract"]["value"] == []


# ----- review fix: exact dense search is replicated on the collection copy that serves it -----
def test_served_config_reads_exact_search_from_the_copy():
    assert (
        served_config(full_config(REFERENCE, {"dense_search": "exact"}))["dense_search"] == "copy"
    )
    assert served_config(REFERENCE) == REFERENCE


def _relaxed_e2a_prereg(world: World) -> dict[str, Any]:
    """Only E2a, and a bar and replication rule every variant clears (the fake data cannot be
    relied on to favour exact search)."""
    payload = _small_prereg(world)
    payload["families"] = [
        {
            "id": "E2a",
            "group": "E2",
            "variants": [{"id": "E2a_exact", "set": {"dense_search": "exact"}}],
        }
    ]
    payload["variant_count"] = 1
    bar = payload["selection"]["bar"]
    bar.update(
        {
            "min_delta_f1": -1.0,
            "min_precision": 0.0,
            "dataset_max_f1_drop": {"tram": 1.0, "derived_procedures": 1.0},
            "min_delta_f1_no_cues": -1.0,
            "min_delta_hit_at_10": -1.0,
            "max_generic_share_rise": 1.0,
            "max_accepted_per_case_rise_relative": 10.0,
        }
    )
    bar["bootstrap"]["ci_lower_above"] = -1.0
    payload["operating_points"]["T1"]["min_precision"] = 0.0
    payload["replication"]["min_delta_f1_per_seed"] = -1.0
    payload["replication"]["pooled_bootstrap_ci_lower_above"] = -1.0
    return payload


def test_copy_check_replicates_exact_search_on_the_collection_copy(tmp_path, monkeypatch, capsys):
    copy_world = _make_world(tmp_path / "world")
    monkeypatch.setattr(tuning_harness, "REPO_ROOT", tmp_path)
    _settings_returning(monkeypatch, copy_world.settings)
    source = copy_world.client.get_collection(name="attack_techniques", embedding_function=None)
    copy_name = "attack_techniques-s4-abcdef123456-m32-efc400"
    create_copy(
        copy_world.client,
        source,
        copy_name,
        read_records(source),
        configuration={"hnsw": {"space": "cosine", "max_neighbors": 32, "ef_construction": 400}},
    )
    copy_cache = build_cache(
        tmp_path / "cache_copy",
        settings=copy_world.settings,
        kb=copy_world.kb,
        cases=copy_world.cases,
        client=copy_world.client,
        embedder_factory=BagEmbedder,
        exemplar_source=copy_world.index,
        ann_depths=(90, 150),
        synthetic_ann_depths=(90,),
        copy_collection=copy_name,
        log=lambda message: None,
    )
    repo, prereg = _repo_with(tmp_path, _relaxed_e2a_prereg(copy_world))
    parity = _run_parity(copy_world, tmp_path / "R21.json")
    sweep = tmp_path / "R23.json"
    assert (
        tune_retrieval.main(
            _sweep_arguments(copy_world.cache, prereg, parity, sweep),
            client=copy_world.client,
            repo_root=repo,
        )
        == 0
    )
    replicate = tmp_path / "R24.json"
    env = tmp_path / "T1.env"
    code = tune_retrieval.main(
        ["replicate", "--sweep", str(sweep), "--output", str(replicate), "--write-env", str(env)],
        client=copy_world.client,
        repo_root=repo,
    )
    payload = json.loads(replicate.read_text(encoding="utf-8"))
    assert payload["selected"]["id"] == "E2a_exact" and payload["selected"]["env"] is None
    assert code == 1 and not env.exists()  # exact search runs only from the copy
    # check-fit never compares a fit against exact rows (the fit read the copy)
    assert (
        tune_retrieval.main(
            ["check-fit", "--rows", "x.npz", "--model", "m.json", "--sweep", str(replicate)],
            client=copy_world.client,
        )
        == 2
    )
    assert "collection copy" in capsys.readouterr().err
    arguments = ["copy-check", "--replicate", str(replicate), "--output"]
    # the sweep's own cache holds no copy
    assert (
        tune_retrieval.main(
            [*arguments, str(tmp_path / "x.json"), "--cache", str(copy_world.cache.path)]
            + ["--parity", str(parity)],
            client=copy_world.client,
            repo_root=repo,
        )
        == 2
    )
    assert "no collection copy" in capsys.readouterr().err
    # P1 on the copy, then the replication again on the copy's hits
    _settings_returning(
        monkeypatch, copy_world.settings.model_copy(update={"technique_collection_name": copy_name})
    )
    copy_parity = _run_parity(
        copy_world, tmp_path / "R21c.json", cache=copy_cache, extra=["--dense-source", "copy"]
    )
    _settings_returning(monkeypatch, copy_world.settings)
    output = tmp_path / "R24c.json"
    code = tune_retrieval.main(
        [*arguments, str(output), "--cache", str(copy_cache.path), "--parity", str(copy_parity)]
        + ["--write-env", str(env)],
        client=copy_world.client,
        repo_root=repo,
    )
    assert code == 0, capsys.readouterr().err
    checked = json.loads(output.read_text(encoding="utf-8"))
    assert checked["same_as_replicate"] is True
    assert checked["selected"]["config"]["dense_search"] == "copy"
    assert [entry["config"]["dense_search"] for entry in checked["path"]] == ["ann", "copy"]
    assert checked["exact_vs_copy"]["cues_on"]["cases"] == len(copy_world.cache.cases)
    lines = env.read_text(encoding="utf-8").split()
    assert lines[-1] == f"TECHNIQUE_COLLECTION_NAME={copy_name}"
    assert "SIMILARITY_THRESHOLD=0.45" in lines
    assert (tmp_path / "T1.fit_args").read_text(encoding="utf-8").split() == ["--l2", "1.0"]
