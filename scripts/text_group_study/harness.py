"""Shared evaluation harness for free-text -> threat-group models (scratch, ATT&CK-only training).

Data (built by build_corpus.py, export_tests.py, embed.py in this folder; stored in STUDY_DIR,
``$TEXT_GROUP_STUDY_DIR`` or <repo>/data/text_group_study):
- corpus.json  labelled ATT&CK text (sources: group, software, group_desc, campaign)
- test_sets.json  CTI-HAL / Elastic report texts at the benchmarks' anonymization levels
- emb.npz  ATTACK-BERT embeddings (L2-normalised, 768-d) of every unique text

Protocol (pre-registered; experiment code must follow it):
- DEV (model selection): the 23 single-group ATT&CK campaigns. Training excludes every campaign
  text and every training text with cosine >= DUP_COS to any campaign text.
  Call ``dev_eval(fit)`` only.
- TEST (reported, never used to choose): ``test_eval(fit)`` refits in FINAL mode (all ATT&CK text,
  campaigns included) and scores CTI-HAL (headline: names) and Elastic (headline: strict).
  Only the orchestrator runs it, once per finalist.

A model is a function ``fit(train) -> predict`` where ``train`` is a ``Train`` and ``predict(docs)``
takes a list of Doc (an (n_i, 768) array whose .texts holds the sentences) and returns an (n_docs, n_groups) score matrix over
``GROUPS`` (higher = more likely). Scores are ranked; ties broken by group id.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(os.environ.get("TEXT_GROUP_STUDY_DIR", REPO / "data" / "text_group_study"))
DUP_COS = 0.95
KB_PATH = REPO / "data" / "kb" / "attack_kb.json"
AUDITS = Path(os.environ.get("TEXT_GROUP_AUDITS_DIR", REPO / "data" / "audits"))

_cache: dict = {}


def _load():
    if _cache:
        return _cache
    data = np.load(STUDY_DIR / "emb.npz", allow_pickle=True)
    vectors, texts = data["vectors"].astype(np.float32), list(data["texts"])
    index = {t: i for i, t in enumerate(texts)}
    corpus = json.loads((STUDY_DIR / "corpus.json").read_text())
    tests = json.loads((STUDY_DIR / "test_sets.json").read_text())
    kb = json.loads(KB_PATH.read_text())
    groups = tuple(sorted(kb["groups"]))
    prominence = {}
    for gid, g in kb["groups"].items():
        prominence[gid] = (
            len(g.get("external_references", []))
            + len(g.get("aliases", []))
            + len(g.get("campaign_ids", []))
            + len(g.get("software_ids", []))
        )
    _cache.update(
        vectors=vectors,
        index=index,
        corpus=corpus,
        tests=tests,
        groups=groups,
        prominence=prominence,
        kb=kb,
    )
    return _cache


def vec(texts):
    c = _load()
    return c["vectors"][[c["index"][t] for t in texts]]


class Doc(np.ndarray):
    """A report: its (n, 768) sentence embeddings, with the sentences themselves in .texts."""

    def __array_finalize__(self, obj):
        self.texts = getattr(obj, "texts", None)


def doc(texts):
    d = vec(texts).view(Doc)
    d.texts = list(texts)
    return d


@dataclass
class Train:
    records: list  # corpus records: text, source, owner, groups, technique_id
    X: np.ndarray  # (n, 768) embeddings aligned with records
    groups: tuple  # the label space (every ATT&CK group id, sorted)
    log_prior: np.ndarray  # centred log(prominence + 1), aligned with groups
    group_software: dict  # group id -> set of software ids it uses (ATT&CK)

    def label_matrix(self, sources=("group", "software", "group_desc", "campaign")):
        """(n, n_groups) soft labels: a text's groups share its unit of mass (empty -> zero row)."""
        gi = {g: i for i, g in enumerate(self.groups)}
        Y = np.zeros((len(self.records), len(self.groups)), np.float32)
        for row, r in enumerate(self.records):
            if r["source"] in sources and r["groups"]:
                for g in r["groups"]:
                    Y[row, gi[g]] = 1.0 / len(r["groups"])
        return Y


def _train(mode: str) -> Train:
    c = _load()
    corpus = c["corpus"]
    campaign = [r for r in corpus if r["source"] == "campaign"]
    rest = [r for r in corpus if r["source"] != "campaign"]
    if mode == "dev":
        Xr, Xc = vec([r["text"] for r in rest]), vec([r["text"] for r in campaign])
        near = np.zeros(len(rest), bool)
        for s in range(0, len(rest), 4096):
            near[s : s + 4096] = (Xr[s : s + 4096] @ Xc.T).max(1) >= DUP_COS
        records = [r for r, n in zip(rest, near, strict=False) if not n]
        _cache["dev_dropped_near_dups"] = int(near.sum())
    elif mode == "final":
        records = rest + [r for r in campaign if r["groups"]]
    else:
        raise ValueError(mode)
    groups = c["groups"]
    values = np.log(np.asarray([c["prominence"].get(g, 0) + 1.0 for g in groups]))
    gs = {}
    for r in corpus:
        if r["source"] == "software":
            for g in r["groups"]:
                gs.setdefault(g, set()).add(r["owner"])
    return Train(
        records,
        vec([r["text"] for r in records]),
        groups,
        (values - values.mean()).astype(np.float32),
        gs,
    )


def dev_docs():
    c = _load()
    by = {}
    for r in c["corpus"]:
        if r["source"] == "campaign" and len(r["groups"]) == 1:
            by.setdefault(r["owner"], []).append(r)
    ids = sorted(k for k, v in by.items() if len(v) >= 3)
    return (
        [doc([r["text"] for r in by[k]]) for k in ids],
        [by[k][0]["groups"][0] for k in ids],
        [frozenset() for _ in ids],
        ids,
    )


def test_docs(name: str):
    rows = [r for r in _load()["tests"][name] if r["group_id"]]
    return (
        [doc(r["texts"]) for r in rows],
        [r["group_id"] for r in rows],
        [frozenset(r["accept"]) for r in rows],
        [r["report_id"] for r in rows],
    )


def open_docs(name: str):
    rows = [r for r in _load()["tests"][name] if not r["group_id"]]
    return [doc(r["texts"]) for r in rows], [r["report_id"] for r in rows]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(centre - half, 3), round(centre + half, 3))


def metrics(scores, truth, accept, groups):
    """top1/top3/top5/MRR and macro-F1 over the true classes; a prediction in ``accept`` counts."""
    order = np.lexsort(
        (np.arange(scores.shape[1])[None, :].repeat(len(scores), 0), -scores), axis=1
    )
    ranks, top1 = [], []
    for row, t, acc in zip(order, truth, accept, strict=False):
        ok = {t} | set(acc)
        ranked = [groups[i] for i in row]
        r = next((k + 1 for k, g in enumerate(ranked) if g in ok), None)
        ranks.append(r)
        top1.append(t if ranked[0] in ok else ranked[0])
    n = len(truth)
    f1s = []
    for label in sorted(set(truth)):
        tp = sum(1 for t, p in zip(truth, top1, strict=False) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, top1, strict=False) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, top1, strict=False) if t == label and p != label)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    k1 = sum(1 for r in ranks if r == 1)
    return {
        "n": n,
        "top1": round(k1 / n, 4),
        "top3": round(sum(1 for r in ranks if r and r <= 3) / n, 4),
        "top5": round(sum(1 for r in ranks if r and r <= 5) / n, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "top1_wilson": wilson(k1, n),
        "ranks": ranks,
    }


def dev_eval(fit):
    """Fit in DEV mode and score the 23 campaigns. The only call experiments may make."""
    train = _train("dev")
    predict = fit(train)
    docs, truth, accept, _ = dev_docs()
    m = metrics(np.asarray(predict(docs)), truth, accept, train.groups)
    m["selection_score"] = round(m["mrr"] + m["macro_f1"], 4)
    return m


TEST_SETS = (
    "cti_hal_names",
    "cti_hal_as_reported",
    "cti_hal_strict",
    "cti_hal_tradecraft",
    "elastic_strict",
    "elastic_names",
    "elastic_tradecraft",
)


def test_eval(fit, sets=TEST_SETS):
    """Orchestrator only: fit in FINAL mode, score every test set and the open-set posts."""
    train = _train("final")
    predict = fit(train)
    out = {}
    for name in sets:
        docs, truth, accept, ids = test_docs(name)
        s = np.asarray(predict(docs))
        m = metrics(s, truth, accept, train.groups)
        m["report_ids"] = ids
        out[name] = m
    docs, _ = open_docs("elastic_strict")
    labelled, _, _, _ = test_docs("elastic_strict")

    def top_prob(s):
        s = s - s.max(1, keepdims=True)
        p = np.exp(s) / np.exp(s).sum(1, keepdims=True)
        return p.max(1)

    out["elastic_open_set_top_prob"] = {
        "open_mean": round(float(top_prob(np.asarray(predict(docs))).mean()), 4),
        "labelled_mean": round(float(top_prob(np.asarray(predict(labelled))).mean()), 4),
    }
    return out


def pipeline_baselines():
    """The shipped pipeline's numbers on the same reports (local, deterministic artifacts)."""
    out = {}
    for name, f in (
        ("cti_hal_names", "attribution_cti_hal_current.json"),
        ("cti_hal_as_reported", "attribution_cti_hal_as_reported.json"),
        ("elastic_strict", "attribution_elastic_current.json"),
        ("elastic_tradecraft", "attribution_elastic_tradecraft.json"),
        ("elastic_names", "attribution_elastic_names.json"),
    ):
        p = AUDITS / f
        if p.is_file():
            d = json.loads(p.read_text())
            m = d["metrics"]
            out[name] = {
                k: m.get(k) for k in ("top1", "top3", "top5", "mrr", "named_correct", "known_count")
            }
            out[name]["generated_at"] = d.get("generated_at")
    return out
