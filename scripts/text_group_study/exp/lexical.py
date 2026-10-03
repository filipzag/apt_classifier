"""Experiment "lexical": words only, no embeddings (reference for what ATTACK-BERT adds).

Two model families over Doc.texts / train.records[i]["text"]:
- groupdoc: one concatenated TF-IDF document per group (its own procedure sentences, its
  description sentences, and the procedure sentences of every software it uses, weighted by
  ``sw_weight``); cosine against the whole report, z-scored per report, plus ``prior`` x log prior.
- sentlr: sentence-level TF-IDF -> multinomial logistic regression with soft labels (a software
  sentence's mass is shared by all groups using that software), trained full-batch in torch
  (float32, Adam, L2 = 1/(2 C n)), aggregated over the report's sentences.
- combo: per-report z-scored sum of a groupdoc and a sentlr model.

Group names in the text are already "the adversary"; the word "adversary" is removed before
tokenisation (it is in every group sentence and carries no group signal). Malware and tool names
stay in. Run as __main__ to perform the pre-planned staged dev search (dev_eval only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the study code folder (harness)
import harness  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import (  # noqa: E402
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.preprocessing import normalize  # noqa: E402

torch.set_num_threads(4)

KEY = "lexical"
HERE = harness.STUDY_DIR / "exp"  # result (data, not code)
EPS = 0.01  # a more complex option must beat the simpler one by more than this on dev
STOP = sorted(ENGLISH_STOP_WORDS | {"adversary"})
_ADV = re.compile(r"\bthe adversary(?:'s|s)?\b|\badversary(?:'s|s)?\b")


def _clean(text: str) -> str:
    return _ADV.sub(" ", text.lower())


def _vectorizer(analyzer: str, tfidf: bool, max_features=None):
    common = dict(preprocessor=_clean, min_df=2, max_features=max_features, dtype=np.float32)
    if analyzer == "word":
        kw = dict(analyzer="word", ngram_range=(1, 2), stop_words=STOP, **common)
    elif analyzer == "char":
        kw = dict(analyzer="char_wb", ngram_range=(3, 5), **common)
    else:
        raise ValueError(analyzer)
    return TfidfVectorizer(sublinear_tf=True, **kw) if tfidf else CountVectorizer(**kw)


def _row_weight(rec, sw_weight):
    return sw_weight if rec["source"] == "software" else 1.0


def _fingerprint(train) -> str:
    h = hashlib.sha1()
    for r in train.records:
        h.update(r["text"].encode())
        h.update(r["source"].encode())
        h.update(",".join(r["groups"]).encode())
    return h.hexdigest()


_FIT_CACHE: dict = {}


def _cached(kind, params, train, builder):
    key = (kind, json.dumps(params, sort_keys=True), _fingerprint(train))
    if key not in _FIT_CACHE:
        if len(_FIT_CACHE) >= 3:
            _FIT_CACHE.pop(next(iter(_FIT_CACHE)))
        _FIT_CACHE[key] = builder()
    return _FIT_CACHE[key]


def _zscore(S):
    S = np.asarray(S, np.float64)
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------- (a) one document per group
def _fit_groupdoc(train, analyzer, sw_weight):
    texts = [r["text"] for r in train.records]
    cv = _vectorizer(analyzer, tfidf=False)
    C = cv.fit_transform(texts).tocsr()
    gi = {g: i for i, g in enumerate(train.groups)}
    rows, cols, vals = [], [], []
    for i, r in enumerate(train.records):
        w = _row_weight(r, sw_weight)
        if w <= 0:
            continue
        for g in r["groups"]:
            rows.append(i)
            cols.append(gi[g])
            vals.append(w)
    Wg = sp.csr_matrix(
        (np.asarray(vals, np.float32), (rows, cols)), shape=(len(train.records), len(train.groups))
    )
    G = (Wg.T @ C).tocsr()
    G.eliminate_zeros()
    n_groups, V = G.shape
    df = np.bincount(G.indices, minlength=V)
    idf = (np.log((1.0 + n_groups) / (1.0 + df)) + 1.0).astype(np.float32)
    G.data = np.log1p(G.data)
    G = normalize(G @ sp.diags(idf), norm="l2")
    return cv, idf, G.T.tocsc()


def _groupdoc(analyzer="word", sw_weight=1.0, prior=0.0):
    def fit(train):
        cv, idf, GT = _cached(
            "groupdoc",
            dict(analyzer=analyzer, sw_weight=sw_weight),
            train,
            lambda: _fit_groupdoc(train, analyzer, sw_weight),
        )
        log_prior = np.asarray(train.log_prior, np.float64)

        def predict(docs):
            R = cv.transform([" ".join(d.texts) for d in docs]).tocsr()
            R.data = np.log1p(R.data)
            R = normalize(R @ sp.diags(idf), norm="l2")
            S = np.asarray((R @ GT).todense())
            return _zscore(S) + prior * log_prior[None, :]

        return predict

    return fit


# ---------------------------------------------------------------- (b) sentence TF-IDF + LR
def _fit_sentlr(train, analyzer, C, sw_weight, balanced, seed, steps=400, lr=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gi = {g: i for i, g in enumerate(train.groups)}
    keep, rows, cols, vals = [], [], [], []
    for i, r in enumerate(train.records):
        w = _row_weight(r, sw_weight)
        if w <= 0 or not r["groups"]:
            continue
        k = len(keep)
        keep.append(i)
        for g in r["groups"]:
            rows.append(k)
            cols.append(gi[g])
            vals.append(w / len(r["groups"]))
    n_groups = len(train.groups)
    Y = np.zeros((len(keep), n_groups), np.float32)
    Y[rows, cols] = vals  # each (row, group) pair is unique
    if balanced:
        mass = Y.sum(0)
        cw = np.where(mass > 0, mass.mean() / np.maximum(mass, 1e-9), 0.0).astype(np.float32)
        Y = Y * cw[None, :]
    total = float(Y.sum())
    vec = _vectorizer(analyzer, tfidf=True, max_features=100_000 if analyzer == "char" else None)
    X = vec.fit_transform([train.records[i]["text"] for i in keep]).tocoo()
    Xt = torch.sparse_coo_tensor(
        np.vstack([X.row, X.col]).astype(np.int64),
        X.data.astype(np.float32),
        X.shape,
        check_invariants=True,
    ).coalesce()
    Yt = torch.from_numpy(Y)
    V = X.shape[1]
    lam = 1.0 / (2.0 * C * total)
    W = torch.zeros(V, n_groups, requires_grad=True)
    b = torch.zeros(n_groups, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    for _ in range(steps):
        opt.zero_grad()
        logp = torch.log_softmax(torch.sparse.mm(Xt, W) + b, dim=1)
        loss = -(Yt * logp).sum() / total + lam * (W * W).sum()
        loss.backward()
        opt.step()
        sched.step()
    return vec, W.detach().numpy(), b.detach().numpy(), float(loss.item())


def _log_softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def _sentlr(
    analyzer="word", C=10.0, sw_weight=1.0, balanced=False, agg="mean_logp", prior=0.0, seed=0
):
    def fit(train):
        vec, W, b, _ = _cached(
            "sentlr",
            dict(analyzer=analyzer, C=C, sw_weight=sw_weight, balanced=balanced, seed=seed),
            train,
            lambda: _fit_sentlr(train, analyzer, C, sw_weight, balanced, seed),
        )
        log_prior = np.asarray(train.log_prior, np.float64)

        def predict(docs):
            out = []
            for d in docs:
                if agg == "doc":
                    x = vec.transform([" ".join(d.texts)])
                    s = _log_softmax(np.asarray(x @ W) + b)[0]
                else:
                    lp = _log_softmax(np.asarray(vec.transform(d.texts) @ W) + b)
                    if agg == "mean_logp":
                        s = lp.mean(0)
                    elif agg == "log_mean_p":
                        s = np.log(np.exp(lp).mean(0) + 1e-12)
                    else:
                        raise ValueError(agg)
                out.append(s)
            return np.stack(out).astype(np.float64) + prior * log_prior[None, :]

        return predict

    return fit


# ---------------------------------------------------------------- combination
def _combo(a, b, alpha=1.0):
    def fit(train):
        pa, pb = _groupdoc(**a)(train), _sentlr(**b)(train)

        def predict(docs):
            return _zscore(pa(docs)) + alpha * _zscore(pb(docs))

        return predict

    return fit


def make_fit(model="sentlr", **config):
    if model == "groupdoc":
        return _groupdoc(**config)
    if model == "sentlr":
        return _sentlr(**config)
    if model == "combo":
        return _combo(**config)
    raise ValueError(model)


# Chosen by the staged dev search below (see lexical.json for every trial).
# groupdoc, word 1-2 grams, software sentences at full weight, prior 0.5: dev selection 1.4334.
BEST = {"model": "groupdoc", "analyzer": "word", "sw_weight": 1.0, "prior": 0.5}


# ---------------------------------------------------------------- staged dev search
TRIALS: list = []


def _run(name, config):
    t0 = time.time()
    m = harness.dev_eval(make_fit(**config))
    m.pop("ranks", None)
    row = {"name": name, "config": config, "dev": m, "seconds": round(time.time() - t0, 1)}
    TRIALS.append(row)
    print(json.dumps(row), flush=True)
    return m["selection_score"]


def _pick(cands):
    """cands: [(config, score)] in order of simplicity; keep a later one only if it beats the
    current choice by more than EPS."""
    best_c, best_s = cands[0]
    for c, s in cands[1:]:
        if s > best_s + EPS:
            best_c, best_s = c, s
    return best_c, best_s


def search():
    # Stage A: one TF-IDF document per group, cosine vs the whole report.
    a = {"model": "groupdoc", "analyzer": "word", "sw_weight": 1.0, "prior": 0.0}
    a_char = dict(a, analyzer="char")
    a, sa = _pick([(a, _run("A1 groupdoc word", a)), (a_char, _run("A2 groupdoc char", a_char))])
    a_sw = dict(a, sw_weight=0.3)
    a, sa = _pick([(a, sa), (a_sw, _run("A3 groupdoc sw0.3", a_sw))])
    a_p5, a_p10 = dict(a, prior=0.5), dict(a, prior=1.0)
    a, sa = _pick(
        [
            (a, sa),
            (a_p5, _run("A4 groupdoc prior0.5", a_p5)),
            (a_p10, _run("A5 groupdoc prior1.0", a_p10)),
        ]
    )

    # Stage B: sentence TF-IDF + logistic regression, aggregated over the report.
    b = {
        "model": "sentlr",
        "analyzer": "word",
        "C": 10.0,
        "sw_weight": 1.0,
        "balanced": False,
        "agg": "mean_logp",
        "prior": 0.0,
        "seed": 0,
    }
    b_lmp, b_doc = dict(b, agg="log_mean_p"), dict(b, agg="doc")
    b, sb = _pick(
        [
            (b, _run("B1 sentlr word C10 mean_logp", b)),
            (b_lmp, _run("B2 sentlr agg log_mean_p", b_lmp)),
            (b_doc, _run("B3 sentlr agg doc", b_doc)),
        ]
    )
    b_c1, b_c100 = dict(b, C=1.0), dict(b, C=100.0)
    b, sb = _pick(
        [(b, sb), (b_c1, _run("B4 sentlr C1", b_c1)), (b_c100, _run("B5 sentlr C100", b_c100))]
    )
    b_char = dict(b, analyzer="char")
    b, sb = _pick([(b, sb), (b_char, _run("B6 sentlr char", b_char))])
    b_sw = dict(b, sw_weight=0.3)
    b, sb = _pick([(b, sb), (b_sw, _run("B7 sentlr sw0.3", b_sw))])
    b_bal = dict(b, balanced=True)
    b, sb = _pick([(b, sb), (b_bal, _run("B8 sentlr balanced", b_bal))])
    b_p5, b_p10 = dict(b, prior=0.5), dict(b, prior=1.0)
    b, sb = _pick(
        [
            (b, sb),
            (b_p5, _run("B9 sentlr prior0.5", b_p5)),
            (b_p10, _run("B10 sentlr prior1.0", b_p10)),
        ]
    )

    # Stage C: z-scored sum of the two winners.
    ac = {k: v for k, v in a.items() if k != "model"}
    bc = {k: v for k, v in b.items() if k != "model"}
    c = {"model": "combo", "a": ac, "b": bc, "alpha": 1.0}
    sc = _run("C1 combo groupdoc+sentlr", c)

    # Simplicity order: groupdoc, sentlr, combo.
    best, best_s = _pick([(a, sa), (b, sb), (c, sc)])
    return best, best_s


def seed_runs(best):
    """Three seeds of the chosen config, each fitted uncached (the LR is convex, zero-initialised
    and full-batch, so the seed only matters if a stochastic component is chosen)."""
    scores, first, secs = [], None, []
    for s in (0, 1, 2):
        cfg = json.loads(json.dumps(best))
        if cfg["model"] == "sentlr":
            cfg["seed"] = s
        elif cfg["model"] == "combo":
            cfg["b"]["seed"] = s
        _FIT_CACHE.clear()
        t0 = time.time()
        m = harness.dev_eval(make_fit(**cfg))
        secs.append(round(time.time() - t0, 1))
        m.pop("ranks", None)
        TRIALS.append(
            {"name": f"seed {s}", "config": cfg, "dev": m, "seed_run": True, "seconds": secs[-1]}
        )
        print("seed", s, json.dumps(m), flush=True)
        scores.append(m["selection_score"])
        first = first or m
    return scores, first, secs


if __name__ == "__main__":
    t0 = time.time()
    best, best_s = search()
    seeds, m, secs = seed_runs(best)
    out = {
        "best": best,
        "dev": m,
        "trials": TRIALS,
        "seed_scores": seeds,
        "seed_mean": float(np.mean(seeds)),
        "seed_std": float(np.std(seeds)),
        "fit_plus_eval_seconds_uncached": secs,
        "total_seconds": round(time.time() - t0, 1),
        "dev_eval_calls": len(TRIALS),
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / f"{KEY}.json").write_text(json.dumps(out, indent=1))
    print("BEST", json.dumps(best), json.dumps(m))
