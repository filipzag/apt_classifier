"""Experiment "knn": sentence-level nearest-neighbour voting over ATTACK-BERT embeddings.

make_fit(**config) -> fit(train) -> predict(docs) -> (n_docs, n_groups) scores.

Options (defaults reproduce baselines.knn(k=10) up to a scale-free epsilon):
- k, beta: neighbours per report sentence; vote weight exp(beta * (s_eff - 1)).
- drop_unlabelled: remove training rows with no group (software no group uses) before the search.
- w_group / w_software / w_desc: per-source vote weights (campaign rows, FINAL mode only, use w_group).
- hub: 'none' or 'csls' -> s_eff = s - hub_lam * (r_x - median r), r_x = mean cosine of training
  sentence x to its hub_k nearest training sentences (penalises hubs). Used for search and weights.
- spread_alpha: training-side genericity weight exp(H_x)^-alpha, H_x = entropy of the group labels
  among x's spread_m nearest training sentences (x excluded).
- class_norm: divide each sentence's votes for g by (training label mass of g)^class_norm.
- agg: 'sum' (sum votes over sentences, log), 'mean' (mean of per-sentence normalised
  distributions, log), 'logsum' (sum over sentences of log((1-smooth) p_s + smooth / G)).
- sent_gamma: per-sentence weight (max_g p_s[g])^sent_gamma (down-weights generic sentences).
- prior: + prior * train.log_prior.
Deterministic (no seeds).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the study code folder (harness)
import harness  # noqa: E402

DEFAULTS = dict(
    k=10,
    beta=10.0,
    drop_unlabelled=False,
    w_group=1.0,
    w_software=1.0,
    w_desc=1.0,
    hub="none",
    hub_lam=0.5,
    hub_k=10,
    spread_alpha=0.0,
    spread_m=20,
    class_norm=0.0,
    agg="sum",
    smooth=0.5,
    sent_gamma=0.0,
    prior=0.0,
)

# Chosen by the staged search below (see knn.json for every trial).
BEST = dict(DEFAULTS, beta=5.0, w_software=0.2, hub="csls", hub_lam=0.5, agg="logsum")


def _train_topk(X, m, chunk=1024):
    """Indices and sims of each training row's m nearest other training rows (self excluded)."""
    n = len(X)
    idx = np.empty((n, m), np.int64)
    sims = np.empty((n, m), np.float32)
    for s in range(0, n, chunk):
        S = X[s : s + chunk] @ X.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
        part = np.argpartition(-S, m, axis=1)[:, :m]
        idx[s : s + chunk] = part
        sims[s : s + chunk] = np.take_along_axis(S, part, 1)
    return idx, sims


def make_fit(**config):
    cfg = dict(DEFAULTS)
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg.update(config)

    def fit(train):
        Y = train.label_matrix(sources=("group", "software", "group_desc", "campaign"))
        src_w = {
            "group": cfg["w_group"],
            "campaign": cfg["w_group"],
            "software": cfg["w_software"],
            "group_desc": cfg["w_desc"],
        }
        sw = np.asarray([src_w.get(r["source"], 0.0) for r in train.records], np.float32)
        keep = np.ones(len(Y), bool)
        if cfg["drop_unlabelled"]:
            keep &= Y.sum(1) > 0
        keep &= sw > 0
        X = np.ascontiguousarray(train.X[keep])
        Y = Y[keep]
        sw = sw[keep]
        G = Y.shape[1]

        row_w = sw.copy()
        shift = np.zeros(len(X), np.float32)
        need = max(
            cfg["hub_k"] if cfg["hub"] == "csls" else 0,
            cfg["spread_m"] if cfg["spread_alpha"] > 0 else 0,
        )
        if need:
            nidx, nsims = _train_topk(X, need)
            order = np.argsort(-nsims, axis=1)
            nidx = np.take_along_axis(nidx, order, 1)
            nsims = np.take_along_axis(nsims, order, 1)
            if cfg["hub"] == "csls":
                r = nsims[:, : cfg["hub_k"]].mean(1)
                shift = (cfg["hub_lam"] * (r - np.median(r))).astype(np.float32)
            if cfg["spread_alpha"] > 0:
                q = Y[nidx[:, : cfg["spread_m"]]].sum(1)  # (n, G) neighbour label mass
                tot = q.sum(1, keepdims=True)
                p = np.where(tot > 0, q / np.maximum(tot, 1e-12), 0.0)
                H = -(p * np.log(np.where(p > 0, p, 1.0))).sum(1)
                row_w = row_w * np.exp(-cfg["spread_alpha"] * H).astype(np.float32)
        elif cfg["hub"] not in ("none", "csls"):
            raise ValueError(cfg["hub"])

        mass = (Y * sw[:, None]).sum(0)
        cls = (np.maximum(mass, 1e-6) ** -cfg["class_norm"]).astype(np.float32)
        WY = Y * row_w[:, None]  # weighted label rows
        k, beta = int(cfg["k"]), float(cfg["beta"])
        lp = np.asarray(train.log_prior, np.float32)

        def predict(docs):
            out = []
            for D in docs:
                D = np.asarray(D, np.float32)
                S = D @ X.T - shift[None, :]
                kk = min(k, S.shape[1] - 1)
                cols = np.argpartition(-S, kk, axis=1)[:, :kk]
                s = np.take_along_axis(S, cols, 1)
                w = np.exp(beta * (s - 1.0))
                V = np.einsum("nk,nkg->ng", w, WY[cols]) * cls[None, :]  # per-sentence votes
                tot = V.sum(1, keepdims=True)
                P = np.where(tot > 0, V / np.maximum(tot, 1e-30), 1.0 / G)
                sent_w = P.max(1) ** cfg["sent_gamma"] if cfg["sent_gamma"] else np.ones(len(P))
                if cfg["agg"] == "sum":
                    agg = (V * sent_w[:, None]).sum(0)
                    agg = agg / max(agg.sum(), 1e-30)
                    sc = np.log(agg + 1e-5)
                elif cfg["agg"] == "mean":
                    agg = (P * sent_w[:, None]).sum(0) / max(sent_w.sum(), 1e-30)
                    sc = np.log(agg + 1e-5)
                elif cfg["agg"] == "logsum":
                    sm = cfg["smooth"]
                    sc = (sent_w[:, None] * np.log((1 - sm) * P + sm / G)).sum(0)
                else:
                    raise ValueError(cfg["agg"])
                out.append(sc + cfg["prior"] * lp)
            return np.stack(out)

        return predict

    return fit


_CACHE: list = []  # trials of an earlier run of this same search (--reuse), deterministic model


def _run(cfg, trials, label):
    t = time.time()
    hit = next((r for r in _CACHE if r["config"] == cfg), None)
    if hit is not None:
        m = dict(hit["dev"])
    else:
        m = harness.dev_eval(make_fit(**cfg))
        m.pop("ranks", None)
        m.pop("top1_wilson", None)
    rec = {
        "trial": len(trials) + 1,
        "stage": label,
        "config": dict(cfg),
        "dev": m,
        "seconds": round(time.time() - t, 1),
        "reused": hit is not None,
    }
    trials.append(rec)
    print(
        json.dumps({k: rec[k] for k in ("trial", "stage", "seconds")}),
        {k: m[k] for k in ("top1", "top3", "top5", "mrr", "macro_f1", "selection_score")},
        {k: v for k, v in cfg.items() if v != DEFAULTS[k]},
        flush=True,
    )
    return m["selection_score"]


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    if "--reuse" in sys.argv and (harness.STUDY_DIR / "exp" / "knn.json").is_file():
        _CACHE.extend(json.loads((harness.STUDY_DIR / "exp" / "knn.json").read_text())["trials"])
    TOL = 0.01  # adopt a change only if it beats the incumbent by more than this
    trials = []
    inc = dict(DEFAULTS)
    best = _run(inc, trials, "S0 baseline (= baselines.knn k=10)")

    stages = [
        ("S1 drop unlabelled", [{"drop_unlabelled": True}]),
        ("S2 k", [{"k": 30}, {"k": 100}]),
        ("S3 beta", [{"beta": 5.0}, {"beta": 25.0}]),
        ("S4 aggregation", [{"agg": "mean"}, {"agg": "logsum"}]),
        ("S5 source weights", [{"w_software": 0.5}, {"w_software": 0.2}, {"w_desc": 3.0}]),
        ("S6 hubness", [{"hub": "csls", "hub_lam": 0.5}, {"spread_alpha": 1.0}]),
        ("S7 generic sentences", [{"sent_gamma": 1.0}, {"sent_gamma": 3.0}]),
        ("S8 class norm", [{"class_norm": 0.5}, {"class_norm": 1.0}]),
        ("S9 prior", [{"prior": 0.5}, {"prior": 1.0}]),
    ]
    for label, cands in stages:
        scored = []
        for ch in cands:
            cfg = {**inc, **ch}
            scored.append((_run(cfg, trials, label), ch))
        sc, ch = max(scored, key=lambda x: x[0])
        if sc > best + TOL:
            inc, best = {**inc, **ch}, sc
            print("  adopt", ch, "->", best, flush=True)
        else:
            print("  keep incumbent", best, flush=True)

    # S10: re-check k around the final configuration (k depends on beta/aggregation choices).
    scored = []
    for kk in sorted({max(3, inc["k"] // 3), inc["k"] * 3}):
        cfg = {**inc, "k": kk}
        scored.append((_run(cfg, trials, "S10 k recheck"), {"k": kk}))
    sc, ch = max(scored, key=lambda x: x[0])
    if sc > best + TOL:
        inc, best = {**inc, **ch}, sc
        print("  adopt", ch, "->", best, flush=True)

    # S11/S12: edge checks for values that ended on the edge of their coarse grid
    # (beta=5 was the lowest beta tried, w_software=0.2 the lowest software weight).
    for label, ch in (
        ("S11 beta edge", {"beta": inc["beta"] / 2}),
        ("S12 software edge", {"w_software": 0.0}),
    ):
        cfg = {**inc, **ch}
        if cfg == inc:
            continue
        sc = _run(cfg, trials, label)
        if sc > best + TOL:
            inc, best = cfg, sc
            print("  adopt", ch, "->", best, flush=True)
        else:
            print("  keep incumbent", best, flush=True)

    final = next(t for t in trials if t["config"] == inc)
    (harness.STUDY_DIR / "exp").mkdir(parents=True, exist_ok=True)
    n_new = sum(1 for t in trials if not t["reused"])
    print("dev_eval calls this run:", n_new, "trials total:", len(trials))
    out = {
        "best": inc,
        "dev": final["dev"],
        "trials": trials,
        "seed_scores": [final["dev"]["selection_score"]],
        "notes": f"deterministic model: one run equals every seed; TOL={TOL:.2f} adoption rule",
    }
    (harness.STUDY_DIR / "exp" / "knn.json").write_text(json.dumps(out, indent=1))
    print("BEST", json.dumps(inc), json.dumps(final["dev"]))
