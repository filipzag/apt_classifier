"""Experiment "prototypes": per-group prototype similarity models (no gradient training).

make_fit(**config) -> fit(train) -> predict(docs) -> (n_docs, n_groups) scores.

Prototypes, per source (all L2-normalised):
- proc: per group, k-means sub-prototypes of its own procedure examples (source 'group';
  in FINAL mode also 'campaign' rows, which are procedure examples of a group's campaigns).
  k = min(k_proc, ceil(n / spp)); a group with n <= k sentences keeps them as exemplars.
- soft: per software, k-means sub-prototypes (k_soft) of its procedure examples; a group's
  software prototypes are those of every software it uses (train.group_software).
- desc: every group-description sentence of the group (exemplars, at most 8 per group).

Score of report sentence s for group g and source src: a_src(s, g) = max over g's prototypes
of cos(s, p). Hubness correction (norm), computed against a background of every training
sentence, with the sentences labelled with g masked out (they built g's prototypes):
- 'z': (a - mu_g) / sd_g (per-group, per-source standardisation);
- 'csls': (a - r_g) / sd_pool, r_g = mean of the top csls_k background values for g;
- 'none': (a - mu_pool) / sd_pool (same scale, no per-group correction).
Sources are combined per sentence by 'max' (best-matching source) or 'wsum' (weighted mean,
missing source = -1). Sentences are aggregated per report by:
- 'softmax': q(g|s) = softmax_g(Z(s, g) / tau), score = log(mean_s q(g|s) + 1e-6);
- 'topm': mean of the m largest Z(s, g) over the report's sentences (m fixed integer).
Final score = score + beta * train.log_prior. Stochastic only through k-means (seed).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the study code folder (harness)
import harness  # noqa: E402

try:
    import torch

    torch.set_num_threads(4)
except Exception:  # torch is not needed here
    pass

HERE = harness.STUDY_DIR / "exp"  # trial log and result (data, not code)
KEY = "prototypes"

DEFAULTS = dict(
    k_proc=3,
    spp=3,
    k_soft=1,
    sources=("proc", "soft"),
    comb="max",
    norm="z",
    csls_k=10,
    agg="softmax",
    tau=1.0,
    m=5,
    beta=0.5,
    w_soft=1.0,
    w_desc=1.0,
    seed=0,
)

# Chosen by the staged coordinate search in __main__ (every trial is in prototypes.json).
BEST = dict(DEFAULTS, k_proc=5, norm="none", tau=0.7, beta=0.0)


def _unit(A):
    return (A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def _subprotos(X, k, seed):
    n = len(X)
    if n == 0:
        return np.zeros((0, X.shape[1]), np.float32)
    if k <= 1 or n == 1:
        return _unit(X.mean(0, keepdims=True))
    if n <= k:
        return _unit(X)
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
    return _unit(km.cluster_centers_)


class _Source:
    """Prototype matrix P and, per group, the columns of P that belong to it."""

    def __init__(self, P, cols):
        self.P = P
        self.cols = cols  # list (len n_groups) of int arrays (may be empty)
        self.has = np.array([len(c) > 0 for c in cols])

    def pool(self, Q):
        """(n, n_groups) max similarity of each row of Q to each group's prototypes (nan if none)."""
        S = Q @ self.P.T
        A = np.full((len(Q), len(self.cols)), np.nan, np.float32)
        for g, c in enumerate(self.cols):
            if len(c):
                A[:, g] = S[:, c].max(1)
        return A


def _build_sources(train, cfg):
    gi = {g: i for i, g in enumerate(train.groups)}
    G = len(train.groups)
    seed = cfg["seed"]
    out = {}
    if "proc" in cfg["sources"]:
        rows = [[] for _ in range(G)]
        for i, r in enumerate(train.records):
            if r["source"] in ("group", "campaign"):
                for g in r["groups"]:
                    rows[gi[g]].append(i)
        Ps, cols, off = [], [], 0
        for g in range(G):
            n = len(rows[g])
            if n == 0:
                cols.append(np.zeros(0, int))
                continue
            k = min(cfg["k_proc"], math.ceil(n / cfg["spp"]))
            P = _subprotos(train.X[rows[g]], k, seed + g)
            Ps.append(P)
            cols.append(np.arange(off, off + len(P)))
            off += len(P)
        out["proc"] = _Source(np.concatenate(Ps), cols)
    if "soft" in cfg["sources"]:
        by_sw = {}
        for i, r in enumerate(train.records):
            if r["source"] == "software" and r["groups"]:
                by_sw.setdefault(r["owner"], []).append(i)
        Ps, sw_cols, off = [], {}, 0
        for j, (sw, idx) in enumerate(sorted(by_sw.items())):
            P = _subprotos(
                train.X[idx], min(cfg["k_soft"], math.ceil(len(idx) / cfg["spp"])), seed + 1000 + j
            )
            Ps.append(P)
            sw_cols[sw] = np.arange(off, off + len(P))
            off += len(P)
        cols = []
        for g in train.groups:
            cs = [sw_cols[s] for s in sorted(train.group_software.get(g, ())) if s in sw_cols]
            cols.append(np.concatenate(cs) if cs else np.zeros(0, int))
        out["soft"] = _Source(np.concatenate(Ps), cols)
    if "desc" in cfg["sources"]:
        rows = [[] for _ in range(G)]
        for i, r in enumerate(train.records):
            if r["source"] == "group_desc":
                for g in r["groups"]:
                    rows[gi[g]].append(i)
        Ps, cols, off = [], [], 0
        for g in range(G):
            if not rows[g]:
                cols.append(np.zeros(0, int))
                continue
            P = _unit(train.X[rows[g]])
            Ps.append(P)
            cols.append(np.arange(off, off + len(P)))
            off += len(P)
        out["desc"] = _Source(np.concatenate(Ps), cols)
    return out


def _background_stats(train, src, cfg, chunk=2048):
    """Per-group mean / sd / top-k mean of pooled similarity over every training sentence
    not labelled with that group (any source)."""
    Yall = train.label_matrix(sources=("group", "software", "group_desc", "campaign")) > 0
    G = len(train.groups)
    s1 = np.zeros(G, np.float64)
    s2 = np.zeros(G, np.float64)
    cnt = np.zeros(G, np.float64)
    kk = cfg["csls_k"]
    top = np.full((kk, G), -np.inf, np.float32)
    for s in range(0, len(train.X), chunk):
        A = src.pool(train.X[s : s + chunk])
        A[Yall[s : s + chunk]] = np.nan
        valid = ~np.isnan(A)
        Af = np.where(valid, A, 0.0)
        s1 += Af.sum(0)
        s2 += (Af.astype(np.float64) ** 2).sum(0)
        cnt += valid.sum(0)
        At = np.where(valid, A, -np.inf)
        both = np.concatenate([top, At], 0)
        top = -np.sort(-both, axis=0)[:kk]
    mu = s1 / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mu**2, 1e-8))
    fin = np.isfinite(top)
    r = np.where(fin, top, 0.0).sum(0) / np.maximum(fin.sum(0), 1)
    has = src.has
    mu_pool = float(np.mean(mu[has]))
    sd_pool = float(np.sqrt(np.mean((sd[has] ** 2) + (mu[has] - mu_pool) ** 2)))
    return dict(
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        r=np.nan_to_num(r).astype(np.float32),
        mu_pool=mu_pool,
        sd_pool=sd_pool,
    )


def make_fit(**config):
    cfg = dict(DEFAULTS)
    cfg.update(config)
    cfg["sources"] = tuple(cfg["sources"])

    def fit(train):
        sources = _build_sources(train, cfg)
        stats = {name: _background_stats(train, src, cfg) for name, src in sources.items()}
        weight = {"proc": 1.0, "soft": cfg["w_soft"], "desc": cfg["w_desc"]}
        G = len(train.groups)

        def normed(name, A):
            st = stats[name]
            if cfg["norm"] == "z":
                return (A - st["mu"]) / st["sd"]
            if cfg["norm"] == "csls":
                return (A - st["r"]) / st["sd_pool"]
            if cfg["norm"] == "none":
                return (A - st["mu_pool"]) / st["sd_pool"]
            raise ValueError(cfg["norm"])

        def sentence_scores(D):
            Zs = [
                normed(name, src.pool(np.asarray(D, np.float32))) for name, src in sources.items()
            ]
            names = list(sources)
            if cfg["comb"] == "max":
                stack = np.stack([np.where(np.isnan(Z), -np.inf, Z) for Z in Zs])
                Z = stack.max(0)
                Z = np.where(np.isfinite(Z), Z, -3.0)
            elif cfg["comb"] == "wsum":
                num = np.zeros((len(D), G), np.float32)
                den = 0.0
                for name, Z in zip(names, Zs, strict=False):
                    num += weight[name] * np.where(np.isnan(Z), -1.0, Z)
                    den += weight[name]
                Z = num / den
            else:
                raise ValueError(cfg["comb"])
            return Z

        def predict(docs):
            out = []
            for D in docs:
                Z = sentence_scores(D)
                if cfg["agg"] == "softmax":
                    L = Z / cfg["tau"]
                    L = L - L.max(1, keepdims=True)
                    q = np.exp(L)
                    q /= q.sum(1, keepdims=True)
                    sc = np.log(q.mean(0) + 1e-6)
                elif cfg["agg"] == "topm":
                    m = max(1, min(int(cfg["m"]), len(Z)))
                    sc = -np.sort(-Z, axis=0)[:m].mean(0)
                else:
                    raise ValueError(cfg["agg"])
                out.append(sc + cfg["beta"] * train.log_prior)
            return np.stack(out)

        return predict

    return fit


# ----------------------------------------------------------------------------- search

TRIALS_LOG = HERE / f"{KEY}_trials.jsonl"


def _key(cfg):
    return json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())})


def _load_log():
    done = {}
    if TRIALS_LOG.is_file():
        for line in TRIALS_LOG.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[_key(dict(DEFAULTS, **row["config"]))] = row
    return done


def run_trial(cfg, note, done):
    full = dict(DEFAULTS, **cfg)
    k = _key(full)
    if k in done:
        return done[k]
    t = time.time()
    m = harness.dev_eval(make_fit(**full))
    secs = round(time.time() - t, 1)
    m.pop("ranks")
    row = {
        "config": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in full.items()},
        "note": note,
        "dev": m,
        "seconds": secs,
    }
    with TRIALS_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    done[k] = row
    print(
        json.dumps(
            {
                "note": note,
                "sel": m["selection_score"],
                "top1": m["top1"],
                "top3": m["top3"],
                "top5": m["top5"],
                "mrr": m["mrr"],
                "f1": m["macro_f1"],
                "s": secs,
            }
        ),
        flush=True,
    )
    return row


def _stages(best):
    """Coordinate search: each stage tries alternatives around the current best config.
    Returns (note, override, complexity) with complexity -1 simpler, 0 same, +1 richer."""
    if best["agg"] == "softmax":
        refine = [
            ("refine tau x0.7", dict(tau=round(best["tau"] * 0.7, 3)), 0),
            ("refine tau x1.4", dict(tau=round(best["tau"] * 1.4, 3)), 0),
        ]
    else:
        refine = [
            ("refine m-2", dict(m=max(1, best["m"] - 2)), 0),
            ("refine m+3", dict(m=best["m"] + 3), 0),
        ]
    return {
        1: [("hubness none", dict(norm="none"), -1), ("hubness csls", dict(norm="csls"), 0)],
        2: [
            ("softmax tau 0.5", dict(agg="softmax", tau=0.5), 0),
            ("softmax tau 2", dict(agg="softmax", tau=2.0), 0),
            ("topm m=3", dict(agg="topm", m=3), 0),
            ("topm m=8", dict(agg="topm", m=8), 0),
        ],
        3: [("k_proc 1 (centroid)", dict(k_proc=1), -1), ("k_proc 5", dict(k_proc=5), 1)],
        4: [
            ("sources proc only", dict(sources=("proc",)), -1),
            ("sources +desc", dict(sources=("proc", "soft", "desc")), 1),
        ],
        5: [
            ("prior 0", dict(beta=0.0), -1),
            ("prior 1", dict(beta=1.0), 0),
            ("prior 2", dict(beta=2.0), 0),
        ],
        6: [("combine wsum", dict(comb="wsum"), 0)],
        7: [("k_soft 2", dict(k_soft=2), 1)],
        8: refine,
        # Added after stages 1-8 had run (post-hoc probes, decided from their results): the
        # per-group hubness corrections lost in stage 1 while the prior was 0.5 and stage 5 then
        # found that the uncorrected scores need no prior (big groups already win on raw
        # similarity), so the corrections are retried with a full prior; desc is retried at the
        # sharper temperature.
        9: [
            ("hubness z + prior 1", dict(norm="z", beta=1.0), 1),
            ("hubness csls + prior 1", dict(norm="csls", beta=1.0), 1),
            ("sources +desc (at best)", dict(sources=("proc", "soft", "desc")), 1),
        ],
    }


EPS = {-1: 0.0, 0: 0.005, 1: 0.01}


def search():
    done = _load_log()
    best = dict(DEFAULTS)
    best_row = run_trial(best, "base", done)
    history = []
    for stage in range(1, 10):
        alts = _stages(best)[stage]
        rows = []
        for note, override, cx in alts:
            if all(
                best.get(k) == (tuple(v) if isinstance(v, list) else v) for k, v in override.items()
            ):
                continue
            row = run_trial(dict(best, **override), f"stage {stage}: {note}", done)
            rows.append((row, override, cx))
        # adopt the alternative with the largest margin over its complexity threshold:
        # simpler: >= current; same complexity: > current + 0.005; richer: > current + 0.01
        base_sel = best_row["dev"]["selection_score"]
        cand = []
        for r, o, cx in rows:
            margin = r["dev"]["selection_score"] - base_sel - EPS[cx]
            if margin > 0 or (cx < 0 and margin >= 0):
                cand.append((margin, -cx, r, o))
        if cand:
            cand.sort(key=lambda c: (-c[0], -c[1]))
            _, _, row, override = cand[0]
            best = dict(
                best, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in override.items()}
            )
            best_row = row
        history.append(
            {
                "stage": stage,
                "best_after": best_row["note"],
                "selection_score": best_row["dev"]["selection_score"],
            }
        )
        print(
            "after stage",
            stage,
            "->",
            best_row["note"],
            best_row["dev"]["selection_score"],
            flush=True,
        )
    return best, best_row, done, history


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    HERE.mkdir(parents=True, exist_ok=True)
    best, best_row, done, history = search()
    print("BEST", best)
    seed_rows = [best_row] + [
        run_trial(dict(best, seed=s), f"seed check {s}", done) for s in (1, 2)
    ]
    seed_scores = [r["dev"]["selection_score"] for r in seed_rows]
    trials = [json.loads(line) for line in TRIALS_LOG.read_text().splitlines() if line.strip()]
    result = {
        "best": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
        "dev": best_row["dev"],
        "trials": trials,
        "seed_scores": seed_scores,
        "seed_mean": round(float(np.mean(seed_scores)), 4),
        "seed_std": round(float(np.std(seed_scores)), 4),
        "history": history,
        "n_dev_eval_calls": len(trials),
    }
    (HERE / f"{KEY}.json").write_text(json.dumps(result, indent=1))
    print("seed_scores", seed_scores, result["seed_mean"], result["seed_std"])
