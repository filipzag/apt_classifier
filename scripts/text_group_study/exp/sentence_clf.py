"""Family "sentence_clf": a sentence-level classifier over ATTACK-BERT embeddings, aggregated per report.

Model
- Training rows: every labelled corpus sentence (sources group / software / group_desc / campaign;
  campaign rows exist only in FINAL mode). Soft labels from ``train.label_matrix`` (a sentence's
  groups share its unit of mass). Optional per-source weight for the noisier software rows.
  Rows with no group are dropped, or (``background=True``) form an extra "generic" class.
- Class balancing: each class's loss weight is (effective mass)^(-balance), rescaled so the total
  mass is unchanged (balance=1 -> every group weighs the same in the loss).
- Classifier: multinomial logistic regression or a one-hidden-layer MLP (torch, CPU), on
  standardised embeddings, AdamW, soft-label cross-entropy. ``epochs="auto"``: hold out 10% of the
  labelled rows (seeded), early-stop on their weighted cross-entropy, then refit on all rows for
  the chosen number of epochs. This is part of ``fit`` and uses training data only.
- Report score per group: ``meanlog`` (mean of sentence log-probabilities, i.e. tempered naive
  Bayes), ``meanprob`` (log of the mean sentence probability at temperature T), optionally only the
  ``top_m`` most confident sentences (highest max group probability), plus ``prior * log_prior``.

Tuning: only ``harness.dev_eval`` (23 ATT&CK campaigns), greedy coarse coordinate search below.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the study code folder (harness)
import harness  # noqa: E402

torch.set_num_threads(4)

HERE = harness.STUDY_DIR / "exp"  # trial log and result (data, not code)
KEY = "sentence_clf"

DEFAULTS = dict(
    arch="linear",  # "linear" | "mlp"
    hidden=1024,  # MLP hidden width
    dropout=0.3,  # MLP dropout (hidden layer)
    sources=("group", "software", "group_desc", "campaign"),
    sw_weight=1.0,  # row weight of software procedure examples
    background=False,  # rows with no group -> extra generic class (else dropped)
    balance=1.0,  # class weight = mass^-balance
    lr=2e-3,
    wd=1e-4,
    batch=512,
    epochs="auto",  # "auto" (internal 10% early stopping) or an int
    max_epochs=60,
    patience=6,
    seed=0,
    agg="meanlog",  # "meanlog" | "meanprob"
    T=1.0,  # sentence softmax temperature
    top_m=0,  # 0 = all sentences, else the m most confident sentences
    prior=0.0,  # weight on train.log_prior
)

_MODEL_CACHE: dict = {}


class _MLP(torch.nn.Module):
    def __init__(self, d, h, k, p):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(p), torch.nn.Linear(h, k)
        )

    def forward(self, x):
        return self.net(x)


def _build_model(cfg, d, k):
    if cfg["arch"] == "linear":
        return torch.nn.Linear(d, k)
    if cfg["arch"] == "mlp":
        return _MLP(d, cfg["hidden"], k, cfg["dropout"])
    raise ValueError(cfg["arch"])


def _weighted_ce(logits, target, cw, rw):
    # target (B,K) rows sum to 1; cw (K,) class weights; rw (B,) row weights.
    logp = torch.log_softmax(logits, dim=1)
    wt = target * cw[None, :]
    num = -(rw[:, None] * wt * logp).sum()
    den = (rw[:, None] * wt).sum()
    return num, den


def _train_loop(cfg, X, T, rw, cw, epochs, Xv=None, Tv=None, rwv=None):
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    model = _build_model(cfg, X.shape[1], T.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n = X.shape[0]
    history, best, best_epoch, bad = [], float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, cfg["batch"]):
            idx = perm[s : s + cfg["batch"]]
            num, den = _weighted_ce(model(X[idx]), T[idx], cw, rw[idx])
            loss = num / den.clamp_min(1e-9)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                num, den = _weighted_ce(model(Xv), Tv, cw, rwv)
                v = float(num / den)
            history.append(round(v, 4))
            if v < best - 1e-4:
                best, best_epoch, bad = v, epoch, 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    model.eval()
    return model, best_epoch, history


def _prepare(cfg, train):
    Y = train.label_matrix(sources=tuple(cfg["sources"]))
    src = np.asarray([r["source"] for r in train.records])
    has = Y.sum(1) > 0
    rw = np.ones(len(Y), np.float32)
    rw[src == "software"] = cfg["sw_weight"]
    k = Y.shape[1]
    if cfg["background"]:
        bg = (~has) & np.isin(src, list(cfg["sources"]))
        T = np.zeros((len(Y), k + 1), np.float32)
        T[:, :k] = Y
        T[bg, k] = 1.0
        keep = has | bg
    else:
        T = Y
        keep = has
    keep &= rw > 0
    X, T, rw = train.X[keep], T[keep], rw[keep]
    mass = (rw[:, None] * T).sum(0)
    cw = np.where(mass > 0, np.power(np.maximum(mass, 1e-6), -cfg["balance"]), 0.0)
    cw *= mass.sum() / max((cw * mass).sum(), 1e-9)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, T, rw, cw.astype(np.float32), mu, sd


def _fingerprint(train):
    h = hashlib.sha1()
    h.update(str(len(train.records)).encode())
    h.update(np.ascontiguousarray(train.X[:: max(1, len(train.X) // 512)]).tobytes())
    return h.hexdigest()


_TRAIN_KEYS = (
    "arch",
    "hidden",
    "dropout",
    "sources",
    "sw_weight",
    "background",
    "balance",
    "lr",
    "wd",
    "batch",
    "epochs",
    "max_epochs",
    "patience",
    "seed",
)


def _fit_model(cfg, train):
    key = (
        _fingerprint(train),
        json.dumps({k: cfg[k] for k in _TRAIN_KEYS}, sort_keys=True, default=list),
    )
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    t0 = time.time()
    Xs, T, rw, cw, mu, sd = _prepare(cfg, train)
    X_t, T_t, rw_t, cw_t = map(torch.from_numpy, (Xs.astype(np.float32), T, rw, cw))
    info = {}
    if cfg["epochs"] == "auto":
        rng = np.random.default_rng(cfg["seed"])
        perm = rng.permutation(len(Xs))
        nv = len(Xs) // 10
        va, tr = perm[:nv], perm[nv:]
        _, best_epoch, hist = _train_loop(
            cfg, X_t[tr], T_t[tr], rw_t[tr], cw_t, cfg["max_epochs"], X_t[va], T_t[va], rw_t[va]
        )
        epochs = max(1, best_epoch)
        info.update(auto_epochs=epochs, val_history=hist)
    else:
        epochs = int(cfg["epochs"])
    model, _, _ = _train_loop(cfg, X_t, T_t, rw_t, cw_t, epochs)
    info["fit_seconds"] = round(time.time() - t0, 1)
    out = (model, mu, sd, info)
    _MODEL_CACHE.clear()  # keep at most one model in memory
    _MODEL_CACHE[key] = out
    return out


def make_fit(**config):
    cfg = dict(DEFAULTS)
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg.update(config)
    cfg["sources"] = tuple(cfg["sources"])

    def fit(train):
        model, mu, sd, info = _fit_model(cfg, train)
        fit.info = info
        k = len(train.groups)

        def predict(docs):
            lens = [len(d) for d in docs]
            allX = np.concatenate([np.asarray(d, np.float32) for d in docs], 0)
            with torch.no_grad():
                z = model(torch.from_numpy(((allX - mu) / sd).astype(np.float32))) / cfg["T"]
                logp = torch.log_softmax(z, dim=1)[:, :k].numpy().astype(np.float64)
            out, s = [], 0
            for n in lens:
                lp = logp[s : s + n]
                s += n
                if cfg["top_m"] and n > cfg["top_m"]:
                    conf = lp.max(1)
                    lp = lp[np.argsort(-conf, kind="stable")[: cfg["top_m"]]]
                if cfg["agg"] == "meanlog":
                    score = lp.mean(0)
                elif cfg["agg"] == "meanprob":
                    m = lp.max()
                    score = np.log(np.exp(lp - m).mean(0) + 1e-300) + m
                else:
                    raise ValueError(cfg["agg"])
                out.append(score + cfg["prior"] * train.log_prior)
            return np.stack(out)

        return predict

    return fit


# Chosen configuration (filled in from the search result below; see sentence_clf.json).
BEST = dict(DEFAULTS)
BEST.update(
    arch="linear",
    agg="meanprob",
    T=2.0,
    top_m=0,
    balance=0.0,
    sw_weight=1.0,
    background=True,
    prior=0.5,
    epochs=40,
    seed=0,
)
BEST["sources"] = list(BEST["sources"])


# ----------------------------------------------------------------------------------------------
# Search (dev only): greedy coarse coordinate search; a candidate replaces the incumbent only when
# it beats it by more than MARGIN selection_score (ties keep the simpler / earlier config).
MARGIN = 0.01
LOG = HERE / f"{KEY}_trials.jsonl"


def _cfg_key(cfg):
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True, default=list)


def _load_log():
    done = {}
    if LOG.is_file():
        for line in LOG.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    return done


def run_trial(cfg, stage, done):
    full = dict(DEFAULTS)
    full.update(cfg)
    full["sources"] = list(full["sources"])
    key = _cfg_key(full)
    if key in done:
        row = done[key]
        print(f"[cached] {stage}: {row['dev']['selection_score']}")
        return row
    fit = make_fit(**full)
    t0 = time.time()
    m = harness.dev_eval(fit)
    ranks = m.pop("ranks")
    row = {
        "key": key,
        "stage": stage,
        "config": full,
        "dev": m,
        "ranks": ranks,
        "info": {k: v for k, v in getattr(fit, "info", {}).items()},
        "eval_seconds": round(time.time() - t0, 1),
    }
    with LOG.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    done[key] = row
    print(
        f"{stage}: sel={m['selection_score']} top1={m['top1']} top3={m['top3']} top5={m['top5']} "
        f"mrr={m['mrr']} f1={m['macro_f1']} epochs={row['info'].get('auto_epochs')} "
        f"fit={row['info'].get('fit_seconds')}s",
        flush=True,
    )
    return row


def search():
    done = _load_log()
    trials = []
    inc = {}
    inc_row = None

    def consider(stage, candidates):
        nonlocal inc, inc_row
        for cand in candidates:
            cfg = dict(inc)
            cfg.update(cand)
            row = run_trial(cfg, stage, done)
            trials.append(row)
            if (
                inc_row is None
                or row["dev"]["selection_score"] > inc_row["dev"]["selection_score"] + MARGIN
            ):
                inc, inc_row = cfg, row
        print(
            f"  -> incumbent after {stage}: {json.dumps({k: v for k, v in inc.items()})} "
            f"sel={inc_row['dev']['selection_score']}",
            flush=True,
        )

    # 1. classifier x aggregation (simplest first)
    consider(
        "1_arch_agg",
        [
            dict(arch="linear", agg="meanlog"),
            dict(arch="linear", agg="meanprob"),
            dict(arch="mlp", agg="meanlog"),
            dict(arch="mlp", agg="meanprob"),
        ],
    )
    # 2. aggregation temperature (meanprob only; meanlog's ranking does not depend on T)
    consider("2_temperature", [dict(agg="meanprob", T=0.5), dict(agg="meanprob", T=2.0)])
    # 3. only the m most confident sentences
    consider("3_top_m", [dict(top_m=5), dict(top_m=15)])
    # 4. class balancing
    consider("4_balance", [dict(balance=0.5), dict(balance=0.0)])
    # 5. software rows: down-weighted; generic background class
    consider("5_software_bg", [dict(sw_weight=0.3), dict(background=True)])
    # 6. knowledge-base prior
    consider("6_prior", [dict(prior=0.5), dict(prior=1.0)])
    if inc.get("prior") == 1.0:
        consider("6b_prior_edge", [dict(prior=2.0)])
    # 7. capacity / regularisation
    if inc.get("arch", DEFAULTS["arch"]) == "mlp":
        consider("7_capacity", [dict(hidden=2048, dropout=0.5), dict(epochs=40)])
    else:
        consider("7_capacity", [dict(wd=1e-2), dict(epochs=40)])
    # 8. re-check two early choices under the final settings: the temperature grid edge (T=2 won
    #    stage 2 at its upper edge) and the MLP, which stage 1 rejected under balance=1 / no prior.
    consider("8_recheck", [dict(agg="meanprob", T=4.0), dict(arch="mlp", epochs="auto")])

    best = dict(DEFAULTS)
    best.update(inc)
    best["sources"] = list(best["sources"])
    # seeds
    seed_rows = [inc_row]
    for s in (1, 2):
        cfg = dict(inc)
        cfg["seed"] = s
        seed_rows.append(run_trial(cfg, f"seed_{s}", done))
    seed_scores = [r["dev"]["selection_score"] for r in seed_rows]
    return best, inc_row, trials, seed_rows, seed_scores


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Smoke test on TRAINING data only (no dev/test): fit time, epochs, a fake 3-sentence doc.
        tr = harness._train("dev")
        for arch in sys.argv[2:] or ["linear"]:
            fit = make_fit(arch=arch)
            pred = fit(tr)
            print(arch, fit.info)
            fake = harness.doc([r["text"] for r in tr.records[:3]])
            print(np.asarray(pred([fake])).shape)
        sys.exit(0)
    HERE.mkdir(parents=True, exist_ok=True)
    best, inc_row, trials, seed_rows, seed_scores = search()
    dev = dict(inc_row["dev"])
    out = {
        "best": best,
        "dev": dev,
        "trials": [
            {
                "stage": r["stage"],
                "config": r["config"],
                "dev": r["dev"],
                "info": {k: v for k, v in r["info"].items() if k != "val_history"},
            }
            for r in trials
        ],
        "seed_scores": seed_scores,
        "seed_rows": [{"stage": r["stage"], "dev": r["dev"]} for r in seed_rows],
        "seed_mean": round(float(np.mean(seed_scores)), 4),
        "seed_std": round(float(np.std(seed_scores)), 4),
        "dev_eval_calls": len({r["key"] for r in trials} | {r["key"] for r in seed_rows}),
    }
    (HERE / f"{KEY}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"best": best, "dev": dev, "seed_scores": seed_scores}, indent=1))
