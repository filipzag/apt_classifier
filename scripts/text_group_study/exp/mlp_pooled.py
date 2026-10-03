"""mlp_pooled: PyTorch MLP over POOLED report embeddings, trained on synthetic reports.

Synthetic-case design follows apt_classifier/group_mlp.py, translated from techniques to sentences:

- a synthetic report about group g has ``size_min``-``size_max`` true sentences (size independent of
  how many sentences ATT&CK documents for g; capped by what the pool holds, never duplicated);
- the true pool is g's own sentences (group procedure examples, group description sentences and, in
  FINAL mode, attributed campaign sentences) and, with probability ``software_rate``, the procedure
  sentences of 1-``max_software`` of g's documented software (at most ``sw_cap`` sentences each);
- plus a share U(0, ``distract_max``) of distractor sentences (relative to the true count) drawn
  uniformly from every training sentence not labelled with g (other groups, other software,
  group-less software): false positives in a real report;
- equal cases per group (``cases_per_group``), groups whose pool holds fewer than
  ``min_group_texts`` sentences are not trained on; cases are redrawn every ``resample_every``
  epochs.

Features: mean pooling of the sentence embeddings, optionally concatenated with max pooling,
standardised with the first draw's statistics. Output: softmax over train.groups, label smoothing.
At inference: logits + prior_power * train.log_prior (knowledge-base prominence prior).
CPU only. Search (all logged in mlp_pooled_trials.jsonl / mlp_pooled.json, 23 dev_eval calls):
stage 1 one-factor-at-a-time on dev_eval (seed std ~0.07 on dev, as large as every difference);
stage 2 network settings ranked on an internal held-out-sentence validation built from the DEV-mode
training data only (mlp_pooled_internal.py; no dev or test data); stage 3 dev_eval for the prior
power, three seeds, and a 1- vs 3-member comparison (tie -> the simpler single network).
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # study code (harness)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import harness  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

HERE = str(harness.STUDY_DIR / "exp")  # trial logs and result (data, not code)
OWN_SOURCES = ("group", "group_desc", "campaign")

DEFAULTS = {
    # synthetic reports
    "cases_per_group": 200,
    "size_min": 3,
    "size_max": 30,
    "software_rate": 0.7,
    "max_software": 3,
    "sw_cap": 10,
    "distract_max": 0.5,
    "min_group_texts": 2,
    "noise": 0.0,  # per-dimension Gaussian jitter added to each sentence embedding (train only)
    # features
    "pooling": "mean",  # "mean" or "meanmax"
    # network
    "hidden": [512],
    "dropout": 0.3,
    "input_dropout": 0.1,
    "members": 1,
    # optimisation
    "epochs": 20,
    "resample_every": 4,
    "batch_size": 512,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    # inference
    "prior_power": 1.0,
    "seed": 7,
}

# Chosen by the search (see __main__ and mlp_pooled.json); make_fit(**BEST) works standalone.
BEST = {
    "cases_per_group": 400,
    "size_min": 3,
    "size_max": 30,
    "software_rate": 0.7,
    "max_software": 5,
    "sw_cap": 20,
    "distract_max": 0.5,
    "min_group_texts": 2,
    "noise": 0.0,
    "pooling": "mean",
    "hidden": [512],
    "dropout": 0.5,
    "input_dropout": 0.1,
    "members": 1,
    "epochs": 30,
    "resample_every": 5,
    "batch_size": 512,
    "lr": 0.002,
    "weight_decay": 0.0001,
    "label_smoothing": 0.1,
    "prior_power": 1.0,
    "seed": 7,
}

_NET_CACHE: dict = {}


# ---------------------------------------------------------------------------- synthetic reports
class Pools:
    """Per-group sentence pools built from a Train (works for DEV and FINAL mode alike)."""

    def __init__(self, train: harness.Train, p: dict):
        self.X = train.X
        self.groups = train.groups
        gi = {g: i for i, g in enumerate(train.groups)}
        n = len(train.records)
        self.labelled = np.zeros((n, len(train.groups)), dtype=bool)  # record mentions group
        own = {g: [] for g in train.groups}
        sw = {}
        for row, r in enumerate(train.records):
            for g in r["groups"]:
                if g in gi:
                    self.labelled[row, gi[g]] = True
            if r["source"] in OWN_SOURCES:
                for g in r["groups"]:
                    if g in own:
                        own[g].append(row)
            elif r["source"] == "software":
                sw.setdefault(r["owner"], []).append(row)
        self.own = {g: np.asarray(v, dtype=np.int64) for g, v in own.items()}
        self.sw = {s: np.asarray(v, dtype=np.int64) for s, v in sw.items()}
        self.tools = {
            g: [s for s in sorted(train.group_software.get(g, ())) if s in self.sw]
            for g in train.groups
        }
        cap = int(p["sw_cap"])
        self.eligible = [
            g
            for g in train.groups
            if len(self.own[g]) + sum(min(cap, len(self.sw[s])) for s in self.tools[g])
            >= int(p["min_group_texts"])
        ]
        self.gi = gi

    def draw(self, p: dict, seed: int):
        """Synthetic reports: (features (m, d) float32, labels (m,) int64)."""
        rng = np.random.default_rng(seed)
        X = self.X
        n_rec = X.shape[0]
        per = int(p["cases_per_group"])
        meanmax = p["pooling"] == "meanmax"
        dim = X.shape[1] * (2 if meanmax else 1)
        feats = np.zeros((len(self.eligible) * per, dim), dtype=np.float32)
        labels = np.zeros(len(self.eligible) * per, dtype=np.int64)
        lo, hi = int(p["size_min"]), int(p["size_max"])
        cap = int(p["sw_cap"])
        noise = float(p.get("noise", 0.0))
        row = 0
        for g in self.eligible:
            col = self.gi[g]
            own = self.own[g]
            tools = self.tools[g]
            for _ in range(per):
                size = int(rng.integers(lo, hi + 1))
                parts = [own]
                if tools and rng.uniform() < p["software_rate"]:
                    k = int(rng.integers(1, min(int(p["max_software"]), len(tools)) + 1))
                    for t in rng.choice(len(tools), size=k, replace=False):
                        rows = self.sw[tools[t]]
                        if rows.size > cap:
                            rows = rng.choice(rows, size=cap, replace=False)
                        parts.append(rows)
                pool = np.unique(np.concatenate(parts)) if len(parts) > 1 else own
                if pool.size == 0:  # group with only software text but none drawn this time
                    tools_rows = np.concatenate([self.sw[s][:cap] for s in tools])
                    pool = np.unique(tools_rows)
                take = rng.choice(pool, size=min(size, pool.size), replace=False)
                n_d = int(round(take.size * rng.uniform(0.0, float(p["distract_max"]))))
                if n_d:
                    cand = rng.integers(0, n_rec, size=n_d * 3 + 4)
                    cand = cand[~self.labelled[cand, col]][:n_d]
                    take = np.concatenate([take, cand])
                E = X[take]
                if noise > 0:  # paraphrase-like jitter of every sentence before pooling
                    E = E + noise * rng.standard_normal(E.shape, dtype=np.float32)
                if meanmax:
                    feats[row, : X.shape[1]] = E.mean(0)
                    feats[row, X.shape[1] :] = E.max(0)
                else:
                    feats[row] = E.mean(0)
                labels[row] = col
                row += 1
        return feats, labels


def pool_docs(docs, pooling: str, d: int) -> np.ndarray:
    out = []
    for D in docs:
        D = np.asarray(D, dtype=np.float32)
        if D.ndim != 2 or D.shape[0] == 0:
            m = np.zeros(d, np.float32)
            v = [m, m] if pooling == "meanmax" else [m]
        else:
            v = [D.mean(0), D.max(0)] if pooling == "meanmax" else [D.mean(0)]
        out.append(np.concatenate(v))
    return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------------------- network
def build_net(n_in: int, n_out: int, hidden, dropout: float, input_dropout: float):
    nn = torch.nn
    layers = [nn.Dropout(input_dropout)]
    width = n_in
    for h in hidden:
        layers += [nn.Linear(width, int(h)), nn.LayerNorm(int(h)), nn.GELU(), nn.Dropout(dropout)]
        width = int(h)
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


def _train_key(train: harness.Train, p: dict) -> str:
    net_p = {k: v for k, v in p.items() if k != "prior_power"}
    fp = (len(train.records), float(train.X[:, :8].sum()), tuple(train.groups[:3]))
    return json.dumps([net_p, repr(fp)], sort_keys=True)


def train_networks(train: harness.Train, p: dict):
    """Fit ``members`` networks (cached per config without prior_power and per Train)."""
    key = _train_key(train, p)
    if key in _NET_CACHE:
        return _NET_CACHE[key]
    t0 = time.time()
    pools = Pools(train, p)
    n_out = len(train.groups)
    members = []
    stats = None
    for m in range(max(1, int(p["members"]))):
        seed = int(p["seed"]) + 1000 * m
        torch.manual_seed(seed)
        feats, labels = pools.draw(p, seed)
        if stats is None:
            mu = feats.mean(0)
            sd = feats.std(0) + 1e-6
            stats = (mu.astype(np.float32), sd.astype(np.float32))
        net = build_net(
            feats.shape[1], n_out, p["hidden"], float(p["dropout"]), float(p["input_dropout"])
        )
        opt = torch.optim.AdamW(
            net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        epochs = int(p["epochs"])
        bs = int(p["batch_size"])
        steps = epochs * int(np.ceil(len(labels) / bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=float(p["lr"]), total_steps=steps, pct_start=0.1, anneal_strategy="cos"
        )
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(p["label_smoothing"]))
        gen = torch.Generator().manual_seed(seed)
        every = int(p["resample_every"]) or epochs
        feats -= stats[0]  # standardise in place (memory: one copy of the cases at a time)
        feats /= stats[1]
        Xt, yt = torch.from_numpy(feats), torch.from_numpy(labels)
        del feats, labels
        last = None
        for epoch in range(epochs):
            if epoch and epoch % every == 0:
                del Xt, yt
                feats, labels = pools.draw(p, seed + epoch)
                feats -= stats[0]
                feats /= stats[1]
                Xt, yt = torch.from_numpy(feats), torch.from_numpy(labels)
                del feats, labels
            net.train()
            perm = torch.randperm(len(yt), generator=gen)
            total, correct, count = 0.0, 0, 0
            for s in range(0, len(perm), bs):
                idx = perm[s : s + bs]
                if len(idx) < 2:
                    continue
                out = net(Xt[idx])
                loss = loss_fn(out, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
                correct += int((out.argmax(1) == yt[idx]).sum())
                count += len(idx)
            last = (round(total / max(count, 1), 4), round(correct / max(count, 1), 4))
        net.eval()
        members.append(net)
        del Xt, yt
    info = {
        "fit_seconds": round(time.time() - t0, 1),
        "eligible": len(pools.eligible),
        "last_epoch_loss_acc": last,
    }
    result = (members, stats, info)
    _NET_CACHE.clear()  # keep memory bounded: only the latest network stays cached
    _NET_CACHE[key] = result
    return result


def make_fit(**config):
    p = {**DEFAULTS, **config}
    if p["pooling"] not in ("mean", "meanmax"):
        raise ValueError(p["pooling"])

    def fit(train: harness.Train):
        members, stats, info = train_networks(train, p)
        fit.info = info
        d = train.X.shape[1]
        prior = np.asarray(train.log_prior, dtype=np.float32)

        def predict(docs):
            F = pool_docs(docs, p["pooling"], d)
            Ft = torch.from_numpy((F - stats[0]) / stats[1])
            with torch.no_grad():
                logits = torch.stack([net(Ft) for net in members]).mean(0).numpy()
            return (logits + float(p["prior_power"]) * prior[None, :]).astype(np.float64)

        return predict

    fit.info = {}
    return fit


# ---------------------------------------------------------------------------- search
# The search is resumable: every dev_eval result is appended to mlp_pooled_trials.jsonl and a rerun
# reuses a logged (name, config) instead of calling dev_eval again, so the dev set is scored once
# per trial in total. Stage 2 (internal validation) uses training data only, see
# mlp_pooled_internal.py, and its rows are reused from mlp_pooled_internal.jsonl the same way.
TRIALS: list = []
TRIAL_LOG = os.path.join(HERE, "mlp_pooled_trials.jsonl")
INTERNAL_LOG = os.path.join(HERE, "mlp_pooled_internal.jsonl")


def _norm(config: dict) -> str:
    cfg = {**DEFAULTS, **config}
    if cfg.get("noise", 0.0) == 0.0:
        cfg.pop("noise", None)  # trials logged before the noise option existed
    return json.dumps(cfg, sort_keys=True)


def _load_log(path: str) -> dict:
    out = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                out[(row["name"], _norm(row["config"]))] = row
    return out


def run_trial(name: str, config: dict, log, logged: dict) -> dict:
    p = {**DEFAULTS, **config}
    row = logged.get((name, _norm(p)))
    reused = row is not None
    if not reused:
        t = time.time()
        fit = make_fit(**p)
        m = harness.dev_eval(fit)
        ranks = m.pop("ranks")
        row = {
            "name": name,
            "config": p,
            "dev": m,
            "ranks": ranks,
            "fit_info": fit.info,
            "seconds": round(time.time() - t, 1),
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
    TRIALS.append(row)
    m = row["dev"]
    print(
        f"[{len(TRIALS):2d}] {name:28s} sel={m['selection_score']:.4f} top1={m['top1']:.3f} "
        f"top3={m['top3']:.3f} top5={m['top5']:.3f} mrr={m['mrr']:.4f} f1={m['macro_f1']:.4f} "
        f"fit={row['fit_info']} t={row['seconds']}s{' (logged)' if reused else ''}",
        flush=True,
    )
    return row


def _sel(row):
    return row["dev"]["selection_score"]


INTERNAL_GRID = [  # stage 2: (name, change relative to the stage-1 choice); training data only
    ("base", {}),
    ("epochs8", {"epochs": 8, "resample_every": 2}),
    ("input_dropout0.3", {"input_dropout": 0.3}),
    ("noise0.02", {"noise": 0.02}),
    ("noise0.04", {"noise": 0.04}),
    ("wd0.05", {"weight_decay": 0.05}),
    ("dropout0.5", {"dropout": 0.5}),
    ("hidden256", {"hidden": [256]}),
    ("mean_only", {"pooling": "mean"}),
    ("distract0.5", {"distract_max": 0.5}),
    # second round, mean pooling
    ("mean_dropout0.5", {"pooling": "mean", "dropout": 0.5}),
    ("mean_noise0.02", {"pooling": "mean", "noise": 0.02}),
    ("mean_distract0.5", {"pooling": "mean", "distract_max": 0.5}),
    ("mean_members3", {"pooling": "mean", "members": 3}),
    (
        "mean_cases400_e30",
        {"pooling": "mean", "cases_per_group": 400, "epochs": 30, "resample_every": 5},
    ),
    ("mean_hidden1024", {"pooling": "mean", "hidden": [1024]}),
    ("mean_swrate0.95", {"pooling": "mean", "software_rate": 0.95}),
    ("mean_sw5_cap20", {"pooling": "mean", "max_software": 5, "sw_cap": 20}),
    ("mean_input_dropout0.0", {"pooling": "mean", "input_dropout": 0.0}),
    # third round, combining the gains
    (
        "C1_mean_c400_sw5cap20",
        {
            "pooling": "mean",
            "cases_per_group": 400,
            "epochs": 30,
            "resample_every": 5,
            "max_software": 5,
            "sw_cap": 20,
        },
    ),
    (
        "C2_C1_swrate0.95",
        {
            "pooling": "mean",
            "cases_per_group": 400,
            "epochs": 30,
            "resample_every": 5,
            "max_software": 5,
            "sw_cap": 20,
            "software_rate": 0.95,
        },
    ),
    (
        "C4_C1_dropout0.5_distract0.5",
        {
            "pooling": "mean",
            "cases_per_group": 400,
            "epochs": 30,
            "resample_every": 5,
            "max_software": 5,
            "sw_cap": 20,
            "dropout": 0.5,
            "distract_max": 0.5,
        },
    ),
    (
        "C3_C1_members3",
        {
            "pooling": "mean",
            "cases_per_group": 400,
            "epochs": 30,
            "resample_every": 5,
            "max_software": 5,
            "sw_cap": 20,
            "members": 3,
        },
    ),
    (
        "C5_C4_members3",
        {
            "pooling": "mean",
            "cases_per_group": 400,
            "epochs": 30,
            "resample_every": 5,
            "max_software": 5,
            "sw_cap": 20,
            "dropout": 0.5,
            "distract_max": 0.5,
            "members": 3,
        },
    ),
]


def stage2_internal() -> list:
    """Internal held-out-sentence validation rows (training data only); logged rows are reused."""
    base = {"pooling": "meanmax", "distract_max": 0.2}  # the stage-1 network choice
    logged = {}
    if os.path.isfile(INTERNAL_LOG):
        with open(INTERNAL_LOG) as fh:
            for line in fh:
                row = json.loads(line)
                logged[row["name"]] = row
    rows = []
    for name, change in INTERNAL_GRID:
        cfg = {**base, **change}
        row = logged.get(name)
        if row is None or row["config"] != cfg:
            import mlp_pooled_internal as iv  # builds the 75/25 training-data split

            row = {"name": name, "config": cfg, **iv.score(cfg)}
            with open(INTERNAL_LOG, "a") as f:
                f.write(json.dumps(row) + "\n")
        rows.append(row)
        print(
            f"  internal {name:30s} val_mrr={row['val_mrr']:.4f} val_top1={row['val_top1']:.4f}",
            flush=True,
        )
    return rows


if __name__ == "__main__":
    TIE = 0.01  # a change must beat the current best by more than this to be adopted
    os.makedirs(HERE, exist_ok=True)
    logged = _load_log(TRIAL_LOG)
    log = open(TRIAL_LOG, "a")  # noqa: SIM115 (appended to by every trial of the search)
    cur = dict(DEFAULTS)
    best_row = run_trial("base(mean,[512],p=1)", cur, log, logged)

    # Stage 1 (dev): one factor at a time against the current best (prior_power fixed at 1.0)
    stage1 = [
        ("meanmax", {"pooling": "meanmax"}),
        ("hidden[1024,512]", {"hidden": [1024, 512]}),
        ("distract_max0.2", {"distract_max": 0.2}),
        ("software_rate0.4", {"software_rate": 0.4}),
        ("software_rate0.95", {"software_rate": 0.95}),
        ("cases400_epochs30", {"cases_per_group": 400, "epochs": 30, "resample_every": 5}),
        ("input_dropout0.3", {"input_dropout": 0.3}),
        ("smoothing0.0", {"label_smoothing": 0.0}),
    ]
    for name, change in stage1:
        if name == "software_rate0.95" and cur["software_rate"] != DEFAULTS["software_rate"]:
            continue  # 0.4 already adopted; the other direction is moot
        row = run_trial(name, {**cur, **change}, log, logged)
        if _sel(row) > _sel(best_row) + TIE:
            cur, best_row = {**cur, **change}, row
    for power in (0.0, 0.5, 2.0, 3.0):
        row = run_trial(f"prior_power{power}", {**cur, "prior_power": power}, log, logged)
        better = _sel(row) > _sel(best_row) + 1e-9
        tie_simpler = abs(_sel(row) - _sel(best_row)) <= 1e-9 and power < cur["prior_power"]
        if better or tie_simpler:
            cur, best_row = {**cur, "prior_power": power}, row
    stage1_choice = dict(cur)
    stage1_seeds = [_sel(best_row)]
    for seed in (8, 9):
        stage1_seeds.append(_sel(run_trial(f"best_seed{seed}", {**cur, "seed": seed}, log, logged)))
    # Stage 1 verdict: seed std ~0.07 on dev, as large as every config difference; dev cannot
    # rank network settings, so stage 2 ranks them on training data instead.

    # Stage 2 (training data only): held-out-sentence synthetic validation
    internal = stage2_internal()
    single = [r for r in internal if r["config"].get("members", 1) == 1]
    pick = max(single, key=lambda r: r["val_mrr"])
    net_cfg = {**DEFAULTS, **pick["config"], "members": 3}  # 3 members: variance reduction only
    ens_row = next((r for r in internal if r["config"] == {**pick["config"], "members": 3}), None)
    print(
        "stage-2 pick",
        pick["name"],
        pick["val_mrr"],
        "-> members 3",
        ens_row and ens_row["val_mrr"],
        flush=True,
    )

    # Stage 3 (dev): prior power for the chosen network (cached between powers, one fit)
    rows = {}
    for power in (0.0, 1.0, 2.0):
        rows[power] = run_trial(f"S3_prior{power}", {**net_cfg, "prior_power": power}, log, logged)
    if max(rows, key=lambda k: (_sel(rows[k]), -k)) == 2.0:
        rows[3.0] = run_trial("S3_prior3.0", {**net_cfg, "prior_power": 3.0}, log, logged)
    power = max(rows, key=lambda k: (_sel(rows[k]), -k))  # ties -> smaller prior
    cur, best_row = {**net_cfg, "prior_power": power}, rows[power]
    seed_scores = [{"seed": cur["seed"], "selection_score": _sel(best_row), "dev": best_row["dev"]}]
    for seed in (8, 9):
        row = run_trial(f"S3_best_seed{seed}", {**cur, "seed": seed}, log, logged)
        seed_scores.append({"seed": seed, "selection_score": _sel(row), "dev": row["dev"]})
    # reference: the same network with one member (is the ensemble worth its 3x fit time?)
    single_scores = [
        {
            "seed": cur["seed"],
            "selection_score": _sel(
                run_trial("S3_ref_members1", {**cur, "members": 1}, log, logged)
            ),
        }
    ]
    for seed in (8, 9):
        row = run_trial(
            f"S3_ref_members1_seed{seed}", {**cur, "members": 1, "seed": seed}, log, logged
        )
        single_scores.append({"seed": seed, "selection_score": _sel(row), "dev": row["dev"]})
    single_scores[0]["dev"] = TRIALS[-3]["dev"]
    mean3 = float(np.mean([s["selection_score"] for s in seed_scores]))
    mean1 = float(np.mean([s["selection_score"] for s in single_scores]))
    print(f"3-seed mean: members3 {mean3:.4f}  members1 {mean1:.4f}", flush=True)
    ensemble_scores = seed_scores
    if mean1 >= mean3 - 0.02:  # tie (within 0.02) or better: the simpler single network wins
        cur, seed_scores = {**cur, "members": 1}, single_scores
        best_row = TRIALS[-3]
    scores = [s["selection_score"] for s in seed_scores]

    out = {
        "best": cur,
        "dev": best_row["dev"],
        "trials": [
            {k: r[k] for k in ("name", "config", "dev", "fit_info", "seconds")} for r in TRIALS
        ],
        "internal_validation": internal,
        "seed_scores": seed_scores,
        "seed_mean": round(float(np.mean(scores)), 4),
        "seed_std": round(float(np.std(scores)), 4),
        "members3_seed_scores": ensemble_scores,
        "members1_seed_scores": single_scores,
        "stage1_choice": stage1_choice,
        "stage1_seed_scores": stage1_seeds,
        "n_dev_eval_calls": len(TRIALS),
    }
    with open(os.path.join(HERE, "mlp_pooled.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("BEST", json.dumps(cur))
    print("seed mean/std", out["seed_mean"], out["seed_std"])
