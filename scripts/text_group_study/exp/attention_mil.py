"""Family "attention_mil": multiple-instance learning over a report's sentence embeddings.

Model (PyTorch, CPU)
- Instance encoder: x * sqrt(768) -> input dropout -> Linear(768, hidden) -> LayerNorm -> GELU -> dropout.
- Pooling: ``attention`` = gated attention (Ilse, Tomczak & Welling 2018):
  a_i = softmax_i( w^T (tanh(V h_i) * sigmoid(U h_i)) ), z = sum_i a_i h_i ; or ``mean`` = masked mean
  of h_i (the comparison asked for: same encoder and classifier, no attention).
- Classifier: Linear(hidden, n_groups) over ``train.groups``. Report score = logits
  + prior_power * train.log_prior (the knowledge-base prominence prior).

Training data: synthetic report-like bags drawn from the training corpus only (redrawn each epoch,
seeded). For a group g (every group with at least one sentence, sampled uniformly):
- bag size n ~ U[size_min, size_max] true sentences (capped by what is available);
- g's own sentences: records labelled g with source group / group_desc / campaign (campaign rows
  exist only in FINAL mode);
- with probability ``sw_rate`` the bag also draws on 1..``max_sw`` of g's software (software
  records whose groups include g); then a share ~ U[0, sw_share_max] of the true sentences comes
  from those tools' procedure sentences (the share is capped by what the tools have);
- distractors: a fraction d ~ U[0, distract_max] of the final bag is drawn uniformly from records
  not labelled g (any source, including software with no group).
Bags are padded and masked per batch (sorted by length into batches, batch order shuffled).
Loss: cross-entropy with label smoothing; AdamW + OneCycle.

No group ids, campaign ids or record owners are used to recognise anything; owners are only used
to group software sentences by tool when composing synthetic bags.
Tuning: only harness.dev_eval (23 ATT&CK campaigns), small staged grid; every call is logged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the study code folder (harness)
import harness  # noqa: E402

torch.set_num_threads(4)

HERE = harness.STUDY_DIR / "exp"  # trial log and result (data, not code)
KEY = "attention_mil"
OWN_SOURCES = ("group", "group_desc", "campaign")

DEFAULTS = dict(
    pooling="attention",  # "attention" (gated) | "mean"
    hidden=256,  # instance embedding width
    att_dim=128,  # gated-attention inner width
    dropout=0.2,
    input_dropout=0.1,
    # synthetic bags
    size_min=3,
    size_max=30,
    sw_rate=0.7,
    max_sw=5,
    sw_share_max=0.7,
    distract_max=0.5,
    bags_per_group=64,  # per epoch
    epochs=25,
    # optimisation
    batch=256,
    lr=2e-3,
    wd=1e-4,
    label_smoothing=0.1,
    seed=0,
    members=1,  # networks averaged (logits); member j trains with seed + 1000 * j
    # prediction
    prior_power=1.0,
)
_PREDICT_KEYS = ("prior_power",)


# ---------------------------------------------------------------------------- data
class BagSampler:
    """Index structures over ``train.records`` and the synthetic bag generator."""

    def __init__(self, train, cfg):
        self.cfg = cfg
        gi = {g: i for i, g in enumerate(train.groups)}
        n = len(train.records)
        self.n_records = n
        own = {i: [] for i in range(len(train.groups))}
        tools = {i: {} for i in range(len(train.groups))}
        member = np.zeros((n, len(train.groups)), bool)
        for row, r in enumerate(train.records):
            for g in r["groups"] or ():
                if g not in gi:
                    continue
                k = gi[g]
                member[row, k] = True
                if r["source"] in OWN_SOURCES:
                    own[k].append(row)
                elif r["source"] == "software":
                    tools[k].setdefault(r["owner"], []).append(row)
        self.member = member
        self.own = {k: np.asarray(v, np.int64) for k, v in own.items()}
        self.tools = {
            k: [np.asarray(v, np.int64) for _, v in sorted(t.items())] for k, t in tools.items()
        }
        self.classes = [
            k for k in range(len(train.groups)) if len(self.own[k]) or len(self.tools[k])
        ]

    def bag(self, k, rng):
        c = self.cfg
        n = int(rng.integers(c["size_min"], c["size_max"] + 1))
        own, tools = self.own[k], self.tools[k]
        sw_rows = np.zeros(0, np.int64)
        if tools and (rng.uniform() < c["sw_rate"] or len(own) == 0):
            m = int(rng.integers(1, min(c["max_sw"], len(tools)) + 1))
            chosen = rng.choice(len(tools), size=m, replace=False)
            sw_rows = np.unique(np.concatenate([tools[i] for i in chosen]))
        if len(sw_rows):
            share = 1.0 if len(own) == 0 else rng.uniform(0.0, c["sw_share_max"])
            n_sw = min(len(sw_rows), int(round(n * share)))
        else:
            n_sw = 0
        n_own = min(len(own), n - n_sw)
        parts = []
        if n_own:
            parts.append(rng.choice(own, size=n_own, replace=False))
        if n_sw:
            parts.append(rng.choice(sw_rows, size=n_sw, replace=False))
        true = np.concatenate(parts) if parts else rng.choice(own if len(own) else sw_rows, size=1)
        d = rng.uniform(0.0, c["distract_max"])
        n_dis = int(round(len(true) * d / (1.0 - d))) if d < 1 else 0
        if n_dis:
            cand = rng.integers(0, self.n_records, size=2 * n_dis + 8)
            cand = cand[~self.member[cand, k]][:n_dis]
            true = np.concatenate([true, cand])
        return rng.permutation(true)

    def epoch(self, rng):
        labels = np.repeat(np.asarray(self.classes, np.int64), self.cfg["bags_per_group"])
        bags = [self.bag(int(k), rng) for k in labels]
        return bags, labels


def _batches(bags, labels, batch, rng):
    order = np.argsort([len(b) for b in bags], kind="stable")
    chunks = [order[s : s + batch] for s in range(0, len(order), batch)]
    for j in rng.permutation(len(chunks)):
        idx = chunks[j]
        L = max(len(bags[i]) for i in idx)
        pad = np.zeros((len(idx), L), np.int64)
        mask = np.zeros((len(idx), L), bool)
        for r, i in enumerate(idx):
            b = bags[i]
            pad[r, : len(b)] = b
            mask[r, : len(b)] = True
        yield torch.from_numpy(pad), torch.from_numpy(mask), torch.from_numpy(labels[idx])


# ---------------------------------------------------------------------------- network
class MIL(nn.Module):
    def __init__(self, n_classes, cfg, d_in=768):
        super().__init__()
        h = int(cfg["hidden"])
        self.scale = math.sqrt(d_in)
        self.pooling = cfg["pooling"]
        self.enc = nn.Sequential(
            nn.Dropout(float(cfg["input_dropout"])),
            nn.Linear(d_in, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
        )
        # attention layers always exist (unused for mean pooling) so both variants share structure
        self.V = nn.Linear(h, int(cfg["att_dim"]))
        self.U = nn.Linear(h, int(cfg["att_dim"]))
        self.w = nn.Linear(int(cfg["att_dim"]), 1)
        self.cls = nn.Linear(h, n_classes)

    def forward(self, x, mask, return_attention=False):
        hid = self.enc(x * self.scale)
        if self.pooling == "attention":
            a = self.w(torch.tanh(self.V(hid)) * torch.sigmoid(self.U(hid))).squeeze(-1)
            a = a.masked_fill(~mask, float("-inf")).softmax(1)
        elif self.pooling == "mean":
            a = mask.float() / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            raise ValueError(self.pooling)
        z = (a.unsqueeze(-1) * hid).sum(1)
        out = self.cls(z)
        return (out, a) if return_attention else out


def _train_model(train, cfg, log=None):
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sampler = BagSampler(train, cfg)
    X = torch.from_numpy(np.ascontiguousarray(train.X, dtype=np.float32))
    model = MIL(len(train.groups), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    epochs = int(cfg["epochs"])
    steps_per_epoch = -(-len(sampler.classes) * int(cfg["bags_per_group"]) // int(cfg["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(cfg["lr"]), total_steps=epochs * steps_per_epoch + 1
    )
    t0 = time.time()
    for ep in range(epochs):
        bags, labels = sampler.epoch(rng)
        model.train()
        tot, cnt, correct = 0.0, 0, 0
        for idx, mask, y in _batches(bags, labels, int(cfg["batch"]), rng):
            out = model(X[idx], mask)
            loss = F.cross_entropy(out, y, label_smoothing=float(cfg["label_smoothing"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(y)
            cnt += len(y)
            correct += int((out.argmax(1) == y).sum())
        if log:
            log(
                f"  epoch {ep + 1}/{epochs} loss {tot / cnt:.4f} train_acc {correct / cnt:.3f} "
                f"{time.time() - t0:.0f}s"
            )
    model.eval()
    return model


def _fingerprint(train):
    h = hashlib.md5()
    h.update(np.ascontiguousarray(train.X).tobytes())
    h.update(json.dumps([len(train.records), list(train.groups)]).encode())
    return h.hexdigest()


_MODEL_CACHE: dict = {}


def make_fit(**config):
    cfg = {**DEFAULTS, **config}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")

    def fit(train):
        train_cfg = {k: v for k, v in cfg.items() if k not in _PREDICT_KEYS}
        key = (_fingerprint(train), json.dumps(train_cfg, sort_keys=True))
        if key not in _MODEL_CACHE:
            t0 = time.time()
            _MODEL_CACHE.clear()  # keep memory flat: one model at a time
            models = [
                _train_model(
                    train,
                    {**cfg, "seed": int(cfg["seed"]) + 1000 * j},
                    log=print if VERBOSE else None,
                )
                for j in range(int(cfg["members"]))
            ]
            _MODEL_CACHE[key] = (models, time.time() - t0)
        models, fit_seconds = _MODEL_CACHE[key]
        fit.fit_seconds = fit_seconds
        prior = torch.from_numpy(np.asarray(train.log_prior, np.float32))
        power = float(cfg["prior_power"])

        def predict(docs):
            out = []
            with torch.no_grad():
                for D in docs:
                    x = torch.from_numpy(np.ascontiguousarray(np.asarray(D), dtype=np.float32))[
                        None
                    ]
                    mask = torch.ones(1, x.shape[1], dtype=torch.bool)
                    logits = torch.stack([m(x, mask)[0] for m in models]).mean(0)
                    out.append((logits + power * prior).numpy())
            return np.stack(out)

        return predict

    return fit


VERBOSE = False

# chosen config (from the search below; see attention_mil.json / attention_mil_trials.jsonl):
# gated attention, 40 epochs, prior_power 1.0, one network, seed 0. Dev selection 0.6242 at seed 0;
# 3-seed mean 0.5935 (std 0.11), below the knn10 baseline (0.639).
BEST = dict(DEFAULTS, pooling="attention", epochs=40, prior_power=1.0, seed=0, members=1)


# ---------------------------------------------------------------------------- search
LOG = HERE / f"{KEY}_trials.jsonl"


def _cfg_key(cfg):
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)


def _load_log():
    rows = {}
    if LOG.is_file():
        for line in LOG.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["config"] = {**DEFAULTS, **r["config"]}  # keys added later take their default
                rows[_cfg_key(r["config"])] = r
    return rows


def run_trial(stage, cfg, done):
    cfg = {**DEFAULTS, **cfg}
    k = _cfg_key(cfg)
    if k in done:  # an identical config already evaluated (resume): reuse, do not re-spend dev
        print(f"[{stage}] reuse {json.dumps(done[k]['dev'])}")
        return done[k]
    fit = make_fit(**cfg)
    t0 = time.time()
    m = harness.dev_eval(fit)
    m.pop("ranks", None)
    m.pop("top1_wilson", None)
    row = {
        "stage": stage,
        "config": cfg,
        "dev": m,
        "fit_seconds": round(getattr(fit, "fit_seconds", float("nan")), 1),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with LOG.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    done[k] = row
    print(
        f"[{stage}] {json.dumps({k2: cfg[k2] for k2 in cfg if cfg[k2] != DEFAULTS[k2]})} -> "
        f"sel {m['selection_score']} top1 {m['top1']} top3 {m['top3']} top5 {m['top5']} "
        f"mrr {m['mrr']} f1 {m['macro_f1']} (fit {row['fit_seconds']}s)",
        flush=True,
    )
    return row


def _better(a, b):
    """a strictly better than b (ties keep the incumbent, i.e. the simpler/earlier config)."""
    return a["dev"]["selection_score"] > b["dev"]["selection_score"] + 1e-9


def search():
    done = _load_log()
    trials = []

    def T(stage, cfg):
        r = run_trial(stage, cfg, done)
        trials.append(r)
        return r

    # Stage 1: pooling x prior_power (one fit per pooling, prior varied on the cached model)
    s1 = {}
    for pooling in ("attention", "mean"):
        for pp in (0.0, 1.0, 2.0):
            s1[(pooling, pp)] = T("pooling_prior", {"pooling": pooling, "prior_power": pp})
    inc = max(s1.values(), key=lambda r: r["dev"]["selection_score"])
    for r in s1.values():  # tie -> prefer the attention model / smaller prior (earlier in order)
        if r["dev"]["selection_score"] == inc["dev"]["selection_score"]:
            inc = r
            break
    base = dict(inc["config"])
    print("incumbent after stage 1:", {k: base[k] for k in ("pooling", "prior_power")})

    # Stage 2: one-at-a-time coarse changes to the data / capacity (at the incumbent prior)
    variants = [
        ("no_distractors", {"distract_max": 0.0}),
        ("no_software", {"sw_rate": 0.0}),
        ("hidden512", {"hidden": 512}),
        ("longer", {"epochs": 40}),
    ]
    for name, change in variants:
        r = T(f"oat_{name}", {**base, **change})
        if _better(r, inc):
            inc = r
            base = dict(r["config"])
    print("incumbent after stage 2:", base)

    # Stage 3: prior_power around the incumbent on the final training config
    pp0 = base["prior_power"]
    for pp in sorted({max(0.0, pp0 - 0.5), pp0 + 0.5}):
        if pp == pp0:
            continue
        r = T("prior_refine", {**base, "prior_power": pp})
        if _better(r, inc):
            inc = r
    base = dict(inc["config"])

    # Comparison: the other pooling with the same final config (honest A/B on the final setting)
    other = "mean" if base["pooling"] == "attention" else "attention"
    cmp_row = T("pooling_compare_final", {**base, "pooling": other})

    # Seeds: the chosen config at seeds 1 and 2 (seed 0 is the incumbent)
    seed_rows = [inc] + [T("seed", {**base, "seed": s}) for s in (1, 2)]
    seed_scores = [r["dev"]["selection_score"] for r in seed_rows]

    # Stage 5 (added after the first pass, where the other pooling scored higher at seed 0 on
    # the final config): decide the pooling on 3-seed means instead of one seed. The other
    # pooling gets its own prior refinement (+-0.5) at seed 0 and seeds 1, 2 at its best prior.
    # Ties (|diff| <= 0.01) keep mean pooling, the simpler model.
    o_inc = cmp_row
    for pp in sorted({max(0.0, base["prior_power"] - 0.5), base["prior_power"] + 0.5}):
        if pp == base["prior_power"]:
            continue
        r = T("other_prior_refine", {**cmp_row["config"], "prior_power": pp})
        if _better(r, o_inc):
            o_inc = r
    o_base = dict(o_inc["config"])
    o_rows = [o_inc] + [T("other_seed", {**o_base, "seed": s}) for s in (1, 2)]
    o_scores = [r["dev"]["selection_score"] for r in o_rows]
    print(
        f"3-seed means: {base['pooling']} {np.mean(seed_scores):.4f} {seed_scores} | "
        f"{o_base['pooling']} {np.mean(o_scores):.4f} {o_scores}"
    )
    pooling_seeds = {
        base["pooling"]: {"config": base, "seed_scores": seed_scores},
        o_base["pooling"]: {"config": o_base, "seed_scores": o_scores},
    }
    a, m_ = pooling_seeds.get("attention"), pooling_seeds.get("mean")
    chosen = a if np.mean(a["seed_scores"]) > np.mean(m_["seed_scores"]) + 0.01 else m_
    if chosen["config"] == o_base:
        base, inc, seed_rows, seed_scores, cmp_row = o_base, o_inc, o_rows, o_scores, inc

    # Stage 6 (added after stage 5 showed a seed std of ~0.1): a 3-member logit ensemble of the
    # chosen config, each member on half the bags per epoch (bags_per_group 32) so one fit stays
    # under ~5 min. Adopted only if its 3-seed mean beats the single network's by > 0.02
    # (it is the more complex model).
    ens = {**base, "members": 3, "bags_per_group": 32}
    e_rows = [T("ensemble_seed", {**ens, "seed": s}) for s in (0, 1, 2)]
    e_scores = [r["dev"]["selection_score"] for r in e_rows]
    print(
        f"3-seed means: single {np.mean(seed_scores):.4f} {seed_scores} | "
        f"ensemble3 {np.mean(e_scores):.4f} {e_scores}"
    )
    pooling_seeds["ensemble3_" + base["pooling"]] = {
        "config": e_rows[0]["config"],
        "seed_scores": e_scores,
    }
    if np.mean(e_scores) > np.mean(seed_scores) + 0.02:
        base, inc, seed_rows, seed_scores = dict(e_rows[0]["config"]), e_rows[0], e_rows, e_scores

    # Stage 7 (added after stages 2 and 6 both pointed to under-training: 25 -> 40 epochs helped,
    # half the bags per member hurt): the chosen single network at 60 epochs, 3 seeds. This is
    # the last stage (dev_eval budget ~25). Adopted only if the 3-seed mean beats the 40-epoch
    # one by > 0.01 (otherwise the cheaper fit stays).
    if base["members"] == 1:
        l_rows = [T("epochs60_seed", {**base, "epochs": 60, "seed": s}) for s in (0, 1, 2)]
        l_scores = [r["dev"]["selection_score"] for r in l_rows]
        print(
            f"3-seed means: epochs {base['epochs']} {np.mean(seed_scores):.4f} {seed_scores} | "
            f"epochs 60 {np.mean(l_scores):.4f} {l_scores}"
        )
        pooling_seeds["epochs60_" + base["pooling"]] = {
            "config": l_rows[0]["config"],
            "seed_scores": l_scores,
        }
        if np.mean(l_scores) > np.mean(seed_scores) + 0.01:
            base, inc, seed_rows, seed_scores = (
                dict(l_rows[0]["config"]),
                l_rows[0],
                l_rows,
                l_scores,
            )
    return base, inc, trials, seed_rows, seed_scores, cmp_row, pooling_seeds


if __name__ == "__main__":
    VERBOSE = "--verbose" in sys.argv
    HERE.mkdir(parents=True, exist_ok=True)
    best, inc, trials, seed_rows, seed_scores, cmp_row, pooling_seeds = search()
    out = {
        "key": KEY,
        "best": best,
        "dev": inc["dev"],
        "fit_seconds": inc["fit_seconds"],
        "trials": trials,
        "seed_scores": seed_scores,
        "seed_mean": round(float(np.mean(seed_scores)), 4),
        "seed_std": round(float(np.std(seed_scores)), 4),
        "pooling_compare_final": cmp_row,
        "pooling_3seed": pooling_seeds,
        "baseline_knn10_selection": 0.639,
        "dev_eval_calls": len(_load_log()),
    }
    (HERE / f"{KEY}.json").write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {k: out[k] for k in ("best", "dev", "seed_scores", "seed_mean", "seed_std")}, indent=2
        )
    )
