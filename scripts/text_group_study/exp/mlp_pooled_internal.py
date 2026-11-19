"""Internal validation on TRAINING data only (no dev/test data): hold out 25% of every group's own
sentences and of every software's sentences, fit on the rest, score synthetic reports built only
from the held-out sentences. Used to choose regularisation (noise, dropout, epochs, members)
without spending the 23-case dev set. Prior power 0 (the synthetic validation is group-balanced)."""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # harness (scripts/text_group_study)
sys.path.insert(0, _HERE)  # mlp_pooled (exp/)
import harness  # noqa: E402
import mlp_pooled as m  # noqa: E402
import numpy as np  # noqa: E402

tr = harness._train("dev")
rng = np.random.default_rng(2024)
key = [
    (r["source"] if r["source"] != "software" else "sw:" + r["owner"], r["owner"])
    for r in tr.records
]
held = np.zeros(len(tr.records), bool)
by = {}
for i, k in enumerate(key):
    by.setdefault(k, []).append(i)
for _k, rows in by.items():
    rows = np.asarray(rows)
    if len(rows) >= 4:
        held[rng.choice(rows, size=len(rows) // 4, replace=False)] = True
fit_idx, val_idx = np.where(~held)[0], np.where(held)[0]


def sub(idx):
    return harness.Train(
        [tr.records[i] for i in idx], tr.X[idx], tr.groups, tr.log_prior, tr.group_software
    )


tr_fit, tr_val = sub(fit_idx), sub(val_idx)


# fixed validation reports from held-out sentences (true share only from held-out own + tools)
def val_reports(p_val, n_per=10, seed=99):
    pools = m.Pools(tr_val, {**m.DEFAULTS, "min_group_texts": 1})
    rng = np.random.default_rng(seed)
    docs, ys = [], []
    for g in tr.groups:
        own = pools.own[g]
        if own.size < 3:
            continue
        for _ in range(n_per):
            size = int(rng.integers(3, 31))
            parts = [own]
            tools = pools.tools[g]
            if tools and rng.uniform() < 0.7:
                k = int(rng.integers(1, min(3, len(tools)) + 1))
                parts += [
                    pools.sw[tools[t]][:10] for t in rng.choice(len(tools), size=k, replace=False)
                ]
            pool = np.unique(np.concatenate(parts))
            take = rng.choice(pool, size=min(size, pool.size), replace=False)
            n_d = int(round(take.size * rng.uniform(0, 0.3)))
            if n_d:
                cand = rng.integers(0, len(tr_val.records), size=3 * n_d + 4)
                take = np.concatenate([take, cand[~pools.labelled[cand, pools.gi[g]]][:n_d]])
            docs.append(tr_val.X[take])
            ys.append(pools.gi[g])
    return docs, np.asarray(ys)


VDOCS, VY = val_reports(None)
print(
    "held-out",
    held.sum(),
    "fit",
    len(fit_idx),
    "val reports",
    len(VY),
    "groups",
    len(set(VY.tolist())),
    flush=True,
)


def score(config):
    p = {**m.DEFAULTS, **config, "prior_power": 0.0}
    t = time.time()
    fit = m.make_fit(**p)
    pred = fit(tr_fit)
    S = np.asarray(pred(VDOCS))
    rank = (S[np.arange(len(VY)), VY][:, None] < S).sum(1) + 1
    out = {
        "val_top1": round(float((rank == 1).mean()), 4),
        "val_mrr": round(float((1 / rank).mean()), 4),
        "val_top5": round(float((rank <= 5).mean()), 4),
        "train_last": fit.info.get("last_epoch_loss_acc"),
        "seconds": round(time.time() - t, 1),
    }
    return out


if __name__ == "__main__":
    base = {"pooling": "meanmax", "distract_max": 0.2}
    grid = [
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
    ]
    if len(sys.argv) > 1:
        grid = [(n, json.loads(c)) for n, c in zip(sys.argv[1::2], sys.argv[2::2], strict=False)]
    with open(os.path.join(m.HERE, "mlp_pooled_internal.jsonl"), "a") as f:
        for name, change in grid:
            r = score({**base, **change})
            row = {"name": name, "config": {**base, **change}, **r}
            print(json.dumps(row), flush=True)
            f.write(json.dumps(row) + "\n")
            f.flush()
