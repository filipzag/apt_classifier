"""Per-report ranks of the headline text model, its combination and the pipeline; paired stats."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "exp"))
import final_eval
import harness
import lexical_fixed as L

groups = harness._load()["groups"]
_, test_mlp = final_eval.mlp_inputs()
fit = L.make_fit(**L.BEST)(harness._train("final"))
out = {}
rng = np.random.default_rng(0)
for name, art in (
    ("cti_hal_names", "attribution_cti_hal_current.json"),
    ("elastic_strict", "attribution_elastic_current.json"),
):
    docs, truth, accept, ids = harness.test_docs(name)
    s = np.asarray(fit(docs))
    text = harness.metrics(s, truth, accept, groups)["ranks"]
    combo = harness.metrics(
        final_eval.log_softmax(s) + 0.25 * test_mlp[name], truth, accept, groups
    )["ranks"]
    rows = {r["report_id"]: r for r in json.loads((harness.AUDITS / art).read_text())["known"]}
    pipe = [rows[i].get("rank") for i in ids]

    def rr(xs):
        return np.array([1 / x if x else 0.0 for x in xs])

    res = {
        "reports": [
            {"report_id": i, "truth": t, "text": a, "combo": b, "pipeline": c}
            for i, t, a, b, c in zip(ids, truth, text, combo, pipe, strict=False)
        ]
    }
    for label, ranks in (("text", text), ("combo", combo)):
        d = rr(ranks) - rr(pipe)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        t1 = np.array([1 if x == 1 else 0 for x in ranks]) - np.array(
            [1 if x == 1 else 0 for x in pipe]
        )
        res[label] = {
            "mrr_diff": round(float(d.mean()), 4),
            "mrr_diff_ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
            "p_diff_le_0": round(float(np.mean(np.array(boot) <= 0)), 4),
            "better": int((d > 0).sum()),
            "worse": int((d < 0).sum()),
            "same": int((d == 0).sum()),
            "top1_gained": int((t1 > 0).sum()),
            "top1_lost": int((t1 < 0).sum()),
        }
    out[name] = res
    print(name, {k: v for k, v in res.items() if k != "reports"})
with open(harness.STUDY_DIR / "final_paired.json", "w") as fh:
    json.dump(out, fh, indent=1)
