"""Orchestrator-only test run: every experiment's BEST on the test sets, a near-copy ablation,
and the combination with the technique+software group network (weight chosen on dev campaigns).

    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/text_group_study/final_eval.py key1 key2 ...

Writes STUDY_DIR/final_eval.json; reads the pipeline artifacts from harness.AUDITS
($TEXT_GROUP_AUDITS_DIR, default <repo>/data/audits).
"""

import importlib.util
import json
import sys
from pathlib import Path

import harness
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(harness.REPO))
AUD = harness.AUDITS
NEAR_COS = 0.92


def load_exp(key):
    spec = importlib.util.spec_from_file_location(key, HERE / "exp" / f"{key}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log_softmax(s):
    s = s - s.max(1, keepdims=True)
    return s - np.log(np.exp(s).sum(1, keepdims=True))


def strip(m):
    return {k: v for k, v in m.items() if k not in ("ranks", "report_ids")}


# --- technique+software network inputs ------------------------------------------------------
def mlp_inputs():
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.group_mlp import GroupMLP
    from scripts.fit_group_mlp import campaign_software

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    model = GroupMLP.load(harness.REPO / "data" / "models" / "mlp" / "group_mlp.npz", device="cpu")
    used = campaign_software(harness.REPO / "data" / "raw" / "enterprise-attack.json")
    _, _, _, dev_ids = harness.dev_docs()
    dev_x = np.stack(
        [model.features.vector(kb.campaigns[c].technique_ids, used.get(c, ())) for c in dev_ids]
    )
    sets = {}
    for name, f in (
        ("cti_hal_names", "attribution_cti_hal_current.json"),
        ("elastic_strict", "attribution_elastic_current.json"),
        ("elastic_tradecraft", "attribution_elastic_tradecraft.json"),
    ):
        rows = {r["report_id"]: r for r in json.loads((AUD / f).read_text())["known"]}
        _, _, _, ids = harness.test_docs(name)
        sets[name] = np.stack(
            [
                model.features.vector(rows[i]["accepted_techniques"], rows[i]["software_ids"])
                for i in ids
            ]
        )
    order = [
        model.group_ids.index(g) if g in model.group_ids else -1 for g in harness._load()["groups"]
    ]

    def logp(x):
        p = model.predict(x)
        out = np.full((len(x), len(order)), -30.0)
        for j, i in enumerate(order):
            if i >= 0:
                out[:, j] = np.log(np.clip(p[:, i], 1e-12, None))
        return out

    return logp(dev_x), {k: logp(v) for k, v in sets.items()}


def main(keys):
    report = {"pipeline": harness.pipeline_baselines(), "models": {}}
    dev_mlp, test_mlp = mlp_inputs()
    _, dev_truth, dev_accept, _ = harness.dev_docs()
    groups = harness._load()["groups"]
    report["models"]["technique_network"] = {
        "dev": strip(harness.metrics(dev_mlp, dev_truth, dev_accept, groups)),
        **{
            k: strip(harness.metrics(v, *harness.test_docs(k)[1:3], groups))
            for k, v in test_mlp.items()
        },
    }
    for key in keys:
        mod = load_exp(key)
        fit = mod.make_fit(**mod.BEST)
        entry = {"best": mod.BEST}
        # dev (re-run) and dev scores kept for the combination weight
        dev_train = harness._train("dev")
        dev_pred = fit(dev_train)
        dev_docs, _, _, _ = harness.dev_docs()
        dev_scores = np.asarray(dev_pred(dev_docs))
        entry["dev"] = strip(harness.metrics(dev_scores, dev_truth, dev_accept, groups))
        # test, FINAL mode
        final = harness._train("final")
        pred = fit(final)
        test_scores = {}
        for name in harness.TEST_SETS:
            docs, truth, accept, ids = harness.test_docs(name)
            s = np.asarray(pred(docs))
            test_scores[name] = s
            entry[name] = strip(harness.metrics(s, truth, accept, groups))
        open_docs, _ = harness.open_docs("elastic_strict")
        lab_docs, _, _, _ = harness.test_docs("elastic_strict")

        def top(s):
            return np.exp(log_softmax(np.asarray(s))).max(1)

        entry["elastic_open_set_top_prob"] = {
            "open": round(float(top(pred(open_docs)).mean()), 4),
            "labelled": round(float(top(pred(lab_docs)).mean()), 4),
        }
        # near-copy ablation: drop training texts within NEAR_COS of any statement of the set
        for name in ("cti_hal_names", "elastic_strict"):
            docs, truth, accept, _ = harness.test_docs(name)
            T = np.concatenate([np.asarray(d) for d in docs])
            near = np.zeros(len(final.records), bool)
            for s0 in range(0, len(final.records), 4096):
                near[s0 : s0 + 4096] = (final.X[s0 : s0 + 4096] @ T.T).max(1) >= NEAR_COS
            keep = ~near
            ablated = harness.Train(
                [r for r, k in zip(final.records, keep, strict=False) if k],
                final.X[keep],
                final.groups,
                final.log_prior,
                final.group_software,
            )
            s = np.asarray(fit(ablated)(docs))
            entry[f"{name}_no_near_copies"] = strip(harness.metrics(s, truth, accept, groups))
            entry[f"{name}_no_near_copies"]["dropped_training_texts"] = int(near.sum())
        # combination with the technique network: weight chosen on dev
        best_w, best = None, -1
        trials = {}
        for w in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            m = harness.metrics(
                log_softmax(dev_scores) + w * dev_mlp, dev_truth, dev_accept, groups
            )
            trials[w] = round(m["mrr"] + m["macro_f1"], 4)
            if trials[w] > best:
                best, best_w = trials[w], w
        entry["combination"] = {"dev_trials": trials, "weight": best_w}
        diag = {}
        for name, mlp in test_mlp.items():
            _, truth, accept, _ = harness.test_docs(name)
            entry["combination"][name] = strip(
                harness.metrics(
                    log_softmax(test_scores[name]) + best_w * mlp, truth, accept, groups
                )
            )
            diag[name] = {
                w: {
                    k: v
                    for k, v in strip(
                        harness.metrics(
                            log_softmax(test_scores[name]) + w * mlp, truth, accept, groups
                        )
                    ).items()
                    if k in ("top1", "top5", "mrr", "macro_f1")
                }
                for w in trials
            }
        entry["combination"]["every_weight_test_diagnostic_not_used_for_choice"] = diag
        report["models"][key] = entry
        print(key, "done", flush=True)
    with open(harness.STUDY_DIR / "final_eval.json", "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
