"""Smoke baselines for the harness: group centroids and sentence kNN voting (dev only)."""

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from harness import dev_eval  # noqa: E402


def centroid(prior=0.0):
    def fit(train):
        Y = train.label_matrix()
        C = Y.T @ train.X
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9

        def predict(docs):
            out = []
            for D in docs:
                m = D.mean(0)
                m /= np.linalg.norm(m) + 1e-9
                out.append(C @ m * 20 + prior * train.log_prior)
            return np.stack(out)

        return predict

    return fit


def knn(k=10, prior=0.0):
    def fit(train):
        Y = train.label_matrix()

        def predict(docs):
            out = []
            for D in docs:
                S = D @ train.X.T
                idx = np.argpartition(-S, k, axis=1)[:, :k]
                votes = np.zeros(len(train.groups))
                for row, cols in zip(S, idx, strict=False):
                    w = np.exp((row[cols] - 1) * 10)
                    votes += (w[:, None] * Y[cols]).sum(0)
                out.append(np.log(votes + 1e-6) + prior * train.log_prior)
            return np.stack(out)

        return predict

    return fit


if __name__ == "__main__":
    for name, fit in (
        ("centroid", centroid()),
        ("centroid+prior1", centroid(1.0)),
        ("knn10", knn(10)),
        ("knn10+prior1", knn(10, 1.0)),
    ):
        m = dev_eval(fit)
        m.pop("ranks")
        print(name, json.dumps(m))
