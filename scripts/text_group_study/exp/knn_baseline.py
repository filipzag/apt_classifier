"""The fixed smoke baseline (baselines.knn, k=10), chosen before any test run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the study code folder (harness)
from baselines import knn

BEST = {"k": 10}


def make_fit(**config):
    return knn(**config)
