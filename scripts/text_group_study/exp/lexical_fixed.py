"""lexical BEST with the corpus placeholder token "operation" as a stop word.

build_corpus.py replaced every campaign name with "the operation", so in FINAL mode that token
marks the groups that have ATT&CK campaigns; a report that says "operation" would favour them.
Fixed before any test run (audit finding on dev: 1.4334 -> 1.3149)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lexical as _lexical  # noqa: E402

_lexical.STOP = sorted(set(_lexical.STOP) | {"operation", "operations"})
BEST = dict(_lexical.BEST)


def make_fit(**config):
    return _lexical.make_fit(**config)
