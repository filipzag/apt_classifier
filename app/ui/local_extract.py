"""Compatibility shim: local extraction lives in apt_classifier.evidence.local_extract."""

from apt_classifier.evidence.local_extract import MIN_WORDS, local_observations

__all__ = ["MIN_WORDS", "local_observations"]
