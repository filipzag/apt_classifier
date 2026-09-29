"""Compatibility shim: the case store lives in apt_classifier.cases."""

from apt_classifier.cases import CASE_SCHEMA_VERSION, CaseRecord, CaseStore, safe_case_id

__all__ = ["CASE_SCHEMA_VERSION", "CaseRecord", "CaseStore", "safe_case_id"]
