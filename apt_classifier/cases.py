"""On-disk case store for the UI.

A case is the evidence ledger (observations, analyst technique reviews, interview answers,
strategic context) plus the analyst's name for it, the classifier weights in force and a
small summary for listings. Cases are JSON files under ``<DATA_DIR>/cases`` so they survive
a browser refresh or a restart; the analysis itself is recomputed from the ledger on load,
so nothing derived (rankings, LLM opinions) is trusted from disk.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apt_classifier.evidence.models import EvidenceLedger

CASE_SCHEMA_VERSION = 1
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    name: str
    created_at: str
    updated_at: str
    observations: int
    stage: str | None
    leading_group: str | None
    path: Path

    @property
    def label(self) -> str:
        when = self.updated_at[:16].replace("T", " ")
        return f"{self.name} · {self.observations} obs · {when}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def safe_case_id(case_id: str) -> str:
    value = str(case_id or "").strip()
    if not _CASE_ID.match(value):
        raise ValueError(f"invalid case id {case_id!r}")
    return value


class CaseStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, case_id: str) -> Path:
        return self.directory / f"{safe_case_id(case_id)}.json"

    def list(self) -> list[CaseRecord]:
        if not self.directory.is_dir():
            return []
        records: list[CaseRecord] = []
        for path in self.directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict) or payload.get("kind") != "apt_case":
                continue
            summary = payload.get("summary") or {}
            records.append(
                CaseRecord(
                    case_id=str(payload.get("case_id") or path.stem),
                    name=str(payload.get("name") or path.stem),
                    created_at=str(payload.get("created_at") or ""),
                    updated_at=str(payload.get("updated_at") or ""),
                    observations=int(summary.get("observations") or 0),
                    stage=summary.get("stage"),
                    leading_group=summary.get("leading_group"),
                    path=path,
                )
            )
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records

    def save(
        self,
        *,
        case_id: str,
        name: str,
        ledger: EvidenceLedger,
        weights: dict[str, float] | None = None,
        settings_overrides: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(case_id)
        created_at = _now()
        if path.is_file():
            with contextlib.suppress(OSError, ValueError):
                created_at = str(
                    json.loads(path.read_text(encoding="utf-8")).get("created_at") or created_at
                )
        payload = {
            "kind": "apt_case",
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": safe_case_id(case_id),
            "name": str(name or case_id).strip()[:120] or case_id,
            "created_at": created_at,
            "updated_at": _now(),
            "ledger": ledger.to_dict(),
            "weights": dict(weights or {}),
            "settings_overrides": dict(settings_overrides or {}),
            "summary": dict(summary or {}),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, case_id: str) -> dict[str, Any]:
        payload = json.loads(self._path(case_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("kind") != "apt_case":
            raise ValueError("not a saved case")
        ledger = EvidenceLedger.from_dict(payload["ledger"])
        if not ledger.case_id:
            ledger.case_id = payload["case_id"]
        return {
            "case_id": str(payload["case_id"]),
            "name": str(payload.get("name") or payload["case_id"]),
            "ledger": ledger,
            "weights": dict(payload.get("weights") or {}),
            "settings_overrides": dict(payload.get("settings_overrides") or {}),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def delete(self, case_id: str) -> bool:
        path = self._path(case_id)
        if path.is_file():
            path.unlink()
            return True
        return False
