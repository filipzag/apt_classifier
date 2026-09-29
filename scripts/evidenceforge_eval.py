#!/usr/bin/env python3
"""Leak-resistant EvidenceForge technique-mapping evaluator.

Labels are read only for scoring.  Model inputs are either a rendering of the
GROUND_TRUTH event's observable ``attributes`` mapping or records selected from
one native source by exact observable matches.  Activity/scenario descriptions,
ATT&CK annotations, explanations, and other label-bearing prose are never used
as model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings, get_settings  # noqa: E402
from apt_classifier.embeddings import SentenceTransformerEmbedder  # noqa: E402
from apt_classifier.evaluation import (  # noqa: E402
    TechniqueEvaluationCase,
    TechniquePrediction,
    bootstrap_technique_metrics,
    evaluate_technique_predictions,
    pipeline_fingerprint,
    write_artifact,
)
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance  # noqa: E402
from apt_classifier.llm import OpenRouterClient  # noqa: E402
from apt_classifier.presets import PRESETS  # noqa: E402
from apt_classifier.session import AttributionEngine  # noqa: E402
from apt_classifier.vectorstore import ChromaVectorStore  # noqa: E402

_LABEL_KEYS = frozenset(
    {
        "activity",
        "attack",
        "attack_id",
        "description",
        "explanation",
        "ground_truth",
        "label",
        "labels",
        "scenario_description",
        "tactic",
        "technique",
        "technique_id",
    }
)
_LABEL_KEY_MARKERS = tuple(re.sub(r"[^a-z]", "", key) for key in _LABEL_KEYS)
_SCENARIO_NON_OBSERVABLE = _LABEL_KEYS | {
    "type",
    "start_time",
    "supplementary",
    "ids_alerts",
    "process_ref",
    "parent_ref",
}


@dataclass(frozen=True, slots=True)
class ScenarioIntent:
    record_id: str
    storyline_id: str
    event_type: str
    technique_id: str
    observable_attributes: dict[str, Any]
    section: str


@dataclass(slots=True)
class CaseBuildResult:
    cases: list[TechniqueEvaluationCase]
    total_events: int
    scorable_events: int
    unsupported: list[dict[str, str]]

    @property
    def coverage(self) -> float:
        return self.scorable_events / self.total_events if self.total_events else 0.0

    def diagnostics(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for item in self.unsupported:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        return {
            "total_events": self.total_events,
            "scorable_events": self.scorable_events,
            "coverage": self.coverage,
            "unsupported_count": len(self.unsupported),
            "unsupported_reasons": reasons,
            "unsupported": self.unsupported,
        }


def _parse_technique_id(value: Any) -> str:
    text = str(value or "").strip()
    return text.split(" - ", 1)[0].upper() if text else ""


def _scenario_root(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("scenario")
    return nested if isinstance(nested, dict) else value


def build_technique_mapping(scenario_path: Path) -> list[ScenarioIntent]:
    """Create exact event intents; no event-type-only fallback is permitted."""
    value = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    scenario = _scenario_root(value)
    intents: list[ScenarioIntent] = []
    for section_name, output_section in (
        ("storyline", "storyline"),
        ("red_herrings", "red_herring"),
    ):
        for group in scenario.get(section_name, ()) or ():
            raw_id = str(group.get("id", ""))
            storyline_id = raw_id if output_section == "storyline" else f"red_herring:{raw_id}"
            for index, event in enumerate(group.get("events", ()) or ()):
                if not isinstance(event, dict):
                    continue
                attributes = {
                    str(key): value
                    for key, value in event.items()
                    if key not in _SCENARIO_NON_OBSERVABLE and value is not None
                }
                intents.append(
                    ScenarioIntent(
                        record_id=f"{storyline_id}#{index}",
                        storyline_id=storyline_id,
                        event_type=str(event.get("type", "")),
                        technique_id=_parse_technique_id(event.get("technique")),
                        observable_attributes=attributes,
                        section=output_section,
                    )
                )
    return intents


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).casefold()


def align_event_to_intent(
    event: dict[str, Any], intents: Sequence[ScenarioIntent]
) -> tuple[ScenarioIntent | None, str | None]:
    """Align by exact record identity/type and verify every shared authored attribute."""
    record_id = str(event.get("record_id", ""))
    candidates = [intent for intent in intents if intent.record_id == record_id]
    if len(candidates) != 1:
        return None, "missing_or_ambiguous_intent"
    intent = candidates[0]
    if str(event.get("storyline_id", "")) != intent.storyline_id:
        return None, "storyline_mismatch"
    if str(event.get("kind", "")) != intent.event_type:
        return None, "event_type_mismatch"
    attributes = event.get("attributes")
    if not isinstance(attributes, dict):
        return None, "missing_observable_attributes"
    shared = set(attributes) & set(intent.observable_attributes)
    if any(
        _canonical(attributes[key]) != _canonical(intent.observable_attributes[key])
        for key in shared
    ):
        return None, "observable_attribute_mismatch"
    return intent, None


def _is_label_key(value: Any) -> bool:
    normalized = re.sub(r"[^a-z]", "", str(value).casefold())
    return any(marker in normalized for marker in _LABEL_KEY_MARKERS)


def _safe_attributes(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if not _is_label_key(key) and item not in (None, "", [], {})
    }


def _render_value(value: Any) -> str:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def build_narrative_text(event: dict[str, Any]) -> str:
    """Render only observable attributes, never top-level GROUND_TRUTH prose."""
    attributes = _safe_attributes(event.get("attributes"))
    return ". ".join(
        f"{key}: {_render_value(attributes[key])}" for key in sorted(attributes, key=str.casefold)
    )


def _visibility_for_event(
    event: dict[str, Any], intent_status: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return dict(intent_status.get(str(event.get("intent_id", "")), {}))


def _case_metadata(
    event: dict[str, Any], intent: ScenarioIntent, visibility: dict[str, Any]
) -> dict[str, Any]:
    return {
        "intent_id": str(event.get("intent_id", "")),
        "event_type": intent.event_type,
        "source_visibility": visibility,
        "label_scope": "non_exhaustive_intent" if intent.technique_id else "zero_label_red_herring",
        "alignment": "exact_record_identity_and_observable_attributes",
    }


def build_narrative_case_set(
    events: Sequence[dict[str, Any]],
    intents: Sequence[ScenarioIntent],
    intent_status: dict[str, dict[str, Any]] | None = None,
) -> CaseBuildResult:
    status = intent_status or {}
    cases: list[TechniqueEvaluationCase] = []
    unsupported: list[dict[str, str]] = []
    for event in events:
        intent, reason = align_event_to_intent(event, intents)
        if intent is None:
            unsupported.append(
                {"record_id": str(event.get("record_id", "")), "reason": str(reason)}
            )
            continue
        text = build_narrative_text(event)
        if not text:
            unsupported.append(
                {"record_id": intent.record_id, "reason": "no_observable_attributes"}
            )
            continue
        cases.append(
            TechniqueEvaluationCase(
                case_id=f"narrative:{intent.record_id}",
                document_id=intent.storyline_id,
                text=text,
                expected_technique_ids=(intent.technique_id,) if intent.technique_id else (),
                source="evidenceforge_narrative",
                label_quality="intent_non_exhaustive" if intent.technique_id else "red_herring",
                metadata=_case_metadata(event, intent, _visibility_for_event(event, status)),
            )
        )
    reject_conflicting_labels(cases)
    return CaseBuildResult(cases, len(events), len(cases), unsupported)


def build_narrative_cases(
    events: Sequence[dict[str, Any]], intents: Sequence[ScenarioIntent]
) -> list[TechniqueEvaluationCase]:
    """Compatibility wrapper returning narrative cases only."""
    return build_narrative_case_set(events, intents).cases


def load_ground_truth(gt_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    value = json.loads(gt_path.read_text(encoding="utf-8"))
    statuses = {
        str(item.get("intent_id", "")): dict(item.get("source_status") or {})
        for item in value.get("intent_reconciliation", {}).get("intents", ())
        if isinstance(item, dict)
    }
    return list(value.get("events", ())), statuses


def load_ground_truth_events(gt_path: Path) -> list[dict[str, Any]]:
    return load_ground_truth(gt_path)[0]


def _parse_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _parse_xml_events(xml_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, event_element in ET.iterparse(xml_path, events=("end",)):
        if event_element.tag.rsplit("}", 1)[-1] != "Event":
            continue
        record: dict[str, Any] = {}
        system = event_element.find("{*}System")
        if system is not None:
            for name in ("EventID", "Channel", "Computer"):
                element = system.find(f"{{*}}{name}")
                if element is not None and element.text:
                    record[name] = int(element.text) if name == "EventID" else element.text
            created = system.find("{*}TimeCreated")
            if created is not None:
                record["TimeCreated"] = created.get("SystemTime", "")
            provider = system.find("{*}Provider")
            if provider is not None:
                record["ProviderName"] = provider.get("Name", "")
        data = event_element.find("{*}EventData")
        if data is not None:
            for element in data.findall("{*}Data"):
                if name := element.get("Name"):
                    record[name] = element.text or ""
        records.append(record)
        event_element.clear()
    return records


def _parse_ecar_records(ecar_path: Path) -> list[dict[str, Any]]:
    records = []
    for line in ecar_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


_ECAR_EVENT_TYPE_MAP: dict[str, set[tuple[str, str]]] = {
    "process": {("PROCESS", "CREATE")},
    "process_access": {("PROCESS", "OPEN")},
    "create_remote_thread": {("THREAD", "REMOTE_CREATE")},
    "service_installed": {("SERVICE", "CREATE")},
    "credential_spray": {("USER_SESSION", "LOGIN")},
    "ssh_session": {("USER_SESSION", "LOGIN")},
    "rdp_session": {("USER_SESSION", "LOGIN")},
    "web_scan": {("FLOW", "CONNECT")},
    "port_scan": {("FLOW", "CONNECT")},
    "beacon": {("FLOW", "CONNECT")},
    "failed_logon": {("USER_SESSION", "LOGIN")},
    "logon": {("USER_SESSION", "LOGIN")},
    "connection": {("FLOW", "CONNECT")},
}
_WINDOWS_EVENT_TYPE_MAP: dict[str, set[int]] = {
    "process": {1, 4688},
    "process_access": {10},
    "create_remote_thread": {8},
    "service_installed": {4697},
    "scheduled_task_created": {4698},
    "account_created": {4720},
    "group_member_added": {4728, 4732, 4756},
    "log_cleared": {1102},
    "credential_spray": {4625, 4776},
    "dns_query": {22},
    "dns_tunnel": {22},
    "ssh_session": {4624, 4625},
    "rdp_session": {4624, 4625},
    "web_scan": {3, 5156},
    "port_scan": {3, 5156},
    "beacon": {3, 5156},
    "failed_logon": {4625},
    "logon": {4624, 4625},
    "connection": {3, 5156},
}


def _record_values(record: dict[str, Any]) -> dict[str, Any]:
    output = {re.sub(r"[^a-z0-9]", "", str(key).casefold()): value for key, value in record.items()}
    properties = record.get("properties")
    if isinstance(properties, dict):
        output.update(
            {
                re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
                for key, value in properties.items()
            }
        )
    return output


_ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "command_line": ("commandline", "cmdline", "processcommandline"),
    "process_name": ("image", "imagepath", "newprocessname", "processname"),
    "dst_ip": ("dstip", "destinationip", "destinationaddress"),
    "dst_port": ("dstport", "destinationport"),
    "source_ip": ("sourceip", "srcip", "ipaddress"),
    "src_ip": ("sourceip", "srcip", "ipaddress"),
    "action": ("action", "outcome"),
    "logon_type": ("logontype",),
    "target_username": ("targetusername", "samaccountname"),
    "service_name": ("servicename",),
    "service_file_name": ("servicefilename", "imagepath"),
    "task_name": ("taskname",),
    "group_name": ("targetusername", "groupname"),
    "member_name": ("membername",),
    "target_process": ("targetimage", "targetimagepath"),
    "access_mask": ("grantedaccess",),
    "base_domain": ("queryname", "query"),
}
# Event types whose native records legitimately carry no event-specific
# observable (e.g. 1102 log cleared).  They are scorable only when exactly one
# record of the allowed type exists on the host inside the window, so the
# alignment is still an identity match rather than a broad time-window guess.
_IDENTITY_ONLY_EVENT_TYPES = frozenset({"log_cleared"})


def _value_matches(expected: Any, actual: Any, key: str) -> bool:
    expected_text = _canonical(expected)
    actual_text = _canonical(actual)
    if key in {"process_name", "target_process", "service_file_name"}:
        return (
            expected_text == actual_text
            or Path(expected_text.replace("\\", "/")).name
            == Path(actual_text.replace("\\", "/")).name
        )
    if key == "member_name":
        # 4728/4732/4756 carry the member as a distinguished name.
        first = actual_text.split(",", 1)[0]
        return expected_text == actual_text or first == f"cn={expected_text}"
    if key == "base_domain":
        return actual_text == expected_text or actual_text.endswith(f".{expected_text}")
    try:
        return float(expected) == float(actual)
    except (TypeError, ValueError):
        return expected_text == actual_text


def _record_matches_observables(record: dict[str, Any], event: dict[str, Any]) -> bool:
    """Require at least one exact, event-specific observable match."""
    values = _record_values(record)
    attributes = _safe_attributes(event.get("attributes"))
    matched = 0
    for key, aliases in _ATTRIBUTE_ALIASES.items():
        if key not in attributes:
            continue
        present = [
            values[alias]
            for alias in aliases
            if alias in values and values[alias] not in (None, "")
        ]
        if present:
            if not any(_value_matches(attributes[key], actual, key) for actual in present):
                return False
            matched += 1
    accounts = attributes.get("target_accounts")
    if isinstance(accounts, list | tuple | set):
        present_accounts = [values[key] for key in ("targetusername",) if key in values]
        if present_accounts:
            if not any(
                any(_value_matches(account, actual, "target_username") for account in accounts)
                for actual in present_accounts
            ):
                return False
            matched += 1
    ports = attributes.get("ports")
    if isinstance(ports, list | tuple | set):
        present_ports = [values[key] for key in ("dstport", "destinationport") if key in values]
        if present_ports:
            if not any(
                any(_value_matches(port, actual, "dst_port") for port in ports)
                for actual in present_ports
            ):
                return False
            matched += 1
    return matched > 0


def _record_time(record: dict[str, Any], source: str) -> datetime | None:
    if source == "ecar":
        timestamp = record.get("timestamp_ms")
        try:
            return datetime.fromtimestamp(float(timestamp) / 1000.0, tz=UTC)
        except (TypeError, ValueError, OSError):
            return None
    return _parse_time(record.get("TimeCreated"))


def _record_host(record: dict[str, Any], source: str) -> str:
    return str(record.get("hostname" if source == "ecar" else "Computer", "")).split(".", 1)[0]


def _record_type_matches(record: dict[str, Any], event_type: str, source: str) -> bool:
    if source == "ecar":
        allowed = _ECAR_EVENT_TYPE_MAP.get(event_type)
        return bool(
            allowed and (str(record.get("object", "")), str(record.get("action", ""))) in allowed
        )
    allowed_ids = _WINDOWS_EVENT_TYPE_MAP.get(event_type)
    try:
        return bool(allowed_ids and int(record.get("EventID")) in allowed_ids)
    except (TypeError, ValueError):
        return False


def _select_records_for_event(
    event: dict[str, Any], records: Sequence[dict[str, Any]], source: str, window_seconds: int
) -> list[dict[str, Any]]:
    event_time = _parse_time(event.get("time"))
    event_host = str(event.get("system", "")).split(".", 1)[0].casefold()
    if event_time is None or not event_host:
        return []
    selected = []
    identity_candidates: list[dict[str, Any]] = []
    for record in records:
        record_time = _record_time(record, source)
        if record_time is None or abs((record_time - event_time).total_seconds()) > window_seconds:
            continue
        if _record_host(record, source).casefold() != event_host:
            continue
        if not _record_type_matches(record, str(event.get("kind", "")), source):
            continue
        if _record_matches_observables(record, event):
            selected.append(record)
        elif str(event.get("kind", "")) in _IDENTITY_ONLY_EVENT_TYPES:
            identity_candidates.append(record)
    if not selected and len(identity_candidates) == 1:
        return identity_candidates
    return selected


def _narrate_xml_record(record: dict[str, Any]) -> str:
    from apt_classifier.evidence.winlogs import narrate_event

    narration = narrate_event(record)
    if narration:
        return narration
    fields = _record_values(record)
    return ". ".join(
        f"{key}: {_render_value(value)}"
        for key, value in sorted(fields.items())
        if value not in (None, "") and not _is_label_key(key)
    )


def _narrate_ecar_record(record: dict[str, Any]) -> str:
    from apt_classifier.evidence.ecar import narrate_ecar_record

    narration = narrate_ecar_record(record)
    if narration and not narration.rstrip(".").endswith("recorded an event"):
        return narration
    fields = _record_values(record)
    return ". ".join(
        f"{key}: {_render_value(value)}"
        for key, value in sorted(fields.items())
        if value not in (None, "") and not _is_label_key(key)
    )


def build_source_case_set(
    events: Sequence[dict[str, Any]],
    intents: Sequence[ScenarioIntent],
    records: Sequence[dict[str, Any]],
    *,
    source: str,
    intent_status: dict[str, dict[str, Any]] | None = None,
    window_seconds: int = 120,
) -> CaseBuildResult:
    if source not in {"windows", "ecar"}:
        raise ValueError(f"unsupported source: {source}")
    status = intent_status or {}
    cases: list[TechniqueEvaluationCase] = []
    unsupported: list[dict[str, str]] = []
    visibility_name = "ecar" if source == "ecar" else "windows"
    for event in events:
        intent, reason = align_event_to_intent(event, intents)
        record_id = str(event.get("record_id", ""))
        if intent is None:
            unsupported.append({"record_id": record_id, "reason": str(reason)})
            continue
        visibility = _visibility_for_event(event, status)
        source_visibility = visibility.get(visibility_name)
        if source == "windows" and source_visibility is None:
            source_visibility = visibility.get("sysmon") or visibility.get("windows_security")
        if source_visibility and not source_visibility.get("visible"):
            unsupported.append({"record_id": record_id, "reason": "source_not_visible"})
            continue
        selected = _select_records_for_event(event, records, source, window_seconds)
        if not selected:
            reason = (
                "unsupported_event_type"
                if not (
                    _ECAR_EVENT_TYPE_MAP.get(intent.event_type)
                    if source == "ecar"
                    else _WINDOWS_EVENT_TYPE_MAP.get(intent.event_type)
                )
                else "no_exact_observable_record_match"
            )
            unsupported.append({"record_id": record_id, "reason": reason})
            continue
        narrate = _narrate_ecar_record if source == "ecar" else _narrate_xml_record
        text = ". ".join(filter(None, (narrate(record) for record in selected[:10])))
        if not text:
            unsupported.append({"record_id": record_id, "reason": "record_not_narratable"})
            continue
        metadata = _case_metadata(event, intent, visibility)
        metadata.update(
            {
                "selected_record_count": len(selected),
                "native_source": source,
                "alignment": "host_time_type_and_exact_observable_attributes",
            }
        )
        cases.append(
            TechniqueEvaluationCase(
                case_id=f"{source}:{record_id}",
                document_id=intent.storyline_id,
                text=text,
                expected_technique_ids=(intent.technique_id,) if intent.technique_id else (),
                source=f"evidenceforge_{source}",
                label_quality="intent_non_exhaustive" if intent.technique_id else "red_herring",
                metadata=metadata,
            )
        )
    reject_conflicting_labels(cases)
    return CaseBuildResult(cases, len(events), len(cases), unsupported)


def _load_native_records(bundle_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    ecar: list[dict[str, Any]] = []
    data_dir = bundle_dir / "data"
    if not data_dir.is_dir():
        return windows, ecar
    for host_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for path in sorted(host_dir.glob("windows_event_*.xml")):
            windows.extend(_parse_xml_events(path))
        path = host_dir / "ecar.json"
        if path.exists():
            ecar.extend(_parse_ecar_records(path))
    return windows, ecar


def build_rawlog_cases(
    events: Sequence[dict[str, Any]],
    intents: Sequence[ScenarioIntent],
    evidenceforge_dir: Path,
    window_seconds: int = 120,
) -> list[TechniqueEvaluationCase]:
    """Compatibility wrapper; sources remain distinguishable in IDs/source fields."""
    windows, ecar = _load_native_records(evidenceforge_dir)
    return (
        build_source_case_set(
            events, intents, windows, source="windows", window_seconds=window_seconds
        ).cases
        + build_source_case_set(
            events, intents, ecar, source="ecar", window_seconds=window_seconds
        ).cases
    )


def reject_conflicting_labels(cases: Sequence[TechniqueEvaluationCase]) -> None:
    """Reject identical normalized model input carrying different labels."""
    observed: dict[str, tuple[str, ...]] = {}
    for case in cases:
        normalized = re.sub(r"[^a-z0-9]+", " ", case.text.casefold()).strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        labels = tuple(sorted(case.expected_technique_ids))
        previous = observed.setdefault(digest, labels)
        if previous != labels:
            raise ValueError(
                f"identical model input has conflicting labels: {previous!r} and {labels!r}"
            )


def run_cases(
    cases: Sequence[TechniqueEvaluationCase],
    *,
    engine: AttributionEngine,
    verify: bool,
) -> list[TechniquePrediction]:
    """Use the application's shared mapper; local explicitly disables verification."""
    predictions: list[TechniquePrediction] = []
    for case in cases:
        ledger = EvidenceLedger(
            observations=[
                Observation(
                    text=case.text,
                    provenance=Provenance(source=case.source, source_name=case.document_id),
                )
            ]
        )
        retrieval, _ = engine.map_techniques(
            ledger,
            verify=verify,
            mode="production" if verify else "local",
        )
        ranked = tuple(
            dict.fromkeys(
                candidate.technique_id
                for candidate in sorted(retrieval.candidates, key=lambda item: item.fused_rank)
            )
        )
        predictions.append(
            TechniquePrediction.from_ranked(
                case.case_id, ranked, accepted=tuple(retrieval.observed_techniques)
            )
        )
    return predictions


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _discover_git_commit(path: Path) -> str | None:
    for parent in (path, *path.parents):
        git = parent / ".git"
        head = git / "HEAD"
        if not head.is_file():
            continue
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            reference = git / value[5:]
            return reference.read_text(encoding="utf-8").strip() if reference.is_file() else None
        return value or None
    return None


def collect_run_metadata(
    *,
    bundle_dir: Path,
    scenario_path: Path,
    settings: Settings,
    kb: AttackKnowledgeBase,
    preset: str,
    pipeline: str,
) -> dict[str, Any]:
    generation_path = bundle_dir / "GENERATION_MANIFEST.json"
    generation = _read_json(generation_path)
    manifest_names = (
        "GENERATION_MANIFEST.json",
        "OBSERVATION_MANIFEST.json",
        "STORAGE_MANIFEST.json",
        "COLLECTION_PROFILE.json",
    )
    manifests = {
        name: digest
        for name in manifest_names
        if (digest := _sha256(bundle_dir / name)) is not None
    }
    model_metadata = {
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "verifier_model": settings.technique_verifier_model,
        "reranker_model": settings.reranker_model,
        "reranker_revision": settings.reranker_model_revision,
    }
    pipeline_metadata = {
        "pipeline": pipeline,
        "verify": pipeline == "production",
        "preset": preset,
        "retrieval_mode": settings.technique_retrieval_mode,
        "candidate_pool": settings.technique_candidate_pool,
        "retrieval_top_k": settings.retrieval_top_k,
        "similarity_threshold": settings.similarity_threshold,
        "acceptance_threshold": settings.technique_acceptance_threshold,
        "relative_margin": settings.technique_relative_margin,
    }
    return {
        "scenario_path": str(scenario_path),
        "scenario_hash": _sha256(scenario_path),
        "ground_truth_hash": _sha256(bundle_dir / "GROUND_TRUTH.json"),
        "bundle_manifest_hashes": manifests,
        "bundle_compiled_hash": generation.get("compiled_sha256"),
        "generation_seed": generation.get("generation_seed", generation.get("seed")),
        "evidenceforge_version": generation.get("evidenceforge_version"),
        "evidenceforge_commit": generation.get("evidenceforge_commit")
        or generation.get("commit")
        or _discover_git_commit(scenario_path.parent),
        "attack_version": getattr(kb, "version", settings.attack_version),
        "models": model_metadata,
        "pipeline": pipeline_metadata,
        "pipeline_fingerprint": pipeline_fingerprint(
            {
                "attack_version": getattr(kb, "version", settings.attack_version),
                "models": model_metadata,
                "pipeline": pipeline_metadata,
            }
        ),
        "label_semantics": "EvidenceForge technique annotations are non-exhaustive intent labels",
    }


def _case_payload(case: TechniqueEvaluationCase, prediction: TechniquePrediction) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "document_id": case.document_id,
        "source": case.source,
        "expected_technique_ids": list(case.expected_technique_ids),
        "ranked_technique_ids": list(prediction.ranked_technique_ids),
        "accepted_technique_ids": list(prediction.accepted_technique_ids),
        "label_quality": case.label_quality,
        "metadata": case.metadata,
        "model_input_hash": hashlib.sha256(case.text.encode()).hexdigest(),
    }


def evaluate_case_set(
    result: CaseBuildResult,
    *,
    engine: AttributionEngine,
    verify: bool,
    kb: AttackKnowledgeBase,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    predictions = run_cases(result.cases, engine=engine, verify=verify)
    metrics = evaluate_technique_predictions(result.cases, predictions, kb=kb)
    intervals = bootstrap_technique_metrics(
        result.cases,
        predictions,
        kb=kb,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    by_case = {prediction.case_id: prediction for prediction in predictions}
    return {
        "metrics": metrics,
        "bootstrap_confidence_intervals": intervals,
        "coverage": result.diagnostics(),
        "cases": [_case_payload(case, by_case[case.case_id]) for case in result.cases],
    }


def _settings_for_preset(settings: Settings, preset: str) -> Settings:
    if preset not in PRESETS:
        raise ValueError(f"unknown mapping preset: {preset}")
    return settings.model_copy(update={"technique_mapping_preset": preset})


def _build_engine(
    settings: Settings,
    kb: AttackKnowledgeBase,
    store: ChromaVectorStore,
    *,
    production: bool,
) -> tuple[AttributionEngine, OpenRouterClient | None]:
    client = OpenRouterClient(settings) if production else None
    return AttributionEngine(settings, kb, store, client), client  # type: ignore[arg-type]


def _output_path(args: argparse.Namespace, pipeline: str) -> Path:
    return args.output_production if pipeline == "production" else args.output_local


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidenceforge-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument(
        "--kb-path", type=Path, default=PROJECT_ROOT / "data" / "kb" / "attack_kb.json"
    )
    parser.add_argument("--pipeline", choices=("local", "production", "both"), default="local")
    parser.add_argument("--mode", choices=("narrative", "windows", "ecar", "all"), default="all")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="balanced")
    parser.add_argument("--window-seconds", type=int, default=120)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument(
        "--output-local",
        type=Path,
        default=PROJECT_ROOT / "data" / "audits" / "evidenceforge_local.json",
    )
    parser.add_argument(
        "--output-production",
        type=Path,
        default=PROJECT_ROOT / "data" / "audits" / "evidenceforge_production.json",
    )
    args = parser.parse_args(argv)
    if args.output_local.resolve() == args.output_production.resolve():
        parser.error("local and production output paths must be distinct")

    gt_path = args.evidenceforge_dir / "GROUND_TRUTH.json"
    if not gt_path.is_file():
        parser.error(f"GROUND_TRUTH.json not found: {gt_path}")
    events, intent_status = load_ground_truth(gt_path)
    intents = build_technique_mapping(args.scenario)
    narrative = build_narrative_case_set(events, intents, intent_status)
    windows_records, ecar_records = _load_native_records(args.evidenceforge_dir)
    case_sets = {
        "narrative": narrative,
        "windows": build_source_case_set(
            events,
            intents,
            windows_records,
            source="windows",
            intent_status=intent_status,
            window_seconds=args.window_seconds,
        ),
        "ecar": build_source_case_set(
            events,
            intents,
            ecar_records,
            source="ecar",
            intent_status=intent_status,
            window_seconds=args.window_seconds,
        ),
    }
    selected_modes = tuple(case_sets) if args.mode == "all" else (args.mode,)
    settings = _settings_for_preset(get_settings(), args.preset)
    kb = AttackKnowledgeBase.load(str(args.kb_path))
    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        revision=settings.embedding_model_revision,
        cache_dir=str(settings.model_cache_dir),
    )
    store = ChromaVectorStore(settings=settings, embedder=embedder)
    pipelines = ("local", "production") if args.pipeline == "both" else (args.pipeline,)
    for pipeline in pipelines:
        engine, client = _build_engine(settings, kb, store, production=pipeline == "production")
        try:
            report = {
                "metadata": collect_run_metadata(
                    bundle_dir=args.evidenceforge_dir,
                    scenario_path=args.scenario,
                    settings=engine.settings,
                    kb=kb,
                    preset=args.preset,
                    pipeline=pipeline,
                ),
                "mode": args.mode,
                "results": {
                    mode: evaluate_case_set(
                        case_sets[mode],
                        engine=engine,
                        verify=pipeline == "production",
                        kb=kb,
                        bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=args.bootstrap_seed,
                    )
                    for mode in selected_modes
                },
            }
            destination = write_artifact(_output_path(args, pipeline), report)
            print(f"{pipeline}: wrote {destination}")
        finally:
            if client is not None:
                client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
