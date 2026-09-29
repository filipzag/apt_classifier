import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ExternalReference = dict[str, str]


@dataclass(slots=True)
class Technique:
    id: str
    name: str
    stix_id: str = ""
    description: str = ""
    detection: str = ""
    platforms: list[str] = field(default_factory=list)
    tactics: list[str] = field(default_factory=list)
    is_subtechnique: bool = False
    parent_id: str | None = None
    external_references: list[ExternalReference] = field(default_factory=list)
    sentences: dict[str, list[str]] = field(default_factory=dict)

    @property
    def sentence_count(self) -> int:
        return sum(len(values) for values in self.sentences.values())


@dataclass(slots=True)
class Group:
    id: str
    name: str
    stix_id: str = ""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    external_references: list[ExternalReference] = field(default_factory=list)
    technique_ids: list[str] = field(default_factory=list)
    software_ids: list[str] = field(default_factory=list)
    campaign_ids: list[str] = field(default_factory=list)
    technique_behaviors: dict[str, list[str]] = field(default_factory=dict)
    prominence: int = 0


@dataclass(slots=True)
class Software:
    id: str
    name: str
    stix_id: str = ""
    software_type: str = ""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    external_references: list[ExternalReference] = field(default_factory=list)
    technique_ids: list[str] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Campaign:
    id: str
    name: str
    stix_id: str = ""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    first_seen: str | None = None
    last_seen: str | None = None
    external_references: list[ExternalReference] = field(default_factory=list)
    technique_ids: list[str] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LogSource:
    """A concrete log source channel referenced by a data component or analytic."""

    name: str = ""
    channel: str = ""


@dataclass(slots=True)
class DataComponent:
    id: str
    name: str
    stix_id: str = ""
    description: str = ""
    data_source_id: str = ""
    log_sources: list[LogSource] = field(default_factory=list)


@dataclass(slots=True)
class DataSource:
    id: str
    name: str
    stix_id: str = ""
    description: str = ""
    platforms: list[str] = field(default_factory=list)
    collection_layers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Analytic:
    id: str
    name: str
    stix_id: str = ""
    description: str = ""
    platforms: list[str] = field(default_factory=list)
    log_source_refs: list[LogSource] = field(default_factory=list)
    data_component_ids: list[str] = field(default_factory=list)
    mutable_elements: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class DetectionStrategy:
    id: str
    name: str
    stix_id: str = ""
    analytic_ids: list[str] = field(default_factory=list)
    technique_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AttackKnowledgeBase:
    version: str = "unknown"
    techniques: dict[str, Technique] = field(default_factory=dict)
    groups: dict[str, Group] = field(default_factory=dict)
    software: dict[str, Software] = field(default_factory=dict)
    campaigns: dict[str, Campaign] = field(default_factory=dict)
    detection_strategies: dict[str, DetectionStrategy] = field(default_factory=dict)
    analytics: dict[str, Analytic] = field(default_factory=dict)
    data_components: dict[str, DataComponent] = field(default_factory=dict)
    data_sources: dict[str, DataSource] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    prominence: dict[str, int] = field(default_factory=dict)

    def recompute_statistics(self) -> None:
        group_count = len(self.groups)
        document_frequency = {
            technique_id: sum(technique_id in group.technique_ids for group in self.groups.values())
            for technique_id in self.techniques
        }
        self.idf = {
            technique_id: math.log((group_count + 1) / (frequency + 1)) + 1.0
            for technique_id, frequency in document_frequency.items()
        }
        self.priors = (
            {group_id: 1.0 / group_count for group_id in self.groups} if group_count else {}
        )
        self.prominence = {}
        for group_id, group in self.groups.items():
            group.prominence = (
                len(group.external_references)
                + len(group.aliases)
                + len(group.campaign_ids)
                + len(group.software_ids)
            )
            self.prominence[group_id] = group.prominence

    @property
    def sentence_count(self) -> int:
        return sum(technique.sentence_count for technique in self.techniques.values())

    @property
    def attack_version(self) -> str:
        return self.version

    @property
    def technique_idf(self) -> dict[str, float]:
        return self.idf

    @property
    def group_priors(self) -> dict[str, float]:
        return self.priors

    @property
    def group_prominence(self) -> dict[str, int]:
        return self.prominence

    @property
    def group_techniques(self) -> dict[str, list[str]]:
        return {group_id: list(group.technique_ids) for group_id, group in self.groups.items()}

    @property
    def group_software(self) -> dict[str, list[str]]:
        return {group_id: list(group.software_ids) for group_id, group in self.groups.items()}

    @property
    def campaign_techniques(self) -> dict[str, list[str]]:
        return {
            campaign_id: list(campaign.technique_ids)
            for campaign_id, campaign in self.campaigns.items()
        }

    @property
    def campaign_groups(self) -> dict[str, list[str]]:
        return {
            campaign_id: list(campaign.group_ids)
            for campaign_id, campaign in self.campaigns.items()
        }

    @property
    def software_techniques(self) -> dict[str, list[str]]:
        return {
            software_id: list(item.technique_ids) for software_id, item in self.software.items()
        }

    @property
    def software_groups(self) -> dict[str, list[str]]:
        return {software_id: list(item.group_ids) for software_id, item in self.software.items()}

    @property
    def technique_detection_strategies(self) -> dict[str, list[str]]:
        """Map technique ID → list of detection strategy IDs that detect it."""
        result: dict[str, list[str]] = {tid: [] for tid in self.techniques}
        for ds_id, ds in self.detection_strategies.items():
            for tid in ds.technique_ids:
                result.setdefault(tid, []).append(ds_id)
        return {tid: sorted(ids) for tid, ids in result.items()}

    @property
    def detection_strategy_analytics(self) -> dict[str, list[str]]:
        """Map detection strategy ID → list of analytic IDs."""
        return {ds_id: list(ds.analytic_ids) for ds_id, ds in self.detection_strategies.items()}

    @property
    def analytic_data_components(self) -> dict[str, list[str]]:
        """Map analytic ID → list of data component IDs."""
        return {an_id: list(an.data_component_ids) for an_id, an in self.analytics.items()}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackKnowledgeBase":
        techniques = {
            key: Technique(**({"id": key} | value))
            for key, value in data.get("techniques", {}).items()
        }
        groups = {
            key: Group(**({"id": key} | value)) for key, value in data.get("groups", {}).items()
        }
        software = {
            key: Software(**({"id": key} | value))
            for key, value in data.get("software", {}).items()
        }
        campaigns = {
            key: Campaign(**({"id": key} | value))
            for key, value in data.get("campaigns", {}).items()
        }
        detection_strategies = {
            key: DetectionStrategy(**({"id": key} | value))
            for key, value in data.get("detection_strategies", {}).items()
        }
        analytics = {
            key: _analytic_from_dict(key, value) for key, value in data.get("analytics", {}).items()
        }
        data_components = {
            key: _data_component_from_dict(key, value)
            for key, value in data.get("data_components", {}).items()
        }
        data_sources = {
            key: DataSource(**({"id": key} | value))
            for key, value in data.get("data_sources", {}).items()
        }
        knowledge_base = cls(
            version=str(data.get("version", data.get("attack_version", "unknown"))),
            techniques=techniques,
            groups=groups,
            software=software,
            campaigns=campaigns,
            detection_strategies=detection_strategies,
            analytics=analytics,
            data_components=data_components,
            data_sources=data_sources,
            idf={str(key): float(value) for key, value in data.get("idf", {}).items()},
            priors={str(key): float(value) for key, value in data.get("priors", {}).items()},
            prominence={str(key): int(value) for key, value in data.get("prominence", {}).items()},
        )
        if not all(key in data for key in ("idf", "priors", "prominence")):
            knowledge_base.recompute_statistics()
        return knowledge_base

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "AttackKnowledgeBase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("ATT&CK knowledge base must be a JSON object")
        return cls.from_dict(data)


def save_knowledge_base(knowledge_base: AttackKnowledgeBase, path: str | Path) -> Path:
    return knowledge_base.save(path)


def load_knowledge_base(path: str | Path) -> AttackKnowledgeBase:
    return AttackKnowledgeBase.load(path)


def _log_source_from_dict(value: dict[str, Any]) -> LogSource:
    return LogSource(
        name=str(value.get("name", "")),
        channel=str(value.get("channel", "")),
    )


def _analytic_from_dict(key: str, value: dict[str, Any]) -> Analytic:
    log_source_refs = [
        _log_source_from_dict(item)
        for item in value.get("log_source_refs", [])
        if isinstance(item, dict)
    ]
    return Analytic(
        id=key,
        name=str(value.get("name", "")),
        stix_id=str(value.get("stix_id", "")),
        description=str(value.get("description", "")),
        platforms=list(value.get("platforms", [])),
        log_source_refs=log_source_refs,
        data_component_ids=list(value.get("data_component_ids", [])),
        mutable_elements=list(value.get("mutable_elements", [])),
    )


def _data_component_from_dict(key: str, value: dict[str, Any]) -> DataComponent:
    log_sources = [
        _log_source_from_dict(item)
        for item in value.get("log_sources", [])
        if isinstance(item, dict)
    ]
    return DataComponent(
        id=key,
        name=str(value.get("name", "")),
        stix_id=str(value.get("stix_id", "")),
        description=str(value.get("description", "")),
        data_source_id=str(value.get("data_source_id", "")),
        log_sources=log_sources,
    )
