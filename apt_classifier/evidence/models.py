from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceStatus(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"
    NOT_VISIBLE = "not_visible"


class TechniqueVerificationVerdict(StrEnum):
    NOT_RUN = "not_run"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"
    ERROR = "error"


class TechniqueCandidateState(StrEnum):
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_REVIEW = "needs_review"
    AUTO_REJECTED = "auto_rejected"


class TechniqueReviewAction(StrEnum):
    CONFIRM = "confirm"
    REJECT = "reject"
    REPLACE = "replace"


ObservationStatus = EvidenceStatus
QuestionStatus = EvidenceStatus
AnswerStatus = EvidenceStatus
Status = EvidenceStatus


_STATUS_ALIASES = {
    "yes": EvidenceStatus.POSITIVE,
    "true": EvidenceStatus.POSITIVE,
    "present": EvidenceStatus.POSITIVE,
    "observed": EvidenceStatus.POSITIVE,
    "no": EvidenceStatus.NEGATIVE,
    "false": EvidenceStatus.NEGATIVE,
    "absent": EvidenceStatus.NEGATIVE,
    "not observed": EvidenceStatus.NEGATIVE,
    "not-observed": EvidenceStatus.NEGATIVE,
    "not visible": EvidenceStatus.NOT_VISIBLE,
    "not-visible": EvidenceStatus.NOT_VISIBLE,
    "not_applicable": EvidenceStatus.NOT_VISIBLE,
}


def _status(value: Any) -> EvidenceStatus:
    if isinstance(value, EvidenceStatus):
        return value
    normalized = str(value).strip().lower()
    normalized = _STATUS_ALIASES.get(normalized, normalized.replace("-", "_"))
    return EvidenceStatus(normalized)


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _identifier(prefix: str, *values: object) -> str:
    material = "\x1f".join(str(value).strip().casefold() for value in values)
    return f"{prefix}-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


class EvidenceModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys, ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        return cls.model_validate_json(value)


class Provenance(EvidenceModel):
    source: str = Field(default="unknown", validation_alias=AliasChoices("source", "source_type"))
    source_name: str | None = None
    line_numbers: list[int] = Field(default_factory=list)
    occurrence_count: int = Field(default=1, ge=1)
    raw_excerpt: str | None = None
    raw_excerpts: list[str] = Field(default_factory=list)
    event_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> str:
        return _compact(str(value or "unknown")).lower().replace(" ", "_")

    @field_validator("source_name", mode="before")
    @classmethod
    def normalize_source_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result or None

    @field_validator("line_numbers", mode="before")
    @classmethod
    def normalize_lines(cls, value: Any) -> list[int]:
        if value is None:
            return []
        values = [value] if isinstance(value, int | str) else list(value)
        lines = sorted({int(item) for item in values})
        if any(line < 1 for line in lines):
            raise ValueError("line numbers are one-based")
        return lines

    @field_validator("raw_excerpts", mode="before")
    @classmethod
    def normalize_raw_excerpts(cls, value: Any) -> list[str]:
        if value is None:
            return []
        values = [value] if isinstance(value, str) else list(value)
        return list(dict.fromkeys(str(item) for item in values if str(item)))

    @model_validator(mode="after")
    def include_primary_excerpt(self) -> Self:
        excerpts = list(self.raw_excerpts)
        if self.raw_excerpt and self.raw_excerpt not in excerpts:
            excerpts.insert(0, self.raw_excerpt)
        if not self.raw_excerpt and excerpts:
            object.__setattr__(self, "raw_excerpt", excerpts[0])
        object.__setattr__(self, "raw_excerpts", excerpts)
        return self

    @property
    def source_type(self) -> str:
        return self.source

    def merged(self, other: Provenance) -> Provenance:
        if (self.source, self.source_name, self.event_id) != (
            other.source,
            other.source_name,
            other.event_id,
        ):
            raise ValueError("cannot merge provenance from different sources")
        excerpts = list(dict.fromkeys([*self.raw_excerpts, *other.raw_excerpts]))
        primary = self.raw_excerpt or other.raw_excerpt
        metadata = {**self.metadata, **other.metadata}
        return self.model_copy(
            update={
                "line_numbers": sorted({*self.line_numbers, *other.line_numbers}),
                "occurrence_count": self.occurrence_count + other.occurrence_count,
                "raw_excerpt": primary,
                "raw_excerpts": excerpts,
                "metadata": metadata,
            }
        )


class Observation(EvidenceModel):
    observation_id: str = Field(default="", validation_alias=AliasChoices("observation_id", "id"))
    text: str = Field(validation_alias=AliasChoices("text", "sentence", "action"))
    status: EvidenceStatus = EvidenceStatus.POSITIVE
    provenance: Provenance = Field(default_factory=Provenance)
    kind: str = "action"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        result = _compact(str(value or ""))
        if not result:
            raise ValueError("observation text cannot be empty")
        return result

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: Any) -> EvidenceStatus:
        return _status(value)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: Any) -> str:
        return _compact(str(value or "action")).lower().replace(" ", "_")

    @model_validator(mode="after")
    def assign_id(self) -> Self:
        if not self.observation_id:
            object.__setattr__(
                self,
                "observation_id",
                _identifier(
                    "obs",
                    self.provenance.source,
                    self.provenance.source_name or "",
                    self.kind,
                    self.status.value,
                    self.text,
                ),
            )
        return self

    @property
    def id(self) -> str:
        return self.observation_id

    @property
    def sentence(self) -> str:
        return self.text

    @property
    def action(self) -> str:
        return self.text

    @property
    def occurrence_count(self) -> int:
        return self.provenance.occurrence_count

    @property
    def line_numbers(self) -> list[int]:
        return self.provenance.line_numbers

    @property
    def raw_excerpt(self) -> str | None:
        return self.provenance.raw_excerpt


class TechniqueMatch(EvidenceModel):
    technique_id: str
    technique_name: str | None = None
    score: float = Field(default=0.0, validation_alias=AliasChoices("score", "similarity"))
    observation_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("observation_ids", "observation_id"),
    )
    status: EvidenceStatus = EvidenceStatus.POSITIVE
    parent_technique_id: str | None = Field(
        default=None, validation_alias=AliasChoices("parent_technique_id", "parent_id")
    )
    matched_text: str | None = Field(
        default=None, validation_alias=AliasChoices("matched_text", "sentence", "text")
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("technique_id", "parent_technique_id", mode="before")
    @classmethod
    def normalize_technique_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _compact(str(value)).upper()

    @field_validator("technique_name", "matched_text", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result or None

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("observation_ids", mode="before")
    @classmethod
    def unique_observation_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        values = [value] if isinstance(value, str) else list(value)
        return list(dict.fromkeys(str(item) for item in values if str(item)))

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: Any) -> EvidenceStatus:
        return _status(value)

    @property
    def name(self) -> str | None:
        return self.technique_name

    @property
    def confidence(self) -> float:
        return self.score

    @property
    def similarity(self) -> float:
        return self.score

    @property
    def observation_id(self) -> str | None:
        return self.observation_ids[0] if self.observation_ids else None

    @property
    def parent_id(self) -> str | None:
        return self.parent_technique_id


class TechniqueRetrievalEvidence(EvidenceModel):
    source: str
    document_id: str | None = None
    field: str | None = None
    text: str | None = None
    rationale: str | None = None
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> str:
        return _compact(str(value or "unknown")).lower().replace(" ", "_")

    @field_validator("document_id", "field", "text", "rationale", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result or None

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class TechniqueVerification(EvidenceModel):
    verdict: TechniqueVerificationVerdict = TechniqueVerificationVerdict.NOT_RUN
    reason: str | None = None
    support_span: str | None = None
    model: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @field_validator("reason", "support_span", "model", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result or None


class TechniqueCandidate(EvidenceModel):
    observation_id: str
    technique_id: str
    technique_name: str
    parent_technique_id: str | None = None
    is_subtechnique: bool = False
    fused_score: float = 0.0
    fused_rank: int = Field(ge=1)
    evidence: list[TechniqueRetrievalEvidence] = Field(default_factory=list)
    verification: TechniqueVerification = Field(default_factory=TechniqueVerification)
    state: TechniqueCandidateState = TechniqueCandidateState.NEEDS_REVIEW
    state_reason: str | None = None

    @field_validator("observation_id", "technique_id", "parent_technique_id", mode="before")
    @classmethod
    def normalize_ids(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result.upper() if result.startswith(("t", "T")) else result

    @field_validator("technique_name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> str:
        result = _compact(str(value or ""))
        if not result:
            raise ValueError("technique name cannot be empty")
        return result

    @field_validator("state_reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result or None

    @field_validator("fused_score")
    @classmethod
    def finite_fused_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fused score must be finite")
        return value

    def evidence_for(self, source: str) -> TechniqueRetrievalEvidence | None:
        return next((item for item in self.evidence if item.source == source), None)


class TechniqueReviewDecision(EvidenceModel):
    decision_id: str = Field(default="", validation_alias=AliasChoices("decision_id", "id"))
    observation_id: str
    technique_id: str
    action: TechniqueReviewAction
    replacement_technique_id: str | None = None
    rationale: str | None = None

    @field_validator("observation_id", "technique_id", "replacement_technique_id", mode="before")
    @classmethod
    def normalize_ids(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result.upper() if result.startswith(("t", "T")) else result

    @field_validator("rationale", mode="before")
    @classmethod
    def normalize_rationale(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result or None

    @model_validator(mode="after")
    def validate_replacement_and_id(self) -> Self:
        if self.action is TechniqueReviewAction.REPLACE and not self.replacement_technique_id:
            raise ValueError("replacement technique ID is required")
        if self.action is not TechniqueReviewAction.REPLACE:
            object.__setattr__(self, "replacement_technique_id", None)
        if not self.decision_id:
            object.__setattr__(
                self,
                "decision_id",
                _identifier(
                    "technique-review",
                    self.observation_id,
                    self.technique_id,
                    self.action.value,
                    self.replacement_technique_id or "",
                ),
            )
        return self

    @property
    def id(self) -> str:
        return self.decision_id


class QuestionAnswer(EvidenceModel):
    question_id: str = Field(default="", validation_alias=AliasChoices("question_id", "id"))
    question: str
    status: EvidenceStatus = Field(
        default=EvidenceStatus.UNKNOWN, validation_alias=AliasChoices("status", "answer")
    )
    technique_id: str | None = None
    answer_text: str | None = None
    provenance: Provenance = Field(default_factory=lambda: Provenance(source="question"))
    metadata: dict[str, Any] = Field(default_factory=dict)
    # "software": a question about one malware entry (software_questions.py); technique_id stays
    # None, so the technique answer paths ignore it
    kind: Literal["technique", "software"] = "technique"
    software_id: str | None = None

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: Any) -> str:
        result = _compact(str(value or ""))
        if not result:
            raise ValueError("question cannot be empty")
        return result

    @field_validator("software_id", mode="before")
    @classmethod
    def normalize_software_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value)).upper()
        return result or None

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: Any) -> EvidenceStatus:
        return _status(value)

    @field_validator("technique_id", mode="before")
    @classmethod
    def normalize_technique_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value)).upper()
        return result or None

    @field_validator("answer_text", mode="before")
    @classmethod
    def normalize_answer_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        result = _compact(str(value))
        return result or None

    @model_validator(mode="after")
    def assign_id(self) -> Self:
        if not self.question_id:
            object.__setattr__(
                self,
                "question_id",
                _identifier("question", self.technique_id or self.software_id or "", self.question),
            )
        return self

    @property
    def id(self) -> str:
        return self.question_id

    @property
    def answer(self) -> EvidenceStatus:
        return self.status


class StrategicContext(EvidenceModel):
    target_geographies: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    suspected_origins: list[str] = Field(default_factory=list)

    @field_validator(
        "target_geographies",
        "target_industries",
        "motivations",
        "objectives",
        "suspected_origins",
        mode="before",
    )
    @classmethod
    def normalize_values(cls, value: Any) -> list[str]:
        if value is None:
            return []
        raw = re.split(r"[,;\n]+", value) if isinstance(value, str) else list(value)
        output: list[str] = []
        seen: set[str] = set()
        for item in raw:
            normalized = _compact(str(item or ""))
            folded = normalized.casefold()
            if normalized and folded not in seen:
                seen.add(folded)
                output.append(normalized)
        return output

    @property
    def is_empty(self) -> bool:
        return not any(self.field_values().values())

    def field_values(self) -> dict[str, list[str]]:
        return {
            "target_geographies": self.target_geographies,
            "target_industries": self.target_industries,
            "motivations": self.motivations,
            "objectives": self.objectives,
            "suspected_origins": self.suspected_origins,
        }


class EvidenceLedger(EvidenceModel):
    observations: list[Observation] = Field(default_factory=list)
    technique_matches: list[TechniqueMatch] = Field(
        default_factory=list, validation_alias=AliasChoices("technique_matches", "matches")
    )
    question_answers: list[QuestionAnswer] = Field(
        default_factory=list, validation_alias=AliasChoices("question_answers", "answers")
    )
    technique_reviews: list[TechniqueReviewDecision] = Field(default_factory=list)
    strategic_context: StrategicContext = Field(default_factory=StrategicContext)
    case_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_observation(self, observation: Observation | dict[str, Any]) -> Observation:
        candidate = (
            observation
            if isinstance(observation, Observation)
            else Observation.model_validate(observation)
        )
        for index, current in enumerate(self.observations):
            same_identity = current.observation_id == candidate.observation_id
            same_content = (
                current.text.casefold(),
                current.status,
                current.kind,
                current.provenance.source,
                current.provenance.source_name,
                current.provenance.event_id,
            ) == (
                candidate.text.casefold(),
                candidate.status,
                candidate.kind,
                candidate.provenance.source,
                candidate.provenance.source_name,
                candidate.provenance.event_id,
            )
            if same_identity or same_content:
                merged = current.model_copy(
                    update={
                        "provenance": current.provenance.merged(candidate.provenance),
                        "metadata": {**current.metadata, **candidate.metadata},
                    }
                )
                self.observations[index] = merged
                return merged
        self.observations.append(candidate)
        return candidate

    def extend_observations(
        self, observations: list[Observation] | tuple[Observation, ...]
    ) -> list[Observation]:
        return [self.add_observation(observation) for observation in observations]

    def add_technique_match(self, match: TechniqueMatch | dict[str, Any]) -> TechniqueMatch:
        candidate = (
            match if isinstance(match, TechniqueMatch) else TechniqueMatch.model_validate(match)
        )
        self.technique_matches.append(candidate)
        return candidate

    def add_question_answer(self, answer: QuestionAnswer | dict[str, Any]) -> QuestionAnswer:
        candidate = (
            answer if isinstance(answer, QuestionAnswer) else QuestionAnswer.model_validate(answer)
        )
        self.question_answers.append(candidate)
        return candidate

    def record_technique_review(
        self, decision: TechniqueReviewDecision | dict[str, Any]
    ) -> TechniqueReviewDecision:
        candidate = (
            decision
            if isinstance(decision, TechniqueReviewDecision)
            else TechniqueReviewDecision.model_validate(decision)
        )
        self.technique_reviews = [
            item
            for item in self.technique_reviews
            if (item.observation_id, item.technique_id)
            != (candidate.observation_id, candidate.technique_id)
        ]
        self.technique_reviews.append(candidate)
        return candidate

    @property
    def answers(self) -> list[QuestionAnswer]:
        return self.question_answers

    @property
    def matches(self) -> list[TechniqueMatch]:
        return self.technique_matches

    @property
    def positive_observations(self) -> list[Observation]:
        return [item for item in self.observations if item.status is EvidenceStatus.POSITIVE]

    @property
    def negative_observations(self) -> list[Observation]:
        return [item for item in self.observations if item.status is EvidenceStatus.NEGATIVE]

    def observations_with_status(self, status: EvidenceStatus | str) -> list[Observation]:
        normalized = _status(status)
        return [item for item in self.observations if item.status is normalized]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceLedger:
        return cls.model_validate(value)
