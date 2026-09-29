"""Optional, manifest-gated local technique reranker.

A reranker is allowed to influence acceptance only when its model artifact has a
valid, compatible ``apt_classifier_manifest.json``.  Loading is deliberately
fail-closed: a missing or incompatible manifest leaves the adapter unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apt_classifier.evidence.models import TechniqueCandidate, TechniqueRetrievalEvidence

MANIFEST_FILENAME = "apt_classifier_manifest.json"
# "balanced_f1_dev" is the pre-2026-09-23 objective (it could collapse promote and demote to
# one threshold); "precision_floor_band_dev" selects an abstention band by precision floors.
SUPPORTED_CALIBRATION_METHODS = frozenset({"precision_floor_band_dev", "balanced_f1_dev"})


def technique_pair_text(kb: Any, technique_id: str, *, max_description_chars: int = 3000) -> str:
    """The technique side of a (observation, technique) cross-encoder pair.

    Shared by training (``scripts/train_technique_reranker.py``) and inference so the
    text the model sees never drifts between the two; it is what
    ``RERANKER_FEATURE_SCHEMA['technique_text_fields']`` declares.
    """
    technique = getattr(kb, "techniques", {}).get(technique_id)
    if technique is None:
        return technique_id
    name = str(getattr(technique, "name", "") or technique_id)
    tactics = ", ".join(str(item) for item in getattr(technique, "tactics", ()) or ())
    description = str(getattr(technique, "description", "") or "")
    parts = [f"{technique_id} {name}."]
    if tactics:
        parts.append(f"Tactics: {tactics}.")
    if description:
        parts.append(description[:max_description_chars])
    return " ".join(parts).strip()


RERANKER_MANIFEST_SCHEMA_VERSION = 1
RERANKER_FEATURE_SCHEMA: dict[str, Any] = {
    "version": 1,
    "inputs": ["observation", "technique_text"],
    "technique_text_fields": ["id", "name", "tactics", "description"],
    "max_length": 512,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model_name: str = ""
    model_revision: str | None = None
    max_candidates: int = 20
    fallback_on_error: bool = True
    attack_version: str | None = None
    expected_base_model: str | None = None
    expected_base_model_revision: str | None = None
    expected_artifact_fingerprint: str | None = None


def artifact_fingerprint(manifest: dict[str, Any]) -> str:
    """Return the deterministic fingerprint for a manifest payload."""
    payload = {key: value for key, value in manifest.items() if key != "artifact_fingerprint"}
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class LocalReranker:
    """Adapter for a manifest-gated sentence-transformers cross-encoder."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: Any = None
        self._manifest: dict[str, Any] | None = None
        self._load_error: str | None = None
        if config.enabled:
            if config.model_name:
                self._load_model()
            else:
                self._load_error = "Reranker model is enabled but RERANKER_MODEL is empty"

    def _manifest_path(self) -> Path:
        model_path = Path(self.config.model_name).expanduser()
        if model_path.is_dir():
            path = model_path / MANIFEST_FILENAME
            if not path.is_file():
                raise ValueError(f"missing {MANIFEST_FILENAME} in reranker artifact {model_path}")
            return path
        if model_path.exists():
            raise ValueError(f"reranker model path is not a directory: {model_path}")

        # A Hub artifact must carry the same manifest.  Pinning model_revision is
        # mandatory for remote artifacts so validation cannot race a moving tag.
        if not self.config.model_revision:
            raise ValueError(
                f"remote reranker artifact {self.config.model_name!r} requires a pinned revision"
            )
        try:
            from huggingface_hub import hf_hub_download

            return Path(
                hf_hub_download(
                    repo_id=self.config.model_name,
                    filename=MANIFEST_FILENAME,
                    revision=self.config.model_revision,
                )
            )
        except Exception as exc:
            raise ValueError(
                f"could not load {MANIFEST_FILENAME} for {self.config.model_name!r}: {exc}"
            ) from exc

    def _load_model(self) -> None:
        try:
            path = self._manifest_path()
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._manifest = self._validate_manifest(raw)

            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.config.model_name,
                revision=self.config.model_revision,
                trust_remote_code=False,
            )
        except Exception as exc:
            self._model = None
            self._manifest = None
            self._load_error = f"Reranker unavailable: {str(exc)[:450]}"

    def _validate_manifest(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("reranker manifest must be a JSON object")
        if raw.get("schema_version") != RERANKER_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "reranker manifest schema_version mismatch: "
                f"expected {RERANKER_MANIFEST_SCHEMA_VERSION}, got {raw.get('schema_version')!r}"
            )
        if raw.get("artifact_type") != "apt_classifier_technique_reranker":
            raise ValueError("reranker manifest artifact_type is incompatible")

        attack_version = raw.get("attack_version")
        if not isinstance(attack_version, str) or attack_version.strip().lower() in {
            "",
            "latest",
            "unknown",
        }:
            raise ValueError("reranker manifest requires a pinned ATT&CK version")
        if self.config.attack_version and attack_version != self.config.attack_version:
            raise ValueError(
                f"ATT&CK version mismatch: expected {self.config.attack_version!r}, "
                f"got {attack_version!r}"
            )

        base_model = raw.get("base_model")
        base_revision = raw.get("base_model_revision")
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("reranker manifest base_model is missing")
        if not isinstance(base_revision, str) or not base_revision.strip():
            raise ValueError("reranker manifest base_model_revision must be pinned")
        if self.config.expected_base_model and base_model != self.config.expected_base_model:
            raise ValueError(
                f"base model mismatch: expected {self.config.expected_base_model!r}, "
                f"got {base_model!r}"
            )
        if (
            self.config.expected_base_model_revision
            and base_revision != self.config.expected_base_model_revision
        ):
            raise ValueError(
                "base model revision mismatch: expected "
                f"{self.config.expected_base_model_revision!r}, got {base_revision!r}"
            )

        training_hash = raw.get("training_data_hash")
        if not isinstance(training_hash, str) or not _SHA256_RE.fullmatch(training_hash):
            raise ValueError("reranker manifest training_data_hash is not a SHA-256 digest")
        if raw.get("feature_schema") != RERANKER_FEATURE_SCHEMA:
            raise ValueError("reranker manifest feature_schema is incompatible")

        calibration = raw.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("reranker manifest calibration is missing")
        if calibration.get("method") not in SUPPORTED_CALIBRATION_METHODS:
            raise ValueError("reranker manifest calibration method is incompatible")
        promote = calibration.get("promote_threshold")
        demote = calibration.get("demote_threshold")
        if not self._valid_threshold(promote) or not self._valid_threshold(demote):
            raise ValueError(
                "reranker manifest calibrated thresholds must be finite values in [0, 1]"
            )
        if float(demote) > float(promote):
            raise ValueError("reranker manifest demote_threshold exceeds promote_threshold")
        if (
            not isinstance(calibration.get("dev_pair_count"), int)
            or calibration["dev_pair_count"] < 1
        ):
            raise ValueError("reranker manifest calibration requires a non-empty held-out dev set")
        metrics = calibration.get("held_out_metrics")
        if not isinstance(metrics, dict) or not self._valid_threshold(metrics.get("balanced_f1")):
            raise ValueError("reranker manifest held_out_metrics are missing or invalid")

        fingerprint = raw.get("artifact_fingerprint")
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError("reranker manifest artifact_fingerprint is not a SHA-256 digest")
        computed = artifact_fingerprint(raw)
        if fingerprint != computed:
            raise ValueError("reranker manifest artifact_fingerprint does not match its contents")
        expected = self.config.expected_artifact_fingerprint
        if expected and fingerprint != expected:
            raise ValueError(
                f"artifact fingerprint mismatch: expected {expected!r}, got {fingerprint!r}"
            )
        return dict(raw)

    @staticmethod
    def _valid_threshold(value: Any) -> bool:
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )

    def _validate_attack_kb(self, kb: Any) -> bool:
        if self._manifest is None:
            return False
        version = getattr(kb, "attack_version", getattr(kb, "version", None))
        if version is None:
            return True
        expected = self._manifest["attack_version"]
        if str(version) != expected:
            self._model = None
            self._manifest = None
            self._load_error = (
                f"Reranker unavailable: ATT&CK version mismatch: artifact {expected!r}, "
                f"knowledge base {str(version)!r}"
            )
            return False
        return True

    @property
    def available(self) -> bool:
        return self.config.enabled and self._model is not None and self._manifest is not None

    @property
    def validated(self) -> bool:
        """Whether the artifact can influence acceptance decisions."""
        return self.available and self._load_error is None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def manifest(self) -> dict[str, Any] | None:
        """Return a defensive copy of the validated manifest."""
        return dict(self._manifest) if self._manifest is not None else None

    @property
    def calibrated_thresholds(self) -> tuple[float, float] | None:
        """Return ``(promote, demote)`` thresholds from the validated artifact."""
        if not self.validated or self._manifest is None:
            return None
        calibration = self._manifest["calibration"]
        return float(calibration["promote_threshold"]), float(calibration["demote_threshold"])

    @property
    def promote_threshold(self) -> float | None:
        thresholds = self.calibrated_thresholds
        return thresholds[0] if thresholds is not None else None

    @property
    def demote_threshold(self) -> float | None:
        thresholds = self.calibrated_thresholds
        return thresholds[1] if thresholds is not None else None

    def acceptance_score(self, candidate: TechniqueCandidate) -> float | None:
        """Return a calibrated acceptance signal for a validated artifact.

        Existing acceptance callers use a stable high/low signal boundary.  The
        artifact-specific dev thresholds are therefore applied here: scores in
        the promote region map to 1, scores in the demote region map to 0, and
        the calibrated abstention band maps to 0.5.  The raw model score remains
        available in the candidate's ``reranker`` evidence.
        """
        if not self.validated:
            return None
        reranker_ev = candidate.evidence_for("reranker")
        thresholds = self.calibrated_thresholds
        if reranker_ev is None or reranker_ev.score is None or thresholds is None:
            return None
        score = float(reranker_ev.score)
        if not math.isfinite(score):
            return None
        promote, demote = thresholds
        if score >= promote:
            return 1.0
        if score <= demote:
            return 0.0
        return 0.5

    def rerank(
        self,
        observation_text: str,
        candidates: list[TechniqueCandidate] | tuple[TechniqueCandidate, ...],
        kb: Any,
    ) -> tuple[TechniqueCandidate, ...]:
        """Rerank existing candidates, failing back without adding candidates."""
        supplied = tuple(candidates)
        if not self.available or not supplied or not self._validate_attack_kb(kb):
            return supplied
        limit = min(len(supplied), self.config.max_candidates)
        subset = supplied[:limit]
        pairs: list[tuple[str, str]] = [
            (observation_text, technique_pair_text(kb, candidate.technique_id))
            for candidate in subset
        ]
        try:
            scores = self._model.predict(pairs)
        except Exception:
            if self.config.fallback_on_error:
                return supplied
            raise
        # Confidence-gated reordering: only candidates the calibrated model is confident
        # about (score >= promote threshold) move to the front, ordered by score; every
        # other candidate keeps its fused retrieval order. A weak or off-distribution
        # reranker therefore cannot scramble a ranking it is unsure about. Without a
        # validated calibration nothing is reordered; scores are still recorded.
        promote = self.promote_threshold if self.validated else None
        score_rank = {
            index: rank
            for rank, (index, _) in enumerate(
                sorted(
                    zip(range(len(subset)), scores, strict=False),
                    key=lambda pair: (-float(pair[1]), pair[0]),
                ),
                1,
            )
        }
        confident = sorted(
            (
                index
                for index in range(len(subset))
                if promote is not None and float(scores[index]) >= promote
            ),
            key=lambda index: (-float(scores[index]), index),
        )
        remaining_order = [index for index in range(len(subset)) if index not in set(confident)]
        order = [*confident, *remaining_order]
        reranked = tuple(
            subset[index].model_copy(
                update={
                    "evidence": [
                        *subset[index].evidence,
                        TechniqueRetrievalEvidence(
                            source="reranker",
                            text=pairs[index][1],
                            score=float(scores[index]),
                            rank=score_rank[index],
                        ),
                    ],
                    "fused_rank": position,
                }
            )
            for position, index in enumerate(order, 1)
        )
        remaining = tuple(
            candidate.model_copy(update={"fused_rank": limit + offset})
            for offset, candidate in enumerate(supplied[limit:], 1)
        )
        return (*reranked, *remaining)
