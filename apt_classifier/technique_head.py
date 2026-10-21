"""The fine-tuned technique head as mapping evidence (docs 21.3.18, product integration).

The head is the family/technique classifier of ``scripts/cribl_finetune/family_head.py``: a
4-bit Qwen3.5-4B base with a LoRA and two cosine heads that read one sentence and return a
probability over every ATT&CK technique family (the parent techniques) and, within each family,
over its techniques. It needs transformers 5, which the product's ``.venv`` cannot hold
(ATTACK-BERT needs 4.57), so it runs as a small local HTTP service
(``scripts/cribl_finetune/serve.py``, ``make technique-head-serve``) and this module is the
product's client.

What the pipeline gets from it, per observation (:class:`TechniqueHeadScores`):

- ``technique_head`` evidence on every candidate: the head's probability for the candidate's
  family and for the candidate itself, the ranks, the head's confidence in its own top family,
  and whether the candidate is one of the head's proposals; the acceptance model reads these as
  :data:`FEATURE_NAMES` (feature schema 4, :mod:`apt_classifier.acceptance`);
- up to ``technique_head_new_candidates`` techniques the head proposes that retrieval did not
  (its best technique in each of the ``technique_head_top_families`` most probable families),
  appended after the retrieved pool with fused score 0, like exemplar-only candidates, so nothing
  retrieved is displaced.

Prose only, for now: observations that come from logs (:func:`scores_prose`) are left alone.

Fail closed: the client is built only when ``TECHNIQUE_HEAD_ENABLED`` is set, is ``None`` (the
pipeline runs exactly as without it, the status says why) when the service does not answer, was
built for another ATT&CK version, or does not carry the pinned
``TECHNIQUE_HEAD_EXPECTED_FINGERPRINT``; an acceptance model fitted with the head records its
fingerprint and refuses to run on any other head or without one. A request that fails mid-run
yields ``None`` scores for its observations (no evidence, no proposals) and is counted in the
diagnostics, so a benchmark can refuse to write a run the head did not score throughout.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from apt_classifier.cyberre import is_telemetry_source

EVIDENCE_SOURCE = "technique_head"
DEFAULT_URL = "http://localhost:8102"
DEFAULT_TOP_FAMILIES = 3
DEFAULT_NEW_CANDIDATES = 3
DEFAULT_BATCH = 16
DEFAULT_TIMEOUT = 120.0
DEFAULT_CACHE_SIZE = 2048
NO_RANK = 0.0
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")

# The acceptance features (schema 4) rebuilt from the ``technique_head`` evidence:
# th_family_p      P(family of the candidate) from the family head
# th_technique_p   P(candidate technique) = P(family) x P(technique | family) (0 outside top-K)
# th_family_rank   1 / rank of the candidate's family (0 outside the kept families)
# th_technique_rank 1 / rank of the candidate technique (0 outside the kept techniques)
# th_top_p         P(the head's own top family): how sure the head is about this sentence
# th_proposed      the candidate is one of the head's proposals (best technique of a top family)
# th_only          the candidate exists only because the head proposed it (no retrieval hit)
FEATURE_NAMES: tuple[str, ...] = (
    "th_family_p",
    "th_technique_p",
    "th_family_rank",
    "th_technique_rank",
    "th_top_p",
    "th_proposed",
    "th_only",
)
# Schema 5 (2026-10-19, prereg cribl_multi_answer_acceptance_2026-10-19): the head's second
# answer as relative evidence, with the statement length
# th_rel_family       P(candidate's family) / P(the head's top family)
# th_rel_technique    P(candidate) / P(the head's top technique)
# th_second           the candidate's family is the head's 2nd or 3rd with th_rel_family >= 0.25
# statement_log_words log(1 + words of the judged text)
# th_second_x_words   th_second x statement_log_words
MULTI_FEATURE_NAMES: tuple[str, ...] = (
    "th_rel_family",
    "th_rel_technique",
    "th_second",
    "statement_log_words",
    "th_second_x_words",
)
ALL_FEATURE_NAMES: tuple[str, ...] = (*FEATURE_NAMES, *MULTI_FEATURE_NAMES)
SECOND_ANSWER_KNEE = 0.25


def scores_prose(observation: Any) -> bool:
    """Whether the head reads this observation: prose only, for now (report sentences, free
    text, alerts, interview answers). Log-derived observations (Windows/Sysmon events, eCAR,
    Elastic records, the EvidenceForge renders: the telemetry sources of the verifier's
    interpreter rule, :func:`apt_classifier.cyberre.is_telemetry_source`) are not scored: the
    head was trained on report prose, and the product owner asked for prose first
    (2026-10-19)."""
    provenance = getattr(observation, "provenance", None)
    return not is_telemetry_source(getattr(provenance, "source", None))


def family_of(technique_id: str) -> str:
    """The family (parent technique id) of a technique id: ``T1059.001 -> T1059``."""
    return str(technique_id).split(".", 1)[0]


@dataclass(frozen=True)
class TechniqueHeadScores:
    """What the head says about one observation: its kept families and techniques (probability
    and rank, most probable first) and its proposals."""

    families: dict[str, float] = field(default_factory=dict)
    techniques: dict[str, float] = field(default_factory=dict)
    proposals: tuple[str, ...] = ()
    status: str = "ok"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def family_ranks(self) -> dict[str, int]:
        return {family: rank for rank, family in enumerate(self.families, 1)}

    @property
    def technique_ranks(self) -> dict[str, int]:
        return {technique: rank for rank, technique in enumerate(self.techniques, 1)}

    @property
    def top_family(self) -> str | None:
        return next(iter(self.families), None)

    @property
    def top_p(self) -> float:
        return float(next(iter(self.families.values()), 0.0))

    def features(
        self, technique_id: str, *, head_only: bool = False, words: int = 0
    ) -> dict[str, float]:
        """The schema-4 features and the schema-5 ones (``MULTI_FEATURE_NAMES``; ``words`` is
        the length of the judged text); a schema-4 model reads its own prefix."""
        family = family_of(technique_id)
        family_rank = self.family_ranks.get(family)
        technique_rank = self.technique_ranks.get(technique_id)
        family_p = float(self.families.get(family, 0.0))
        technique_p = float(self.techniques.get(technique_id, 0.0))
        top_technique_p = float(next(iter(self.techniques.values()), 0.0))
        rel_family = family_p / self.top_p if self.top_p > 0 else 0.0
        rel_technique = technique_p / top_technique_p if top_technique_p > 0 else 0.0
        second = 1.0 if family_rank in (2, 3) and rel_family >= SECOND_ANSWER_KNEE else 0.0
        log_words = math.log1p(max(int(words), 0))
        return {
            "th_family_p": family_p,
            "th_technique_p": technique_p,
            "th_family_rank": 1.0 / family_rank if family_rank else NO_RANK,
            "th_technique_rank": 1.0 / technique_rank if technique_rank else NO_RANK,
            "th_top_p": self.top_p,
            "th_proposed": 1.0 if technique_id in self.proposals else 0.0,
            "th_only": 1.0 if head_only else 0.0,
            "th_rel_family": rel_family,
            "th_rel_technique": rel_technique,
            "th_second": second,
            "statement_log_words": log_words,
            "th_second_x_words": second * log_words,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, top_families: int) -> TechniqueHeadScores:
        """One service row (``scripts/cribl_finetune/head_service.py``): ``families`` and
        ``techniques`` as ``[[id, p], ...]`` most probable first."""
        if str(row.get("status", "ok")) != "ok":
            return cls(status="error", error=str(row.get("error") or "scoring failed"))
        families = _ordered(row.get("families"))
        techniques = _ordered(row.get("techniques"))
        return cls(
            families=families,
            techniques=techniques,
            proposals=propose(families, techniques, top_families=top_families),
        )


def _ordered(pairs: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in pairs or ():
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            continue
        identifier, probability = item
        try:
            value = float(probability)
        except (TypeError, ValueError):
            continue
        if identifier and value >= 0.0:
            output.setdefault(str(identifier), value)
    return dict(sorted(output.items(), key=lambda item: (-item[1], item[0])))


def propose(
    families: Mapping[str, float], techniques: Mapping[str, float], *, top_families: int
) -> tuple[str, ...]:
    """The head's proposals: for each of the ``top_families`` most probable families its most
    probable technique (the family itself when none of its techniques was kept)."""
    proposals: list[str] = []
    for family in list(families)[: max(int(top_families), 0)]:
        best = next((tid for tid in techniques if family_of(tid) == family), family)
        if best not in proposals:
            proposals.append(best)
    return tuple(proposals)


def pool_scores(parts: Sequence[TechniqueHeadScores], *, top_families: int) -> TechniqueHeadScores:
    """One observation's scores pooled over its sentences: per family and technique the largest
    probability any sentence gave (a paragraph describes several techniques; the head reads one
    sentence at a time), re-ranked, with the proposals recomputed from the pooled ranking."""
    scored = [part for part in parts if part.ok]
    if not scored:
        return parts[0] if parts else TechniqueHeadScores(status="error", error="no sentences")
    if len(scored) == 1:
        return scored[0]
    families: dict[str, float] = {}
    techniques: dict[str, float] = {}
    for part in scored:
        for family, probability in part.families.items():
            families[family] = max(families.get(family, 0.0), probability)
        for technique, probability in part.techniques.items():
            techniques[technique] = max(techniques.get(technique, 0.0), probability)
    families = dict(sorted(families.items(), key=lambda item: (-item[1], item[0])))
    techniques = dict(sorted(techniques.items(), key=lambda item: (-item[1], item[0])))
    return TechniqueHeadScores(
        families=families,
        techniques=techniques,
        proposals=propose(families, techniques, top_families=top_families),
    )


def empty_features() -> dict[str, float]:
    """The features of a candidate without ``technique_head`` evidence (the head off)."""
    return {name: 0.0 for name in ALL_FEATURE_NAMES}


def features_from_evidence(evidence: Any) -> dict[str, float]:
    """Rebuild the features from a candidate's ``technique_head`` evidence item."""
    if evidence is None:
        return empty_features()
    metadata = dict(getattr(evidence, "metadata", {}) or {})
    values = metadata.get("features")
    base = empty_features()
    if isinstance(values, Mapping):
        for name in ALL_FEATURE_NAMES:
            try:
                base[name] = float(values.get(name, base[name]))
            except (TypeError, ValueError):
                continue
    return base


@dataclass(frozen=True)
class TechniqueHeadConfig:
    enabled: bool = False
    url: str = DEFAULT_URL
    timeout: float = DEFAULT_TIMEOUT
    expected_fingerprint: str = ""
    attack_version: str | None = None
    # the knowledge base's technique ids: a head fitted on an older ATT&CK release is accepted
    # on a newer one when every technique it knows still exists (new techniques are simply
    # never proposed); empty = strict version match
    known_technique_ids: frozenset[str] = frozenset()
    top_families: int = DEFAULT_TOP_FAMILIES
    new_candidates: int = DEFAULT_NEW_CANDIDATES
    batch: int = DEFAULT_BATCH
    cache_size: int = DEFAULT_CACHE_SIZE


def config_from_settings(
    settings: Any,
    attack_version: str | None = None,
    known_technique_ids: Iterable[str] = (),
) -> TechniqueHeadConfig:
    return TechniqueHeadConfig(
        enabled=bool(getattr(settings, "technique_head_enabled", False)),
        url=str(getattr(settings, "technique_head_url", DEFAULT_URL) or DEFAULT_URL),
        timeout=float(getattr(settings, "technique_head_timeout", DEFAULT_TIMEOUT)),
        expected_fingerprint=str(
            getattr(settings, "technique_head_expected_fingerprint", "") or ""
        ).strip(),
        attack_version=attack_version,
        known_technique_ids=frozenset(str(t) for t in known_technique_ids),
        top_families=int(getattr(settings, "technique_head_top_families", DEFAULT_TOP_FAMILIES)),
        new_candidates=int(
            getattr(settings, "technique_head_new_candidates", DEFAULT_NEW_CANDIDATES)
        ),
        batch=int(getattr(settings, "technique_head_batch", DEFAULT_BATCH)),
        cache_size=int(getattr(settings, "technique_head_cache_size", DEFAULT_CACHE_SIZE)),
    )


class TechniqueHeadClient:
    """The product's client of the technique head service.

    ``probe()`` reads ``/health`` and validates the service's manifest; ``score(texts)`` posts
    the texts to ``/score`` in batches and returns one :class:`TechniqueHeadScores` per text
    (``None`` when the request failed, counted in ``errors``). Texts are cached per process
    (``cache_size`` most recent), so reopening a case or re-running a benchmark arm never
    re-scores a sentence."""

    def __init__(self, config: TechniqueHeadConfig, *, transport: Any = None) -> None:
        self.config = config
        self.manifest: dict[str, Any] = {}
        self.status = "disabled" if not config.enabled else "not probed"
        self.errors = 0
        self.requests = 0
        self.last_error: str | None = None
        self._cache: OrderedDict[str, TechniqueHeadScores] = OrderedDict()
        self._client = httpx.Client(timeout=config.timeout, transport=transport)

    # ----- identity -----
    @property
    def fingerprint(self) -> str | None:
        value = self.manifest.get("fingerprint")
        return str(value) if value else None

    @property
    def available(self) -> bool:
        return self.status.startswith("active")

    def identity(self) -> dict[str, Any]:
        """What a run records about the head it used."""
        keys = (
            "fingerprint",
            "checkpoint",
            "base",
            "adapter",
            "attack_version",
            "families",
            "techniques",
            "families_considered",
            "top_kept",
        )
        return {
            "status": self.status,
            "url": self.config.url,
            "top_families": self.config.top_families,
            "new_candidates": self.config.new_candidates,
            "errors": self.errors,
            "requests": self.requests,
            **{key: self.manifest.get(key) for key in keys},
        }

    # ----- health -----
    def probe(self) -> str:
        """Validate the service and set ``status``; returns the status."""
        if not self.config.enabled:
            self.status = "disabled"
            return self.status
        try:
            response = self._client.get(f"{self.config.url.rstrip('/')}/health")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            self.status = f"unavailable: {str(error)[:200]}"
            return self.status
        problem = manifest_problem(payload, self.config)
        if problem:
            self.status = f"unavailable: {problem}"
            return self.status
        self.manifest = dict(payload)
        self.status = "active" if self.config.expected_fingerprint else "active (unpinned)"
        version = str(payload.get("attack_version") or "")
        if self.config.attack_version and version and version != self.config.attack_version:
            self.status += (
                f" (head fitted on ATT&CK {version}, knowledge base {self.config.attack_version}: "
                f"all {len(payload.get('technique_ids') or ())} head techniques present)"
            )
        return self.status

    # ----- scoring -----
    def score(self, texts: Sequence[str]) -> list[TechniqueHeadScores | None]:
        """One scores object per text (``None`` for a text the service failed to answer)."""
        output: list[TechniqueHeadScores | None] = [None] * len(texts)
        pending: dict[str, list[int]] = {}
        for position, text in enumerate(texts):
            key = _cache_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                output[position] = cached
            else:
                pending.setdefault(key, []).append(position)
        keys = list(pending)
        batch = max(int(self.config.batch), 1)
        for start in range(0, len(keys), batch):
            chunk = keys[start : start + batch]
            rows = self._post([texts[pending[key][0]] for key in chunk])
            if rows is None:
                continue
            for key, row in zip(chunk, rows, strict=True):
                scores = TechniqueHeadScores.from_row(row, top_families=self.config.top_families)
                self._remember(key, scores)
                for position in pending[key]:
                    output[position] = scores
        return output

    def _post(self, texts: list[str]) -> list[Mapping[str, Any]] | None:
        self.requests += 1
        try:
            response = self._client.post(
                f"{self.config.url.rstrip('/')}/score", json={"texts": list(texts)}
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("rows") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list) or len(rows) != len(texts):
                raise ValueError("the service answered with the wrong number of rows")
            if self.fingerprint and str(payload.get("fingerprint") or "") != self.fingerprint:
                raise ValueError("the service's fingerprint changed since it was probed")
            return [row if isinstance(row, Mapping) else {"status": "error"} for row in rows]
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            self.errors += 1
            self.last_error = str(error)[:300]
            return None

    def _remember(self, key: str, scores: TechniqueHeadScores) -> None:
        if self.config.cache_size <= 0:
            return
        self._cache[key] = scores
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)


def _cache_key(text: str) -> str:
    return " ".join(str(text).split())


def manifest_problem(payload: Any, config: TechniqueHeadConfig) -> str | None:
    """Why a service manifest (``/health``) is refused, ``None`` when it may be used."""
    if not isinstance(payload, Mapping):
        return "the health answer is not an object"
    if str(payload.get("status") or "") != "ready":
        return f"the service is not ready ({payload.get('status')!r})"
    if str(payload.get("task") or "technique") != "technique":
        return f"the service runs a {payload.get('task')!r} head, not the technique head"
    fingerprint = str(payload.get("fingerprint") or "")
    if not _FINGERPRINT_RE.match(fingerprint):
        return "the service reports no checkpoint fingerprint"
    if config.expected_fingerprint and fingerprint != config.expected_fingerprint:
        return (
            f"checkpoint fingerprint {fingerprint[:12]}... is not the pinned "
            f"TECHNIQUE_HEAD_EXPECTED_FINGERPRINT {config.expected_fingerprint[:12]}..."
        )
    version = str(payload.get("attack_version") or "")
    if config.attack_version and version and version != config.attack_version:
        problem = newer_release_problem(payload, config)
        if problem:
            return problem
    if not payload.get("techniques"):
        return "the service carries no technique head (family only)"
    return None


def _release_key(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in version.strip().removeprefix("v").split("."))
    except ValueError:
        return None


def newer_release_problem(payload: Mapping[str, Any], config: TechniqueHeadConfig) -> str | None:
    """A head fitted on an older ATT&CK release may serve a newer knowledge base when the
    service lists its technique ids and every one of them still exists there (ATT&CK only adds
    techniques between releases; a revoked id would make the head's answers unmappable).
    Returns the refusal, ``None`` when the newer release is accepted."""
    version = str(payload.get("attack_version") or "")
    expected = str(config.attack_version or "")
    refusal = f"ATT&CK version {version} != {expected}"
    head_key, kb_key = _release_key(version), _release_key(expected)
    if head_key is None or kb_key is None or kb_key <= head_key:
        return refusal
    if not config.known_technique_ids:
        return refusal + " (no knowledge base to check the head's techniques against)"
    ids = payload.get("technique_ids")
    if not isinstance(ids, list | tuple) or not ids:
        return refusal + " (the service lists no technique ids; restart it on current code)"
    missing = sorted(str(t) for t in ids if str(t) not in config.known_technique_ids)
    if missing:
        return (
            refusal
            + f" ({len(missing)} head techniques are missing from the knowledge base: "
            + ", ".join(missing[:5])
            + ")"
        )
    return None


def load_technique_head(
    settings: Any, kb: Any = None, *, transport: Any = None
) -> tuple[TechniqueHeadClient | None, str]:
    """The configured client, probed, fail-closed: ``(client, status)`` with ``client`` ``None``
    when the head is disabled or the service cannot be used (the status says why)."""
    attack_version = str(getattr(kb, "version", "") or "") or None
    config = config_from_settings(
        settings, attack_version, tuple(getattr(kb, "techniques", {}) or {})
    )
    if not config.enabled:
        return None, "disabled"
    client = TechniqueHeadClient(config, transport=transport)
    status = client.probe()
    if not client.available:
        return None, status
    return client, status


def technique_head_kwargs(settings: Any, client: TechniqueHeadClient | None) -> dict[str, Any]:
    """Keyword arguments for ``retrieve_techniques`` from the settings and a loaded client."""
    return {
        "technique_head": client,
        "technique_head_new_candidates": int(
            getattr(settings, "technique_head_new_candidates", DEFAULT_NEW_CANDIDATES)
        ),
        "technique_head_top_families": int(
            getattr(settings, "technique_head_top_families", DEFAULT_TOP_FAMILIES)
        ),
    }


def describe_head_scores(scores: TechniqueHeadScores, technique_id: str) -> str:
    """One line for the evidence rationale."""
    family = family_of(technique_id)
    family_p = scores.families.get(family)
    technique_p = scores.techniques.get(technique_id)
    rank = scores.family_ranks.get(family)
    if family_p is None:
        return "the technique head does not rank this technique's family among its top families"
    parts = [f"the technique head puts the family at rank {rank} with P {family_p:.2f}"]
    if technique_p is not None:
        parts.append(f"the technique itself at P {technique_p:.2f}")
    if technique_id in scores.proposals:
        parts.append("and proposes it")
    return ", ".join(parts)
