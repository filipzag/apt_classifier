import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from apt_classifier.acceptance import candidate_features
from apt_classifier.attack.parse import split_sentences, strip_attack_citations
from apt_classifier.classifiers.base import ClassifierContext
from apt_classifier.evidence.iocs import match_software_aliases
from apt_classifier.evidence.models import (
    EvidenceLedger,
    EvidenceStatus,
    TechniqueCandidate,
    TechniqueCandidateState,
    TechniqueMatch,
    TechniqueRetrievalEvidence,
    TechniqueReviewAction,
    TechniqueVerificationVerdict,
)
from apt_classifier.exemplars import (
    DEFAULT_DUPLICATE_SIMILARITY as EXEMPLAR_DUPLICATE_SIMILARITY,
)
from apt_classifier.exemplars import DEFAULT_K as EXEMPLAR_K
from apt_classifier.exemplars import DEFAULT_TAU as EXEMPLAR_TAU
from apt_classifier.exemplars import EVIDENCE_SOURCE as EXEMPLAR_SOURCE
from apt_classifier.exemplars import ExemplarVotes, exemplar_query_text
from apt_classifier.exemplars import parent_map as exemplar_parent_map
from apt_classifier.technique_signals import find_signals
from apt_classifier.vectorstore import (
    ChromaVectorStore,
    VectorHit,
    load_synthetic_sentences,
    procedure_sentences_by_technique,
)

_BENIGN_INDICATORS = re.compile(
    r"\b(?:self[- ]service|own\s+password|user\s+reset|password\s+reset"
    r"|automatic(?:ally)?\s+(?:update|updated|signatures)"
    r"|antivirus\s+(?:software\s+)?(?:update|updated|signatures)"
    r"|updated\s+(?:its\s+)?signatures"
    r"|mistype(?:s|d)?\s+(?:a\s+)?password|typo\s+in\s+password"
    r"|fat[- ]fingered|accidentally\s+(?:locked|typed|entered)"
    r"|administrator\s+(?:logged|check|checked|normal|routine|installed)"
    r"|routine\s+(?:maintenance|check|update|backup)|legitimate\s+(?:user|admin|activity)"
    r"|normal\s+(?:user|admin|activity|operation|logon|login)"
    r"|scheduled\s+backup|backup\s+of\s+(?:user|documents|files|data)"
    r"|software\s+(?:center|update|installation|deploy(?:ed|ment)|upgrade)"
    r"|generic\s+(?:admin|administrative)\s+activity"
    r"|system\s+(?:update|upgrade|maintenance|patching)"
    r"|patch\s+(?:management|deployment|installation))\b",
    re.IGNORECASE,
)


_EXPLICIT_TECHNIQUE_CUES = {
    "T1059.001": re.compile(
        r"\bpowershell\b[\s\S]{0,80}(?:encoded|enc\b|-enc\b|-e\b|hidden\s+window|-w\s+hidden)"
        r"|(?:encoded|enc\b|-enc\b|-e\b|hidden\s+window|-w\s+hidden)[\s\S]{0,40}\bpowershell\b",
        re.IGNORECASE,
    ),
    "T1003.001": re.compile(
        r"\blsass\b[\s\S]{0,60}(?:dump|memory|credential|access)"
        r"|(?:dump|memory|credential)[\s\S]{0,60}\blsass\b",
        re.IGNORECASE,
    ),
    "T1003.002": re.compile(
        r"\bsam\b[\s\S]{0,60}(?:registry|hive|dump|dumped)"
        r"|(?:registry|hive)[\s\S]{0,60}\bsam\b"
        r"|dumped\s+from\s+(?:the\s+)?sam",
        re.IGNORECASE,
    ),
    "T1021.002": re.compile(
        r"\bsmb\b[\s\S]{0,60}(?:admin|c\$|ipc\$|share)"
        r"|(?:admin|c\$|ipc\$)[\s\S]{0,40}(?:smb|share)",
        re.IGNORECASE,
    ),
    "T1105": re.compile(
        r"(?:download|ingress|transfer|fetch|retrieved)[\s\S]{0,60}"
        r"(?:payload|binary|executable|tool|file)"
        r"|(?:payload|binary|executable|tool)[\s\S]{0,40}"
        r"(?:download|ingress|transfer|fetch|retrieved)",
        re.IGNORECASE,
    ),
    "T1218.005": re.compile(
        r"\bmshta\b[\s\S]{0,80}(?:execute|run|payload|hta|remote)"
        r"|(?:execute|run|payload)[\s\S]{0,40}\bmshta\b"
        r"|\.hta\b",
        re.IGNORECASE,
    ),
    "T1218.011": re.compile(
        r"\brundll32\b[\s\S]{0,60}(?:dll|load|execute)"
        r"|(?:dll)[\s\S]{0,40}\brundll32\b",
        re.IGNORECASE,
    ),
    "T1055": re.compile(
        r"(?:inject|injection|hijack|createremotethread|setwindowshookex)"
        r"[\s\S]{0,60}(?:process|thread|dll|code|payload)"
        r"|(?:process|thread)[\s\S]{0,60}"
        r"(?:inject|injection|createremotethread)",
        re.IGNORECASE,
    ),
    "T1053.005": re.compile(
        r"(?:scheduled\s+task|schtasks|at\s+\d)[\s\S]{0,60}"
        r"(?:create|persistence|persist)",
        re.IGNORECASE,
    ),
    "T1113": re.compile(
        r"(?:screen\s*capture|screenshot|desktop\s+capture)",
        re.IGNORECASE,
    ),
    "T1056.001": re.compile(
        r"(?:keylog|key\s*log|keystroke|capture\s+keystroke)",
        re.IGNORECASE,
    ),
    "T1560.001": re.compile(
        r"(?:utility|utilities|command|executable|binary|tool|cli|shell)"
        r"[\s\S]{0,60}"
        r"(?:zip|7[- ]?zip|winrar|winzip|rar|tar|makecab|diantz)"
        r"|(?:zip|7[- ]?zip|winrar|winzip|rar|tar|makecab|diantz)"
        r"[\s\S]{0,60}"
        r"(?:utility|utilities|command|executable|binary|tool|cli|shell)"
        r"|(?:compress|compressed|compression)[\s\S]{0,40}"
        r"(?:zip|7[- ]?zip|winrar|winzip|rar|tar|archive)"
        r"|(?:zip|7[- ]?zip|winrar|winzip|rar|tar|archive)"
        r"[\s\S]{0,40}(?:compress|compressed|compression)",
        re.IGNORECASE,
    ),
    "T1047": re.compile(
        r"\bwmic\b[\s\S]{0,80}(?:execute|remote|process|call\s+create|lateral)"
        r"|(?:remote\s+process|lateral\s+movement)[\s\S]{0,40}\bwmic\b",
        re.IGNORECASE,
    ),
    "T1574.001": re.compile(
        r"(?:side[- ]?load|side[- ]?loading)[\s\S]{0,60}(?:dll|binary|executable|library)"
        r"|(?:dll|library)[\s\S]{0,40}(?:side[- ]?load|side[- ]?loading)",
        re.IGNORECASE,
    ),
    "T1687": re.compile(
        r"(?:disable|disabled|disabling|turn\s+off|stop)[\s\S]{0,60}"
        r"(?:defender|antivirus|anti[- ]?virus|security\s+tool|edr|av\b|windows\s+defender)"
        r"|(?:defender|antivirus|edr)[\s\S]{0,40}(?:disable|disabled|disabling|turn\s+off|stop)",
        re.IGNORECASE,
    ),
    "T1547.001": re.compile(
        r"(?:registry\s+run\s+key|run\s+key|runmru|HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run)"
        r"|(?:run\s+key)[\s\S]{0,40}(?:persistence|persist|startup)",
        re.IGNORECASE,
    ),
    "T1003.003": re.compile(
        r"\bntds\b[\s\S]{0,60}(?:dit|dump|copy|extract|credential|snapshot)"
        r"|(?:dit\b|ntds\.dit)[\s\S]{0,40}(?:copy|dump|extract|snapshot)",
        re.IGNORECASE,
    ),
    "T1134.001": re.compile(
        r"(?:token\s+impersonation|impersonate\s+(?:a\s+)?token|token\s+manipulation)"
        r"|(?:impersonat\w+)[\s\S]{0,40}(?:token|privilege|escalat)",
        re.IGNORECASE,
    ),
    "T1218": re.compile(
        r"(?:system\s+binar(?:y|ies))[\s\S]{0,60}(?:proxy|execution)"
        r"|(?:proxy\s+execution)[\s\S]{0,40}(?:system\s+binar)",
        re.IGNORECASE,
    ),
    "T1041": re.compile(
        r"(?:exfiltrat\w+)[\s\S]{0,60}(?:c2|c&c|command\s+and\s+control|https|http\s+server)"
        r"|(?:c2|c&c|command\s+and\s+control)[\s\S]{0,40}(?:exfiltrat\w+)",
        re.IGNORECASE,
    ),
}

_BEHAVIOR_GENERIC_TOKENS = {
    "access",
    "accessed",
    "actor",
    "adversary",
    "attack",
    "command",
    "commands",
    "compromised",
    "create",
    "created",
    "credential",
    "credentials",
    "data",
    "domain",
    "encoded",
    "execute",
    "executed",
    "execution",
    "file",
    "group",
    "host",
    "http",
    "https",
    "lsass",
    "malicious",
    "malware",
    "memory",
    "microsoft",
    "network",
    "newly",
    "payload",
    "periodic",
    "persistence",
    "powershell",
    "process",
    "registered",
    "scheduled",
    "script",
    "service",
    "spawned",
    "system",
    "task",
    "tool",
    "used",
    "user",
    "windows",
    "with",
}


@dataclass(frozen=True)
class BehaviorMatch:
    group_id: str
    observation_id: str
    technique_id: str
    score: float
    matched_text: str
    vector_id: str


@dataclass(frozen=True)
class LexicalDocument:
    id: str
    technique_id: str
    field: str
    text: str
    tokens: tuple[str, ...]
    technique_name: str = ""


LEXICAL_DEFAULT_FIELDS: tuple[str, ...] = ("name", "description", "procedure", "synthetic")


@dataclass(frozen=True)
class LexicalTechniqueIndex:
    """BM25 index over technique sentences with an inverted index for query time.

    ``postings`` maps a token to ``((document_index, term_frequency), ...)`` and
    ``norms`` holds the per-document BM25 length normalization, so a query touches only
    documents that share a token with it instead of rescanning the whole corpus.
    """

    documents: tuple[LexicalDocument, ...]
    document_frequencies: dict[str, int]
    average_length: float
    postings: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=dict, repr=False)
    norms: tuple[float, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.documents and (not self.postings or len(self.norms) != len(self.documents)):
            postings, norms = _build_postings(self.documents, self.average_length)
            object.__setattr__(self, "postings", postings)
            object.__setattr__(self, "norms", norms)

    @classmethod
    def from_kb(
        cls,
        kb: Any,
        *,
        fields: tuple[str, ...] = LEXICAL_DEFAULT_FIELDS,
        synthetic: dict[str, list[str]] | None = None,
    ) -> "LexicalTechniqueIndex":
        """Build the BM25 index over technique sentences.

        ``procedure`` sentences (anonymized actor procedure examples, the same text the
        dense index embeds) are included by default so that a dense procedure hit can be
        corroborated lexically; before this they could never be, by construction.
        """
        documents: list[LexicalDocument] = []
        frequencies: Counter[str] = Counter()
        procedures = procedure_sentences_by_technique(kb) if "procedure" in fields else {}
        if "synthetic" in fields and synthetic is None:
            synthetic = load_synthetic_sentences()
        synthetic = synthetic or {}
        for technique_id, technique in sorted(getattr(kb, "techniques", {}).items()):
            sentence_fields = dict(
                getattr(technique, "sentences", None)
                or {
                    "name": [strip_attack_citations(_technique_name(kb, technique_id))],
                    "description": split_sentences(str(getattr(technique, "description", ""))),
                    "detection": split_sentences(str(getattr(technique, "detection", ""))),
                }
            )
            if "procedure" in fields:
                sentence_fields["procedure"] = procedures.get(technique_id, [])
            if "synthetic" in fields:
                sentence_fields["synthetic"] = synthetic.get(technique_id, [])
            for field_name in fields:
                for index, sentence in enumerate(sentence_fields.get(field_name, ())):
                    text = str(sentence).strip()
                    tokens = tuple(_lexical_tokens(text))
                    if not text or not tokens:
                        continue
                    documents.append(
                        LexicalDocument(
                            id=f"{technique_id}:{field_name}:{index:04d}",
                            technique_id=technique_id,
                            field=field_name,
                            text=text,
                            tokens=tokens,
                            technique_name=_technique_name(kb, technique_id),
                        )
                    )
                    frequencies.update(set(tokens))
        average = (
            sum(len(document.tokens) for document in documents) / len(documents)
            if documents
            else 1.0
        )
        return cls(tuple(documents), dict(frequencies), average)

    def query(self, text: str, *, n_results: int) -> list[VectorHit]:
        query_tokens = _lexical_tokens(text)
        if not query_tokens or not self.documents:
            return []
        query_counts = Counter(query_tokens)
        document_count = len(self.documents)
        scores: dict[int, float] = {}
        for token, query_frequency in query_counts.items():
            posting = self.postings.get(token)
            if not posting:
                continue
            document_frequency = self.document_frequencies.get(token, len(posting))
            inverse_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            query_term = 1.0 + math.log(query_frequency)
            for document_index, frequency in posting:
                scores[document_index] = scores.get(document_index, 0.0) + (
                    inverse_frequency
                    * frequency
                    * 2.2
                    / (frequency + self.norms[document_index])
                    * query_term
                )
        scored: list[tuple[float, LexicalDocument]] = [
            (score, self.documents[index]) for index, score in scores.items() if score > 0
        ]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            VectorHit(
                id=document.id,
                document=document.text,
                metadata={
                    "technique_id": document.technique_id,
                    "name": document.technique_name or document.technique_id,
                    "field": document.field,
                    "parent_id": "",
                    # BM25 scores are unbounded; they are not cosine similarities.
                    "score_kind": "bm25",
                },
                distance=0.0,
                similarity=score,
            )
            for score, document in scored[: max(1, n_results)]
        ]


def _build_postings(
    documents: tuple[LexicalDocument, ...], average_length: float
) -> tuple[dict[str, tuple[tuple[int, int], ...]], tuple[float, ...]]:
    postings: dict[str, list[tuple[int, int]]] = {}
    norms: list[float] = []
    for index, document in enumerate(documents):
        norms.append(1.2 * (0.25 + 0.75 * len(document.tokens) / max(average_length, 1.0)))
        for token, frequency in Counter(document.tokens).items():
            postings.setdefault(token, []).append((index, frequency))
    return {token: tuple(items) for token, items in postings.items()}, tuple(norms)


@dataclass(frozen=True)
class RetrievalResult:
    matches: tuple[TechniqueMatch, ...]
    observed_techniques: dict[str, float]
    technique_observations: dict[str, tuple[str, ...]]
    semantic_profile_scores: dict[str, float]
    group_text_scores: dict[str, float]
    specific_behavior_scores: dict[str, float]
    specific_behavior_signals: dict[str, tuple[str, ...]]
    specific_behavior_matches: tuple[BehaviorMatch, ...]
    candidates: tuple[TechniqueCandidate, ...] = ()
    mapping_mode: str = "legacy_dense"
    parent_policy: str = "legacy_expand"
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": [match.to_dict() for match in self.matches],
            "observed_techniques": dict(self.observed_techniques),
            "technique_observations": {
                key: list(value) for key, value in self.technique_observations.items()
            },
            "semantic_profile_scores": dict(self.semantic_profile_scores),
            "group_text_scores": dict(self.group_text_scores),
            "specific_behavior_scores": dict(self.specific_behavior_scores),
            "specific_behavior_signals": {
                key: list(value) for key, value in self.specific_behavior_signals.items()
            },
            "specific_behavior_matches": [
                asdict(match) for match in self.specific_behavior_matches
            ],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "mapping_mode": self.mapping_mode,
            "parent_policy": self.parent_policy,
            "diagnostics": dict(self.diagnostics or {}),
        }


# Reranker acceptance thresholds. A reranker score above
# RERANKER_PROMOTE_THRESHOLD promotes a NEEDS_REVIEW candidate to
# AUTO_ACCEPTED. A reranker score below RERANKER_DEMOTE_THRESHOLD
# demotes an AUTO_ACCEPTED candidate to NEEDS_REVIEW. These thresholds
# are conservative — the reranker must be very confident to promote
# and very unconfident to demote.
RERANKER_PROMOTE_THRESHOLD = 0.8
RERANKER_DEMOTE_THRESHOLD = 0.2


def retrieve_techniques(
    ledger: EvidenceLedger,
    store: ChromaVectorStore,
    kb: Any,
    *,
    threshold: float = 0.45,
    top_k: int = 10,
    behavior_threshold: float = 0.55,
    behavior_top_k: int = 20,
    retrieval_mode: str = "hybrid",
    candidate_pool: int = 12,
    include_detection: bool = False,
    parent_policy: str = "ancestor_compatible",
    lexical_index: LexicalTechniqueIndex | None = None,
    acceptance_threshold: float = 0.75,
    relative_margin: float = 0.08,
    reranker: Any = None,
    finalize: bool = True,
    cues_enabled: bool = True,
    acceptance_model: Any = None,
    rrf_k: int = 60,
    corroboration_bonus: float = 0.0,
    segment_acceptance: bool = False,
    hierarchy_parent_mode: str = "off",
    include_synthetic: bool = True,
    rank_by_acceptance: bool = False,
    corroboration_accept_observations: int = 0,
    corroboration_accept_min_probability: float = 0.2,
    exemplar_index: Any = None,
    exemplar_k: int = EXEMPLAR_K,
    exemplar_tau: float = EXEMPLAR_TAU,
    exemplar_duplicate_similarity: float = EXEMPLAR_DUPLICATE_SIMILARITY,
    exemplar_new_candidates: int = 0,
) -> RetrievalResult:
    """Retrieve ATT&CK technique candidates for the positive observations in a ledger.

    ``exemplar_index`` (:class:`apt_classifier.exemplars.ExemplarIndex`) adds what the nearest
    labelled real-report sentences say about each candidate as ``exemplar`` evidence (the
    acceptance model reads it) and appends up to ``exemplar_new_candidates`` techniques that
    only the exemplars suggest. An acceptance model fitted with exemplars is not used when the
    same index is unavailable (diagnostics say why).

    ``cues_enabled=False`` disables the regex cue tables (``_EXPLICIT_TECHNIQUE_CUES``
    and ``technique_signals.SIGNALS``) so benchmarks can report retrieval quality
    without them. Production always runs with cues enabled.

    ``acceptance_model`` is an optional calibrated per-candidate scorer
    (:class:`apt_classifier.acceptance.AcceptanceModel`); when supplied it replaces the
    rule-based one-primary acceptance in :func:`finalize_retrieval_candidates`.

    Observations longer than ``LONG_OBSERVATION_WORDS`` words are split into sentences for
    retrieval and the per-sentence hits are merged per technique, so multi-technique
    paragraphs are not collapsed into one blurred query vector.
    """
    if retrieval_mode not in {"hybrid", "dense", "lexical", "legacy_dense"}:
        raise ValueError("unsupported technique retrieval mode")
    if parent_policy not in {"ancestor_compatible", "exact_only", "legacy_expand"}:
        raise ValueError("unsupported parent policy")
    observations = [
        observation
        for observation in ledger.observations
        if observation.status is EvidenceStatus.POSITIVE
        and observation.kind not in {"ioc", "software"}
    ]
    action_texts = [observation.text for observation in observations]
    dense_enabled = retrieval_mode in {"hybrid", "dense", "legacy_dense"}
    lexical_enabled = retrieval_mode in {"hybrid", "lexical"}
    base_fields = ("name", "description", "procedure") + (
        ("synthetic",) if include_synthetic else ()
    )
    fields = (
        None
        if retrieval_mode == "legacy_dense"
        else (base_fields + ("detection",) if include_detection else base_fields)
    )
    # Long paragraphs are retrieved sentence by sentence (P2.4); segments are flattened
    # for one batched embedding call and regrouped per observation below.
    segments = [_observation_segments(observation.text) for observation in observations]
    flat_segments = [segment for group in segments for segment in group]
    # Over-fetch dense sentence vectors so that duplicate sentences from the
    # same technique do not crowd out unique techniques. The results are
    # aggregated by technique in _best_technique_hits before candidate scoring.
    dense_overfetch = max(top_k * 3, candidate_pool * 4)
    flat_dense = (
        _query_technique_vectors(store, flat_segments, n_results=dense_overfetch, fields=fields)
        if dense_enabled
        else [[] for _ in flat_segments]
    )
    dense_hits = _regroup(flat_dense, segments)
    if lexical_enabled and lexical_index is None:
        lexical_fields = tuple(
            field for field in LEXICAL_DEFAULT_FIELDS if include_synthetic or field != "synthetic"
        )
        lexical_index = LexicalTechniqueIndex.from_kb(
            kb,
            fields=(*lexical_fields, "detection") if include_detection else lexical_fields,
        )
    flat_lexical = (
        [
            lexical_index.query(segment, n_results=max(top_k * 3, candidate_pool * 2))
            for segment in flat_segments
        ]
        if lexical_enabled and lexical_index is not None
        else [[] for _ in flat_segments]
    )
    lexical_hits = _regroup(flat_lexical, segments)
    exemplar_votes, exemplar_status = _exemplar_votes(
        observations,
        store,
        kb,
        exemplar_index,
        k=exemplar_k,
        tau=exemplar_tau,
        duplicate_similarity=exemplar_duplicate_similarity,
    )
    acceptance_disabled: str | None = None
    if acceptance_model is not None and getattr(acceptance_model, "requires_exemplars", False):
        runtime = (
            getattr(exemplar_index, "fingerprint", None) if exemplar_status == "active" else None
        )
        if runtime != getattr(acceptance_model, "exemplar_fingerprint", None):
            acceptance_disabled = (
                "acceptance model needs exemplar index "
                f"{getattr(acceptance_model, 'exemplar_fingerprint', None)}, runtime has "
                f"{runtime or exemplar_status}"
            )
            acceptance_model = None
    exemplar_added = 0
    candidates: list[TechniqueCandidate] = []
    segmented_observations = 0
    for observation_index, (observation, dense_groups, lexical_groups) in enumerate(
        zip(observations, dense_hits, lexical_hits, strict=True)
    ):
        required_platform = _required_platform(observation)
        segment_texts = segments[observation_index]
        segment_count = len(dense_groups)
        segmented_observations += int(segment_count > 1)
        if segment_count <= 1 or not segment_acceptance:
            selected_dense: dict[str, VectorHit] = {}
            for raw_dense in dense_groups:
                _merge_best(
                    selected_dense,
                    _best_technique_hits(
                        raw_dense, threshold, required_platform, max_candidates=max(top_k, 1)
                    ),
                )
            selected_lexical: dict[str, VectorHit] = {}
            for raw_lexical in lexical_groups:
                _merge_best(
                    selected_lexical,
                    _best_lexical_hits(
                        raw_lexical,
                        kb,
                        required_platform=required_platform,
                        max_candidates=max(top_k, 1),
                    ),
                )
            observation_candidates = _build_observation_candidates(
                observation,
                kb,
                selected_dense,
                selected_lexical,
                candidate_pool=_segment_pool(candidate_pool, segment_count),
                cues_enabled=cues_enabled,
                rrf_k=rrf_k,
            )
        else:
            # Each sentence of a long observation gets its own candidate pool, fused ranks
            # and margins (recorded in evidence metadata as segment_index/segment_rank), so a
            # technique described in the second sentence is not judged against the first
            # sentence's best match. Pools are merged per technique afterwards.
            observation_candidates = _build_segmented_candidates(
                observation,
                kb,
                segment_texts,
                dense_groups,
                lexical_groups,
                threshold=threshold,
                required_platform=required_platform,
                top_k=top_k,
                candidate_pool=candidate_pool,
                cues_enabled=cues_enabled,
                rrf_k=rrf_k,
            )
        if cues_enabled:
            observation_candidates = _merge_explicit_candidates(
                observation, kb, observation_candidates
            )
        if reranker is not None:
            observation_candidates = list(
                reranker.rerank(observation.text, observation_candidates, kb)
            )
        observation_candidates = _expand_hierarchy_candidates(
            observation, kb, observation_candidates
        )
        votes = exemplar_votes[observation_index]
        if votes is not None:
            before = len(observation_candidates)
            if exemplar_new_candidates > 0 and (segment_count <= 1 or not segment_acceptance):
                observation_candidates = _append_exemplar_candidates(
                    observation,
                    kb,
                    observation_candidates,
                    votes,
                    limit=int(exemplar_new_candidates),
                    required_platform=required_platform,
                )
            exemplar_added += len(observation_candidates) - before
            observation_candidates = _attach_exemplar_evidence(observation_candidates, votes)
        candidates.extend(observation_candidates)
    behavior_query = getattr(store, "query_group_behaviors", None)
    behavior_hits = (
        behavior_query(action_texts, n_results=behavior_top_k)
        if callable(behavior_query) and action_texts
        else []
    )
    behavior_scores, behavior_signals, behavior_matches = _specific_behavior_results(
        observations,
        behavior_hits,
        threshold=behavior_threshold,
    )
    group_count = max(len(getattr(kb, "groups", {})), 1)
    group_query = getattr(store, "query_group_profile", None)
    semantic_hits = (
        group_query(action_texts, document_type="technique_profile", n_results=group_count)
        if callable(group_query) and action_texts
        else []
    )
    narrative = " ".join(action_texts)
    description_hits = (
        group_query([narrative], document_type="description", n_results=group_count)
        if callable(group_query) and narrative
        else []
    )
    accepted_count = sum(
        candidate.state is TechniqueCandidateState.AUTO_ACCEPTED for candidate in candidates
    )
    retrieval = RetrievalResult(
        matches=(),
        observed_techniques={},
        technique_observations={},
        semantic_profile_scores=_group_scores(semantic_hits),
        group_text_scores=_group_scores(description_hits),
        specific_behavior_scores=behavior_scores,
        specific_behavior_signals=behavior_signals,
        specific_behavior_matches=tuple(behavior_matches),
        candidates=tuple(candidates),
        mapping_mode=retrieval_mode,
        parent_policy=parent_policy,
        diagnostics={
            "observation_count": len(observations),
            "candidate_count": len(candidates),
            "auto_accepted_count": accepted_count,
            "needs_review_count": len(candidates) - accepted_count,
            "dense_enabled": dense_enabled,
            "lexical_enabled": lexical_enabled,
            "cues_enabled": cues_enabled,
            "segmented_observations": segmented_observations,
            "acceptance_model": acceptance_model is not None,
            "include_detection": include_detection,
            "reranker_enabled": reranker is not None,
            "reranker_available": bool(getattr(reranker, "available", False)),
            "reranker_load_error": getattr(reranker, "load_error", None),
            "exemplars": exemplar_status,
            "exemplar_candidates_added": exemplar_added,
            **({"acceptance_model_disabled": acceptance_disabled} if acceptance_disabled else {}),
        },
    )
    if not finalize:
        return retrieval
    return finalize_retrieval_candidates(
        retrieval,
        ledger,
        kb,
        retrieval_mode=retrieval_mode,
        threshold=threshold,
        acceptance_threshold=acceptance_threshold,
        relative_margin=relative_margin,
        reranker=reranker,
        acceptance_model=acceptance_model,
        corroboration_bonus=corroboration_bonus,
        segment_acceptance=segment_acceptance,
        hierarchy_parent_mode=hierarchy_parent_mode,
        rank_by_acceptance=rank_by_acceptance,
        corroboration_accept_observations=corroboration_accept_observations,
        corroboration_accept_min_probability=corroboration_accept_min_probability,
    )


LONG_OBSERVATION_WORDS = 40
MAX_SEGMENT_POOL = 30
EXEMPLAR_TEXT_LIMIT = 300


def _exemplar_votes(
    observations: list[Any],
    store: Any,
    kb: Any,
    exemplar_index: Any,
    *,
    k: int,
    tau: float,
    duplicate_similarity: float,
) -> tuple[list[ExemplarVotes | None], str]:
    """Exemplar votes per observation (``None`` entries when the index is unavailable)."""
    empty: list[ExemplarVotes | None] = [None] * len(observations)
    if exemplar_index is None:
        return empty, "off"
    if not observations:
        return empty, "active"
    embedder = getattr(store, "embedder", None)
    if embedder is None or not callable(getattr(embedder, "embed", None)):
        return empty, "no embedder on the store"
    vectors = embedder.embed(
        [exemplar_query_text(str(observation.text)) for observation in observations]
    )
    votes = exemplar_index.query_vectors(
        vectors,
        parent_of=exemplar_parent_map(kb),
        k=k,
        tau=tau,
        duplicate_similarity=duplicate_similarity,
    )
    return list(votes), "active"


def _exemplar_evidence(
    votes: ExemplarVotes, technique_id: str, *, exemplar_only: bool
) -> TechniqueRetrievalEvidence:
    features = votes.features(technique_id, exemplar_only=exemplar_only)
    rank = votes.ranks.get(technique_id)
    count = votes.counts.get(technique_id, 0)
    text = votes.best_text.get(technique_id)
    return TechniqueRetrievalEvidence(
        source=EXEMPLAR_SOURCE,
        field="labelled_report_sentence",
        text=text[:EXEMPLAR_TEXT_LIMIT] if text else None,
        score=features["ex_vote"],
        rank=rank,
        rationale=(
            f"{count} of the {votes.neighbours} most similar labelled report sentences carry "
            "this technique"
            if rank
            else "no similar labelled report sentence carries this technique"
        ),
        metadata={
            "features": features,
            "source_dataset": votes.best_source.get(technique_id),
            "top_similarity": round(votes.top_similarity, 4),
            "excluded_duplicates": votes.excluded_duplicates,
            "exemplar_only": exemplar_only,
        },
    )


def _append_exemplar_candidates(
    observation: Any,
    kb: Any,
    candidates: list[TechniqueCandidate],
    votes: ExemplarVotes,
    *,
    limit: int,
    required_platform: str | None,
) -> list[TechniqueCandidate]:
    """Techniques only the exemplars suggest, appended after the retrieved pool (fused score
    0, so they never displace a retrieved candidate); the acceptance model judges them on the
    exemplar evidence alone."""
    techniques = getattr(kb, "techniques", {}) or {}
    present = {candidate.technique_id for candidate in candidates}
    output = list(candidates)
    next_rank = len(candidates) + 1
    for technique_id in votes.ranked:
        if len(output) - len(candidates) >= limit:
            break
        if technique_id in present or technique_id not in techniques:
            continue
        technique = techniques[technique_id]
        if not _platform_matches(_record_value(technique, "platforms", ()), required_platform):
            continue
        output.append(
            TechniqueCandidate(
                observation_id=observation.id,
                technique_id=technique_id,
                technique_name=_technique_name(kb, technique_id),
                parent_technique_id=str(_record_value(technique, "parent_id", "") or "") or None,
                is_subtechnique=bool(_record_value(technique, "is_subtechnique", False)),
                fused_score=0.0,
                fused_rank=next_rank,
                evidence=[_exemplar_evidence(votes, technique_id, exemplar_only=True)],
                state=TechniqueCandidateState.NEEDS_REVIEW,
                state_reason="exemplar-only candidate; acceptance not yet evaluated",
            )
        )
        present.add(technique_id)
        next_rank += 1
    return output


def _attach_exemplar_evidence(
    candidates: list[TechniqueCandidate], votes: ExemplarVotes
) -> list[TechniqueCandidate]:
    output = []
    for candidate in candidates:
        if candidate.evidence_for(EXEMPLAR_SOURCE) is not None:
            output.append(candidate)
            continue
        evidence = _exemplar_evidence(votes, candidate.technique_id, exemplar_only=False)
        output.append(candidate.model_copy(update={"evidence": [*candidate.evidence, evidence]}))
    return output


def _observation_segments(text: str) -> list[str]:
    """Split long observations into sentences for retrieval; short ones stay whole."""
    if len(text.split()) <= LONG_OBSERVATION_WORDS:
        return [text]
    sentences = [part for part in split_sentences(text) if len(part.split()) >= 3]
    return sentences if len(sentences) > 1 else [text]


def _regroup(flat: list[list[VectorHit]], segments: list[list[str]]) -> list[list[list[VectorHit]]]:
    grouped: list[list[list[VectorHit]]] = []
    index = 0
    for group in segments:
        grouped.append(flat[index : index + len(group)])
        index += len(group)
    return grouped


def _merge_best(target: dict[str, VectorHit], hits: dict[str, VectorHit]) -> None:
    for technique_id, hit in hits.items():
        current = target.get(technique_id)
        if current is None or hit.similarity > current.similarity:
            target[technique_id] = hit


def _segment_pool(candidate_pool: int, segment_count: int) -> int:
    if segment_count <= 1:
        return candidate_pool
    return min(MAX_SEGMENT_POOL, candidate_pool + 4 * (segment_count - 1))


def _segment_view(observation: Any, text: str) -> SimpleNamespace:
    """A per-sentence stand-in for the observation (same id/provenance, segment text)."""
    return SimpleNamespace(
        id=observation.id,
        text=text,
        provenance=getattr(observation, "provenance", None),
        kind=getattr(observation, "kind", None),
        metadata=getattr(observation, "metadata", {}),
        status=getattr(observation, "status", None),
    )


def _tag_segment(
    candidate: TechniqueCandidate, index: int, rank: int, text: str
) -> TechniqueCandidate:
    return candidate.model_copy(
        update={
            "evidence": [
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "segment_index": index,
                            "segment_rank": rank,
                            "segment_text": text[:300],
                        }
                    }
                )
                for item in candidate.evidence
            ]
        }
    )


def _build_segmented_candidates(
    observation: Any,
    kb: Any,
    segment_texts: list[str],
    dense_groups: list[list[VectorHit]],
    lexical_groups: list[list[VectorHit]],
    *,
    threshold: float,
    required_platform: str | None,
    top_k: int,
    candidate_pool: int,
    cues_enabled: bool,
    rrf_k: int,
) -> list[TechniqueCandidate]:
    merged: dict[str, TechniqueCandidate] = {}
    support: dict[str, int] = {}
    for index, (text, raw_dense, raw_lexical) in enumerate(
        zip(segment_texts, dense_groups, lexical_groups, strict=True)
    ):
        view = _segment_view(observation, text)
        dense = _best_technique_hits(
            raw_dense, threshold, required_platform, max_candidates=max(top_k, 1)
        )
        lexical = _best_lexical_hits(
            raw_lexical, kb, required_platform=required_platform, max_candidates=max(top_k, 1)
        )
        for candidate in _build_observation_candidates(
            view,
            kb,
            dense,
            lexical,
            candidate_pool=candidate_pool,
            cues_enabled=cues_enabled,
            rrf_k=rrf_k,
        ):
            tagged = _tag_segment(candidate, index, candidate.fused_rank, text)
            support[candidate.technique_id] = support.get(candidate.technique_id, 0) + 1
            current = merged.get(candidate.technique_id)
            if current is None or tagged.fused_score > current.fused_score:
                merged[candidate.technique_id] = tagged
    ordered = sorted(merged.values(), key=lambda item: (-item.fused_score, item.technique_id))[
        : _segment_pool(candidate_pool, len(segment_texts))
    ]
    output: list[TechniqueCandidate] = []
    for rank, candidate in enumerate(ordered, 1):
        evidence = list(candidate.evidence)
        if evidence:
            evidence[0] = evidence[0].model_copy(
                update={
                    "metadata": {
                        **evidence[0].metadata,
                        "segment_support": support.get(candidate.technique_id, 1),
                    }
                }
            )
        output.append(candidate.model_copy(update={"fused_rank": rank, "evidence": evidence}))
    return output


def _segment_index(candidate: TechniqueCandidate) -> int | None:
    for item in candidate.evidence:
        value = item.metadata.get("segment_index")
        if value is not None:
            return int(value)
    return None


def _segment_rank(candidate: TechniqueCandidate) -> int | None:
    for item in candidate.evidence:
        value = item.metadata.get("segment_rank")
        if value is not None:
            return int(value)
    return None


def _segment_text(group: list[TechniqueCandidate]) -> str | None:
    for candidate in group:
        for item in candidate.evidence:
            text = item.metadata.get("segment_text")
            if text:
                return str(text)
    return None


def acceptance_views(
    observation: Any,
    candidates: list[TechniqueCandidate],
    *,
    segment_acceptance: bool = False,
) -> list[tuple[Any, list[TechniqueCandidate], list[TechniqueCandidate]]]:
    """Return ``(observation_view, candidate_views, originals)`` per judged group.

    This is exactly how ``_assign_candidate_states`` sees candidates (per sentence for long
    observations, with segment ranks), so training-time feature extraction for the
    acceptance model (``scripts/fit_acceptance_model.py``) cannot drift from inference.
    """
    groups = _segment_groups(candidates) if segment_acceptance else {0: list(candidates)}
    if len(groups) <= 1:
        return [(observation, list(candidates), list(candidates))]
    output = []
    for _, group in sorted(groups.items()):
        view = _segment_view(observation, _segment_text(group) or str(observation.text))
        views = [
            candidate.model_copy(
                update={"fused_rank": _segment_rank(candidate) or candidate.fused_rank}
            )
            for candidate in group
        ]
        output.append((view, views, group))
    return output


def _segment_groups(candidates: list[TechniqueCandidate]) -> dict[int, list[TechniqueCandidate]]:
    """Group candidates by the sentence they came from; derived parents follow their children,
    other untagged candidates (e.g. cue injections over the whole text) join segment 0."""
    groups: dict[int, list[TechniqueCandidate]] = {}
    untagged: list[TechniqueCandidate] = []
    child_segment: dict[str, list[int]] = {}
    for candidate in candidates:
        index = _segment_index(candidate)
        if index is None:
            untagged.append(candidate)
            continue
        groups.setdefault(index, []).append(candidate)
        if candidate.parent_technique_id:
            child_segment.setdefault(candidate.parent_technique_id, []).append(index)
    for candidate in untagged:
        indices = child_segment.get(candidate.technique_id, [])
        index = max(set(indices), key=indices.count) if indices else min(groups, default=0)
        groups.setdefault(index, []).append(candidate)
    return groups


def finalize_retrieval_candidates(
    retrieval: RetrievalResult,
    ledger: EvidenceLedger,
    kb: Any,
    *,
    retrieval_mode: str,
    threshold: float,
    acceptance_threshold: float,
    relative_margin: float,
    reranker: Any = None,
    acceptance_model: Any = None,
    corroboration_bonus: float = 0.0,
    segment_acceptance: bool = False,
    hierarchy_parent_mode: str = "off",
    rank_by_acceptance: bool = False,
    corroboration_accept_observations: int = 0,
    corroboration_accept_min_probability: float = 0.2,
) -> RetrievalResult:
    """Assign mapping states only after all candidate evidence is available.

    Retrieval, explicit/structured signals, hierarchy expansion, reranking, and
    optional verifier output are deliberately collected before this boundary.
    This prevents an early retrieval acceptance from bypassing a later
    verifier rejection.  The helper is also used by local evaluation with no
    verifier, keeping both paths on the same acceptance implementation.
    """
    observations = {observation.id: observation for observation in ledger.observations}
    grouped: dict[str, list[TechniqueCandidate]] = {}
    for candidate in retrieval.candidates:
        grouped.setdefault(candidate.observation_id, []).append(candidate)
    finalized: list[TechniqueCandidate] = []
    for observation_id, candidates in grouped.items():
        observation = observations.get(observation_id)
        if observation is None:
            finalized.extend(candidates)
            continue
        finalized.extend(
            _assign_candidate_states(
                observation,
                candidates,
                retrieval_mode=retrieval_mode,
                threshold=threshold,
                acceptance_threshold=acceptance_threshold,
                relative_margin=relative_margin,
                reranker=reranker,
                acceptance_model=acceptance_model,
                segment_acceptance=segment_acceptance,
                hierarchy_parent_mode=hierarchy_parent_mode,
            )
        )
    corroborated = 0
    if corroboration_accept_observations > 0 and acceptance_model is not None:
        finalized, corroborated = accept_by_corroboration(
            finalized,
            min_observations=corroboration_accept_observations,
            min_probability=corroboration_accept_min_probability,
        )
    reordered = bool(rank_by_acceptance and acceptance_model is not None)
    if reordered:
        finalized = rank_candidates_by_acceptance(finalized)
    matches, observed, observation_ids = _materialize_candidates(
        finalized,
        kb,
        parent_policy=retrieval.parent_policy,
        corroboration_bonus=corroboration_bonus,
    )
    _apply_positive_answers(ledger, kb, observed, observation_ids, matches)
    ledger.technique_matches = matches
    diagnostics = dict(retrieval.diagnostics or {})
    diagnostics.update(
        {
            "acceptance_finalized": True,
            "acceptance_model_active": acceptance_model is not None,
            "rank_by_acceptance": reordered,
            "corroboration_accepted": corroborated,
            "auto_accepted_count": sum(
                candidate.state is TechniqueCandidateState.AUTO_ACCEPTED for candidate in finalized
            ),
            "needs_review_count": sum(
                candidate.state is TechniqueCandidateState.NEEDS_REVIEW for candidate in finalized
            ),
            "auto_rejected_count": sum(
                candidate.state is TechniqueCandidateState.AUTO_REJECTED for candidate in finalized
            ),
        }
    )
    return replace(
        retrieval,
        matches=tuple(matches),
        observed_techniques=observed,
        technique_observations={
            key: tuple(sorted(value)) for key, value in observation_ids.items()
        },
        candidates=tuple(finalized),
        diagnostics=diagnostics,
    )


def apply_candidate_decisions(
    retrieval: RetrievalResult,
    ledger: EvidenceLedger,
    kb: Any,
) -> RetrievalResult:
    decisions = {
        (decision.observation_id, decision.technique_id): decision
        for decision in ledger.technique_reviews
    }
    effective: list[TechniqueCandidate] = []
    for candidate in retrieval.candidates:
        decision = decisions.get((candidate.observation_id, candidate.technique_id))
        if decision is None:
            effective.append(candidate)
            continue
        if decision.action is TechniqueReviewAction.REJECT:
            effective.append(
                candidate.model_copy(
                    update={
                        "state": TechniqueCandidateState.AUTO_REJECTED,
                        "state_reason": "analyst rejected mapping",
                    }
                )
            )
            continue
        if decision.action is TechniqueReviewAction.CONFIRM:
            effective.append(
                candidate.model_copy(
                    update={
                        "state": TechniqueCandidateState.AUTO_ACCEPTED,
                        "state_reason": "analyst confirmed mapping",
                    }
                )
            )
            continue
        replacement_id = str(decision.replacement_technique_id or "")
        technique = getattr(kb, "techniques", {}).get(replacement_id)
        if technique is None:
            effective.append(candidate)
            continue
        effective.append(
            TechniqueCandidate(
                observation_id=candidate.observation_id,
                technique_id=replacement_id,
                technique_name=_technique_name(kb, replacement_id),
                parent_technique_id=str(_record_value(technique, "parent_id", "") or "") or None,
                is_subtechnique=bool(_record_value(technique, "is_subtechnique", False)),
                fused_score=candidate.fused_score,
                fused_rank=candidate.fused_rank,
                evidence=[
                    *candidate.evidence,
                    TechniqueRetrievalEvidence(
                        source="analyst_replacement",
                        text=decision.rationale,
                        score=1.0,
                        rank=1,
                    ),
                ],
                state=TechniqueCandidateState.AUTO_ACCEPTED,
                state_reason=f"analyst replaced {candidate.technique_id}",
            )
        )
    matches, observed, observation_ids = _materialize_candidates(
        effective,
        kb,
        parent_policy=retrieval.parent_policy,
    )
    _apply_positive_answers(ledger, kb, observed, observation_ids, matches)
    ledger.technique_matches = matches
    diagnostics = dict(retrieval.diagnostics or {})
    diagnostics["analyst_review_count"] = len(ledger.technique_reviews)
    diagnostics["effective_technique_count"] = len(observed)
    return replace(
        retrieval,
        matches=tuple(matches),
        observed_techniques=observed,
        technique_observations={
            technique_id: tuple(sorted(ids)) for technique_id, ids in observation_ids.items()
        },
        diagnostics=diagnostics,
    )


def _query_technique_vectors(
    store: Any,
    texts: list[str],
    *,
    n_results: int,
    fields: tuple[str, ...] | None,
) -> list[list[VectorHit]]:
    if not texts:
        return []
    try:
        return store.query_techniques(texts, n_results=n_results, fields=fields)
    except TypeError:
        return store.query_techniques(texts, n_results=n_results)


def _best_lexical_hits(
    hits: list[VectorHit],
    kb: Any,
    *,
    required_platform: str | None,
    max_candidates: int,
) -> dict[str, VectorHit]:
    output: dict[str, VectorHit] = {}
    for hit in hits:
        technique_id = str(hit.metadata.get("technique_id") or "")
        technique = getattr(kb, "techniques", {}).get(technique_id)
        platforms = _record_value(technique, "platforms", ())
        if not technique_id or not _platform_matches(platforms, required_platform):
            continue
        if technique_id not in output or hit.similarity > output[technique_id].similarity:
            output[technique_id] = hit
        if len(output) >= max_candidates:
            break
    return output


def _build_observation_candidates(
    observation: Any,
    kb: Any,
    dense: dict[str, VectorHit],
    lexical: dict[str, VectorHit],
    *,
    candidate_pool: int,
    cues_enabled: bool = True,
    rrf_k: int = 60,
) -> list[TechniqueCandidate]:
    dense_order = sorted(
        dense,
        key=lambda technique_id: (
            -dense[technique_id].similarity * _field_weight(dense[technique_id]),
            technique_id,
        ),
    )
    lexical_order = sorted(
        lexical,
        key=lambda technique_id: (-lexical[technique_id].similarity, technique_id),
    )
    dense_ranks = {technique_id: rank for rank, technique_id in enumerate(dense_order, 1)}
    lexical_ranks = {technique_id: rank for rank, technique_id in enumerate(lexical_order, 1)}
    fused_scores: dict[str, float] = {}
    for technique_id, rank in dense_ranks.items():
        fused_scores[technique_id] = fused_scores.get(technique_id, 0.0) + 1.0 / (rrf_k + rank)
    for technique_id, rank in lexical_ranks.items():
        fused_scores[technique_id] = fused_scores.get(technique_id, 0.0) + 1.0 / (rrf_k + rank)
    observation_text_lower = observation.text.casefold()
    dense_top3 = set(dense_order[:3])
    lexical_top3_set = set(lexical_order[:3])
    for technique_id in dense_top3 & lexical_top3_set:
        fused_scores[technique_id] = fused_scores.get(technique_id, 0.0) + 0.005
    for technique_id in list(fused_scores):
        name_bonus = _name_token_bonus(kb, technique_id, observation_text_lower)
        if name_bonus > 0:
            fused_scores[technique_id] = fused_scores.get(technique_id, 0.0) + name_bonus
    signal_map: dict[str, tuple[Any, str]] = {}
    for signal, matched_span in find_signals(observation.text) if cues_enabled else ():
        if signal.technique_id not in getattr(kb, "techniques", {}):
            continue
        signal_map.setdefault(signal.technique_id, (signal, matched_span))
        fused_scores[signal.technique_id] = fused_scores.get(signal.technique_id, 0.0) + 1.0 / 61
    ordered = sorted(fused_scores, key=lambda item: (-fused_scores[item], item))[
        : max(1, candidate_pool)
    ]
    candidates: list[TechniqueCandidate] = []
    for fused_rank, technique_id in enumerate(ordered, 1):
        technique = getattr(kb, "techniques", {}).get(technique_id)
        evidence: list[TechniqueRetrievalEvidence] = []
        dense_hit = dense.get(technique_id)
        if dense_hit is not None:
            evidence.append(
                TechniqueRetrievalEvidence(
                    source="dense",
                    document_id=dense_hit.id,
                    field=str(dense_hit.metadata.get("field") or "") or None,
                    text=dense_hit.document,
                    score=dense_hit.similarity * _source_reliability(observation, dense_hit),
                    rank=dense_ranks[technique_id],
                )
            )
        lexical_hit = lexical.get(technique_id)
        if lexical_hit is not None:
            evidence.append(
                TechniqueRetrievalEvidence(
                    source="lexical",
                    document_id=lexical_hit.id,
                    field=str(lexical_hit.metadata.get("field") or "") or None,
                    text=lexical_hit.document,
                    score=lexical_hit.similarity,
                    rank=lexical_ranks[technique_id],
                )
            )
        signal_match = signal_map.get(technique_id)
        if signal_match is not None:
            signal, signal_span = signal_match
            evidence.append(
                TechniqueRetrievalEvidence(
                    source="structured_signal",
                    text=signal_span,
                    rationale=signal.rationale,
                    score=0.85,
                    rank=1,
                    metadata={"requires_context": signal.requires_context},
                )
            )
        state = TechniqueCandidateState.NEEDS_REVIEW
        reason = "candidate evidence collected; acceptance not yet evaluated"
        candidates.append(
            TechniqueCandidate(
                observation_id=observation.id,
                technique_id=technique_id,
                technique_name=_technique_name(kb, technique_id),
                parent_technique_id=str(_record_value(technique, "parent_id", "") or "") or None,
                is_subtechnique=bool(_record_value(technique, "is_subtechnique", False)),
                fused_score=fused_scores[technique_id],
                fused_rank=fused_rank,
                evidence=evidence,
                state=state,
                state_reason=reason,
            )
        )
    return candidates


# Similarity alone is unreliable on near-content-free text (a bare IP address,
# a lone field name).  Pure-retrieval acceptance requires at least this many
# distinct alphabetic words in the observation; cue-supported and verifier-
# supported mappings are unaffected.
MIN_SIMILARITY_ONLY_WORDS = 3
_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _distinct_word_count(text: str) -> int:
    return len({match.group(0).casefold() for match in _WORD_RE.finditer(text)})


def _passes_similarity_gate(
    candidate: TechniqueCandidate,
    *,
    dense_floor: float,
    dense_margin: float,
    acceptance_threshold: float,
    relative_margin: float,
) -> bool:
    """Retrieval-only acceptance test for the fused rank-1 candidate.

    Accept when the dense similarity is strong on its own, when it is the best dense hit
    with a clear margin over the runner-up, or when lexical retrieval independently
    ranks the same technique highly.
    """
    dense = candidate.evidence_for("dense")
    lexical = candidate.evidence_for("lexical")
    dense_score = float(dense.score or 0.0) if dense is not None else 0.0
    strong_dense = dense_score >= acceptance_threshold
    margin_supported = (
        dense is not None
        and dense.rank == 1
        and dense_score >= dense_floor
        and dense_margin >= relative_margin
    )
    cross_validated = (
        dense is not None
        and lexical is not None
        and dense_score >= dense_floor
        and (lexical.rank or 999) <= 5
    )
    return strong_dense or margin_supported or cross_validated


ACCEPTANCE_EVIDENCE_SOURCE = "acceptance_model"


def _acceptance_probability(
    model: Any, candidate: TechniqueCandidate, features: dict[str, float]
) -> tuple[float | None, bool]:
    """Return ``(probability, accepted)`` from the calibrated acceptance model.

    A probability already stored on the candidate (from an earlier finalization pass) is
    reused so that the second pass after verification, or after the candidates were
    reordered by probability, judges exactly the same number: features such as the
    fused rank and margins are only meaningful in the fusion order they were fitted on.
    """
    threshold = getattr(model, "threshold", None)
    if hasattr(model, "prior_features"):
        features = {**features, **model.prior_features(candidate.technique_id)}
    stored = candidate.evidence_for(ACCEPTANCE_EVIDENCE_SOURCE)
    if stored is not None and stored.score is not None and threshold is not None:
        probability = float(stored.score)
        return probability, probability >= float(threshold)
    if hasattr(model, "probability") and threshold is not None:
        probability = float(model.probability(features))
        return probability, probability >= float(threshold)
    return None, bool(model.accepts(features))


def accept_by_corroboration(
    candidates: Sequence[TechniqueCandidate],
    *,
    min_observations: int,
    min_probability: float,
) -> tuple[list[TechniqueCandidate], int]:
    """Accept a technique the gate left in review when several independent observations of
    the same case each retrieved it with a calibrated probability of at least
    ``min_probability``. One statement mentioning a technique weakly is a near miss; the same
    technique surfacing in ``min_observations`` different statements is corroboration. Only
    NEEDS_REVIEW candidates without a hierarchy-parent origin qualify; benign/sparse-context
    guards stay in force because those candidates carry no stored probability above the floor
    only when the model actually scored them."""
    if min_observations <= 0:
        return list(candidates), 0
    support: dict[str, set[str]] = {}
    for candidate in candidates:
        if candidate.state is not TechniqueCandidateState.NEEDS_REVIEW:
            continue
        if candidate.evidence_for("hierarchy_parent") is not None:
            continue
        evidence = candidate.evidence_for(ACCEPTANCE_EVIDENCE_SOURCE)
        if evidence is None or evidence.score is None or float(evidence.score) < min_probability:
            continue
        support.setdefault(candidate.technique_id, set()).add(candidate.observation_id)
    qualifying = {tid for tid, ids in support.items() if len(ids) >= min_observations}
    if not qualifying:
        return list(candidates), 0
    output: list[TechniqueCandidate] = []
    accepted = 0
    for candidate in candidates:
        evidence = candidate.evidence_for(ACCEPTANCE_EVIDENCE_SOURCE)
        if (
            candidate.state is TechniqueCandidateState.NEEDS_REVIEW
            and candidate.technique_id in qualifying
            and candidate.evidence_for("hierarchy_parent") is None
            and evidence is not None
            and evidence.score is not None
            and float(evidence.score) >= min_probability
        ):
            accepted += 1
            output.append(
                candidate.model_copy(
                    update={
                        "state": TechniqueCandidateState.AUTO_ACCEPTED,
                        "state_reason": (
                            f"corroborated across {len(support[candidate.technique_id])} "
                            "observations (each above the corroboration probability floor)"
                        ),
                    }
                )
            )
        else:
            output.append(candidate)
    return output, accepted


def rank_candidates_by_acceptance(
    candidates: Sequence[TechniqueCandidate],
) -> list[TechniqueCandidate]:
    """Order each observation's candidates by calibrated acceptance probability.

    The fused RRF order decides which candidates enter the pool; the acceptance model was
    fitted on exactly the question "is this candidate a correct mapping", so its probability
    is a better ranking key than the raw fusion score. Candidates the model did not score
    (derived hierarchy parents, verifier-decided candidates without a stored probability)
    follow in fused order. ``fused_rank`` is rewritten so every consumer sees the new order.
    """
    grouped: dict[str, list[TechniqueCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.observation_id, []).append(candidate)

    def key(item: TechniqueCandidate) -> tuple[int, float, float, str]:
        evidence = item.evidence_for(ACCEPTANCE_EVIDENCE_SOURCE)
        if evidence is not None and evidence.score is not None:
            return (0, -float(evidence.score), -float(item.fused_score), item.technique_id)
        return (1, 0.0, -float(item.fused_score), item.technique_id)

    output: list[TechniqueCandidate] = []
    for group in grouped.values():
        ordered = sorted(group, key=key)
        output.extend(
            item.model_copy(update={"fused_rank": rank}) for rank, item in enumerate(ordered, 1)
        )
    return output


def _assign_candidate_states(
    observation: Any,
    candidates: list[TechniqueCandidate],
    *,
    retrieval_mode: str,
    threshold: float,
    acceptance_threshold: float,
    relative_margin: float,
    reranker: Any = None,
    acceptance_model: Any = None,
    segment_acceptance: bool = False,
    hierarchy_parent_mode: str = "off",
) -> list[TechniqueCandidate]:
    """Assign AUTO_ACCEPTED / NEEDS_REVIEW / AUTO_REJECTED per candidate.

    Long observations were retrieved sentence by sentence; their candidates are judged
    per sentence (own fused ranks, margins, benign/sparse checks) and then returned in
    the observation's merged order. Hierarchy consistency is applied within each group.
    """
    judged = acceptance_views(observation, candidates, segment_acceptance=segment_acceptance)
    if len(judged) == 1:
        return _assign_states_for_group(
            observation,
            candidates,
            retrieval_mode=retrieval_mode,
            threshold=threshold,
            acceptance_threshold=acceptance_threshold,
            relative_margin=relative_margin,
            reranker=reranker,
            acceptance_model=acceptance_model,
            hierarchy_parent_mode=hierarchy_parent_mode,
        )
    decided: dict[str, TechniqueCandidate] = {}
    for view, views, group in judged:
        results = _assign_states_for_group(
            view,
            views,
            retrieval_mode=retrieval_mode,
            threshold=threshold,
            acceptance_threshold=acceptance_threshold,
            relative_margin=relative_margin,
            reranker=reranker,
            acceptance_model=acceptance_model,
            hierarchy_parent_mode=hierarchy_parent_mode,
        )
        for original, result in zip(group, results, strict=True):
            decided[original.technique_id] = original.model_copy(
                update={
                    "state": result.state,
                    "state_reason": result.state_reason,
                    "evidence": list(result.evidence),
                }
            )
    return [decided.get(candidate.technique_id, candidate) for candidate in candidates]


def _assign_states_for_group(
    observation: Any,
    candidates: list[TechniqueCandidate],
    *,
    retrieval_mode: str,
    threshold: float,
    acceptance_threshold: float,
    relative_margin: float,
    reranker: Any = None,
    acceptance_model: Any = None,
    hierarchy_parent_mode: str = "off",
) -> list[TechniqueCandidate]:
    dense_scores = sorted(
        (
            float(evidence.score)
            for candidate in candidates
            if (evidence := candidate.evidence_for("dense")) is not None
            and evidence.score is not None
        ),
        reverse=True,
    )
    best_dense = dense_scores[0] if dense_scores else 0.0
    second_dense = dense_scores[1] if len(dense_scores) > 1 else 0.0
    dense_margin = best_dense - second_dense
    dense_floor = max(threshold, acceptance_threshold - 0.15)
    # Similarity-only acceptance is reserved for the fused rank-1 candidate. Earlier logic
    # chose the best *qualifying* candidate instead, which promoted rank 2-3 whenever
    # rank 1 was cue-supported or its only dense hit came from a procedure sentence; on
    # the held-out public benchmark that was worse than simply accepting rank 1. Now a
    # cue-supported rank 1 consumes the slot, a failing rank 1 means abstention, and
    # procedure-field hits are judged by the same score rules as other fields.
    primary_id: str | None = None
    if retrieval_mode != "lexical":
        ranked = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.evidence_for("hierarchy_parent") is None
            ),
            key=lambda item: (item.fused_rank, item.technique_id),
        )
        if ranked:
            top = ranked[0]
            top_is_cue = (
                top.evidence_for("structured_signal") is not None
                or top.evidence_for("explicit_cue") is not None
            )
            if not top_is_cue and _passes_similarity_gate(
                top,
                dense_floor=dense_floor,
                dense_margin=dense_margin,
                acceptance_threshold=acceptance_threshold,
                relative_margin=relative_margin,
            ):
                primary_id = top.technique_id
    promote_value = getattr(reranker, "promote_threshold", None)
    demote_value = getattr(reranker, "demote_threshold", None)
    promote_threshold = (
        float(promote_value) if promote_value is not None else RERANKER_PROMOTE_THRESHOLD
    )
    demote_threshold = (
        float(demote_value) if demote_value is not None else RERANKER_DEMOTE_THRESHOLD
    )
    reranker_validated = bool(reranker is not None and getattr(reranker, "validated", False))
    observation_text = str(getattr(observation, "text", ""))
    benign_context = _BENIGN_INDICATORS.search(observation_text) is not None
    sparse_context = _distinct_word_count(observation_text) < MIN_SIMILARITY_ONLY_WORDS
    output: list[TechniqueCandidate] = []
    for candidate in candidates:
        verification = candidate.verification
        if candidate.evidence_for("hierarchy_parent") is not None:
            output.append(
                candidate.model_copy(
                    update={
                        "state": TechniqueCandidateState.NEEDS_REVIEW,
                        "state_reason": "hierarchy relationship is not direct observation evidence",
                    }
                )
            )
            continue
        if verification.verdict is TechniqueVerificationVerdict.UNSUPPORTED:
            output.append(
                candidate.model_copy(
                    update={
                        "state": TechniqueCandidateState.AUTO_REJECTED,
                        "state_reason": "OpenRouter verifier: unsupported",
                    }
                )
            )
            continue
        if verification.verdict is TechniqueVerificationVerdict.SUPPORTED:
            output.append(
                candidate.model_copy(
                    update={
                        "state": TechniqueCandidateState.AUTO_ACCEPTED,
                        "state_reason": "OpenRouter verifier: supported",
                    }
                )
            )
            continue
        dense = candidate.evidence_for("dense")
        lexical = candidate.evidence_for("lexical")
        signal = candidate.evidence_for("structured_signal")
        explicit = candidate.evidence_for("explicit_cue")
        dense_score = float(dense.score or 0.0) if dense is not None else 0.0
        lexical_support = lexical is not None and (lexical.rank or 999) <= 3
        cue_supported = (signal is not None or explicit is not None) and (
            dense_score >= threshold or lexical_support
        )
        # Signals and explicit cues propose a mapping but can never accept it
        # alone: each needs dense or lexical corroboration.  A benign-context
        # observation never hides candidates, but similarity alone is not
        # enough to auto-accept the primary retrieval candidate there; cue-
        # supported and verifier-supported mappings are unaffected.
        extra_evidence: list[TechniqueRetrievalEvidence] = []
        if acceptance_model is not None:
            # Calibrated multi-label acceptance: every candidate is scored on its own
            # evidence; benign/sparse contexts still forbid similarity-only acceptance.
            features = candidate_features(
                candidate,
                candidates,
                benign_context=benign_context,
                sparse_context=sparse_context,
            )
            probability, model_says_yes = _acceptance_probability(
                acceptance_model, candidate, features
            )
            if (
                probability is not None
                and candidate.evidence_for(ACCEPTANCE_EVIDENCE_SOURCE) is None
            ):
                extra_evidence.append(
                    TechniqueRetrievalEvidence(
                        source=ACCEPTANCE_EVIDENCE_SOURCE,
                        field="probability",
                        text=f"calibrated probability {probability:.2f} that this mapping is correct",
                        score=probability,
                        rationale="calibrated acceptance model",
                    )
                )
            model_accepted = model_says_yes and not benign_context and not sparse_context
            accepted = model_accepted or cue_supported
            reason = (
                "structured or explicit behavior supported by retrieval"
                if cue_supported
                else "calibrated acceptance model score passed the dev-selected threshold"
                if model_accepted
                else "benign-context observation: similarity-only mapping requires review"
                if benign_context and model_says_yes
                else "observation too sparse for similarity-only acceptance"
                if sparse_context and model_says_yes
                else "calibrated acceptance model score below threshold; requires review"
            )
        else:
            primary = (
                candidate.technique_id == primary_id and not benign_context and not sparse_context
            )
            accepted = primary or cue_supported
            reason = (
                "structured or explicit behavior supported by retrieval"
                if cue_supported
                else "primary retrieval candidate passed calibrated evidence gating"
                if primary
                else "benign-context observation: similarity-only mapping requires review"
                if candidate.technique_id == primary_id and benign_context
                else "observation too sparse for similarity-only acceptance"
                if candidate.technique_id == primary_id
                else "candidate requires verifier or analyst review"
            )
        if reranker_validated:
            reranker_evidence = candidate.evidence_for("reranker")
            reranker_score = (
                float(reranker_evidence.score)
                if reranker_evidence is not None and reranker_evidence.score is not None
                else None
            )
            # The reranker may only *narrow* what the guards allow: it never promotes in a
            # benign or sparse context, never promotes without retrieval evidence, and never
            # demotes a mapping that a cue or structured signal independently corroborates.
            if (
                reranker_score is not None
                and reranker_score <= demote_threshold
                and not cue_supported
            ):
                accepted = False
                reason = "validated reranker score was below the demotion threshold"
            elif (
                reranker_score is not None
                and reranker_score >= promote_threshold
                and (dense is not None or lexical is not None)
                and not benign_context
                and not sparse_context
            ):
                accepted = True
                reason = "validated reranker score passed the promotion threshold"
        update: dict[str, Any] = {
            "state": (
                TechniqueCandidateState.AUTO_ACCEPTED
                if accepted
                else TechniqueCandidateState.NEEDS_REVIEW
            ),
            "state_reason": reason,
        }
        if extra_evidence:
            update["evidence"] = [*candidate.evidence, *extra_evidence]
        output.append(candidate.model_copy(update=update))
    if hierarchy_parent_mode == "off":
        return output
    return _apply_hierarchy_consistency(
        output,
        dense_floor=dense_floor,
        relative_margin=relative_margin,
        fill_unaccepted=hierarchy_parent_mode == "swap_and_fill",
    )


def _plausible(candidate: TechniqueCandidate, dense_floor: float) -> bool:
    if candidate.state is TechniqueCandidateState.AUTO_ACCEPTED:
        return True
    dense = candidate.evidence_for("dense")
    lexical = candidate.evidence_for("lexical")
    return (dense is not None and float(dense.score or 0.0) >= dense_floor) or (
        lexical is not None and (lexical.rank or 999) <= 3
    )


def _similarity_only(candidate: TechniqueCandidate) -> bool:
    return (
        candidate.evidence_for("explicit_cue") is None
        and candidate.evidence_for("structured_signal") is None
        and candidate.verification.verdict is not TechniqueVerificationVerdict.SUPPORTED
    )


def _apply_hierarchy_consistency(
    candidates: list[TechniqueCandidate],
    *,
    dense_floor: float,
    relative_margin: float,
    fill_unaccepted: bool = False,
) -> list[TechniqueCandidate]:
    """Accept the parent instead of guessing between plausible sub-techniques.

    When two or more sub-techniques of one parent are plausible for the same sentence and
    none is confidently accepted, or the only similarity-based acceptance leads its best
    sibling by less than ``relative_margin`` in dense similarity, the evidence supports the
    parent technique but not the choice of child: the derived parent candidate is accepted
    and the children stay in review. This targets the sub-technique-for-parent false
    positives measured on the public benchmark (parent-aware F1 0.48 vs exact 0.37).
    """
    parents = {
        candidate.technique_id: candidate
        for candidate in candidates
        if candidate.evidence_for("hierarchy_parent") is not None
    }
    if not parents:
        return candidates
    children_by_parent: dict[str, list[TechniqueCandidate]] = {}
    for candidate in candidates:
        if (
            candidate.parent_technique_id in parents
            and candidate.evidence_for("hierarchy_parent") is None
        ):
            children_by_parent.setdefault(candidate.parent_technique_id, []).append(candidate)
    updates: dict[str, TechniqueCandidate] = {}
    for parent_id, children in children_by_parent.items():
        plausible = [child for child in children if _plausible(child, dense_floor)]
        if len(plausible) < 2:
            continue
        accepted = [
            child for child in plausible if child.state is TechniqueCandidateState.AUTO_ACCEPTED
        ]
        if any(not _similarity_only(child) for child in accepted):
            continue  # a cue- or verifier-backed child is a deliberate choice; keep it
        if len(accepted) > 1:
            continue  # several children independently accepted: keep them all
        if not accepted and not fill_unaccepted:
            # No child was accepted: without ``fill_unaccepted`` the abstention stands. On the
            # public benchmark filling the parent here collapsed precision (0.50 -> 0.27).
            continue
        if len(accepted) == 1:
            leader = accepted[0]
            leader_dense = float(
                (leader.evidence_for("dense") or SimpleNamespace(score=0.0)).score or 0.0
            )
            rival_dense = max(
                float((child.evidence_for("dense") or SimpleNamespace(score=0.0)).score or 0.0)
                for child in plausible
                if child.technique_id != leader.technique_id
            )
            if leader_dense - rival_dense >= relative_margin:
                continue  # the accepted child is clearly ahead of its siblings
            updates[leader.technique_id] = leader.model_copy(
                update={
                    "state": TechniqueCandidateState.NEEDS_REVIEW,
                    "state_reason": (
                        f"sub-technique not separable from sibling(s) within {relative_margin:.2f}; "
                        f"parent {parent_id} accepted instead"
                    ),
                }
            )
        names = ", ".join(sorted(child.technique_id for child in plausible))
        updates[parent_id] = parents[parent_id].model_copy(
            update={
                "state": TechniqueCandidateState.AUTO_ACCEPTED,
                "state_reason": (
                    f"parent accepted: plausible sub-techniques {names} are not individually "
                    "confident"
                ),
            }
        )
    if not updates:
        return candidates
    return [updates.get(candidate.technique_id, candidate) for candidate in candidates]


def _expand_hierarchy_candidates(
    observation: Any,
    kb: Any,
    candidates: list[TechniqueCandidate],
) -> list[TechniqueCandidate]:
    """Add parent techniques when sub-techniques are in the candidate pool.

    When multiple sub-techniques of the same parent are retrieved, the parent
    technique is added as a derived candidate so that groups documenting only
    the parent can still be matched. The parent is marked as needs_review
    because it was not directly retrieved.
    """
    techniques = getattr(kb, "techniques", {})
    by_id = {c.technique_id: c for c in candidates}
    parent_counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.parent_technique_id:
            parent_counts[candidate.parent_technique_id] = (
                parent_counts.get(candidate.parent_technique_id, 0) + 1
            )
    for parent_id, child_count in parent_counts.items():
        if parent_id in by_id or parent_id not in techniques:
            continue
        if child_count < 2:
            continue
        by_id[parent_id] = TechniqueCandidate(
            observation_id=observation.id,
            technique_id=parent_id,
            technique_name=_technique_name(kb, parent_id),
            parent_technique_id=None,
            is_subtechnique=False,
            fused_score=0.5 / 61,
            fused_rank=len(by_id) + 1,
            evidence=[
                TechniqueRetrievalEvidence(
                    source="hierarchy_parent",
                    text=f"Derived from {child_count} sub-techniques of {parent_id}",
                    score=0.5,
                    rank=1,
                )
            ],
            state=TechniqueCandidateState.NEEDS_REVIEW,
            state_reason=f"hierarchy parent of {child_count} retrieved sub-techniques (not directly observed)",
        )
    ordered = sorted(by_id.values(), key=_candidate_order_key)
    return [item.model_copy(update={"fused_rank": rank}) for rank, item in enumerate(ordered, 1)]


def _candidate_order_key(item: TechniqueCandidate) -> tuple[int, float, str]:
    """Preserve the order ``LocalReranker.rerank`` assigned (kept in ``fused_rank``) for
    reranked candidates; sort everything else (e.g. derived parents) by fused score after them.

    Before this the hierarchy expansion re-sorted by ``fused_score`` and silently threw the
    reranker's ordering away; only its evidence score survived.
    """
    if item.evidence_for("reranker") is not None:
        return (0, float(item.fused_rank), item.technique_id)
    return (1, -item.fused_score, item.technique_id)


def _merge_explicit_candidates(
    observation: Any,
    kb: Any,
    candidates: list[TechniqueCandidate],
) -> list[TechniqueCandidate]:
    by_id = {candidate.technique_id: candidate for candidate in candidates}
    for technique_id, cue in _EXPLICIT_TECHNIQUE_CUES.items():
        if technique_id not in getattr(kb, "techniques", {}):
            continue
        match = cue.search(observation.text)
        if match is None:
            continue
        evidence = TechniqueRetrievalEvidence(
            source="explicit_cue",
            text=match.group(0),
            rationale="explicit behavior cue matched the observation",
            score=0.85,
            rank=1,
        )
        if technique_id in by_id:
            current = by_id[technique_id]
            if current.evidence_for("explicit_cue") is None:
                by_id[technique_id] = current.model_copy(
                    update={
                        "evidence": [*current.evidence, evidence],
                        "fused_score": current.fused_score + 1.0 / 61,
                    }
                )
            continue
        technique = getattr(kb, "techniques", {})[technique_id]
        by_id[technique_id] = TechniqueCandidate(
            observation_id=observation.id,
            technique_id=technique_id,
            technique_name=_technique_name(kb, technique_id),
            parent_technique_id=str(_record_value(technique, "parent_id", "") or "") or None,
            is_subtechnique=bool(_record_value(technique, "is_subtechnique", False)),
            fused_score=1.0 / 61,
            fused_rank=len(by_id) + 1,
            evidence=[evidence],
            state=TechniqueCandidateState.NEEDS_REVIEW,
            state_reason="explicit behavior cue requires independent support",
        )
    ordered = sorted(by_id.values(), key=lambda item: (-item.fused_score, item.technique_id))
    return [item.model_copy(update={"fused_rank": rank}) for rank, item in enumerate(ordered, 1)]


def _materialize_candidates(
    candidates: list[TechniqueCandidate],
    kb: Any,
    *,
    parent_policy: str,
    corroboration_bonus: float = 0.0,
) -> tuple[list[TechniqueMatch], dict[str, float], dict[str, set[str]]]:
    # Include both AUTO_ACCEPTED and NEEDS_REVIEW candidates so that cue-only
    # proposals remain visible to the analyst and to benchmarks. The state is
    # preserved in metadata["mapping_state"] so reviewers can filter on it.
    visible = [
        candidate
        for candidate in candidates
        if candidate.state
        in (TechniqueCandidateState.AUTO_ACCEPTED, TechniqueCandidateState.NEEDS_REVIEW)
    ]
    child_parents_by_observation: dict[str, set[str]] = {}
    for candidate in visible:
        if candidate.parent_technique_id:
            child_parents_by_observation.setdefault(candidate.observation_id, set()).add(
                candidate.parent_technique_id
            )
    matches: list[TechniqueMatch] = []
    score_evidence: dict[str, list[float]] = {}
    observation_ids: dict[str, set[str]] = {}
    for candidate in visible:
        if (
            candidate.technique_id
            in child_parents_by_observation.get(candidate.observation_id, set())
            and candidate.evidence_for("hierarchy_parent") is None
            and candidate.evidence_for("dense") is None
            and candidate.evidence_for("lexical") is None
        ):
            continue
        score = _candidate_strength(candidate)
        best_evidence = min(
            candidate.evidence,
            key=lambda item: (item.rank or 999, item.source),
            default=None,
        )
        sources = [item.source for item in candidate.evidence]
        metadata = {
            "mapping_state": candidate.state.value,
            "mapping_reason": candidate.state_reason,
            "fused_rank": candidate.fused_rank,
            "sources": sources,
            "source_text": candidate.observation_id,
        }
        if "explicit_cue" in sources:
            metadata["source"] = "explicit_technique_cue"
        if best_evidence is not None:
            metadata.update(
                {
                    "vector_id": best_evidence.document_id,
                    "field": best_evidence.field,
                }
            )
        matches.append(
            TechniqueMatch(
                technique_id=candidate.technique_id,
                technique_name=candidate.technique_name,
                score=score,
                observation_ids=[candidate.observation_id],
                parent_technique_id=candidate.parent_technique_id,
                matched_text=best_evidence.text if best_evidence else None,
                metadata=metadata,
            )
        )
        # Only AUTO_ACCEPTED candidates feed observed_techniques, which is
        # used downstream for attribution scoring. NEEDS_REVIEW candidates
        # remain visible in matches but do not vote until reviewed.
        if candidate.state is TechniqueCandidateState.AUTO_ACCEPTED:
            score_evidence.setdefault(candidate.technique_id, []).append(score)
            observation_ids.setdefault(candidate.technique_id, set()).add(candidate.observation_id)
        if parent_policy == "legacy_expand" and candidate.parent_technique_id:
            parent_id = candidate.parent_technique_id
            parent_score = score * 0.95
            matches.append(
                TechniqueMatch(
                    technique_id=parent_id,
                    technique_name=_technique_name(kb, parent_id),
                    score=parent_score,
                    observation_ids=[candidate.observation_id],
                    matched_text=best_evidence.text if best_evidence else None,
                    metadata={
                        "rolled_up_from": candidate.technique_id,
                        "source_text": candidate.observation_id,
                    },
                )
            )
    # Cross-observation corroboration: a technique accepted from several independent
    # observations is more credible than one seen once (campaign-level voting; the bonus
    # is 0 by default and measured with the attribution benchmark before adoption).
    observed = {
        technique_id: min(
            1.0,
            _aggregate_scores(scores)
            + corroboration_bonus * min(len(observation_ids.get(technique_id, ())) - 1, 3),
        )
        for technique_id, scores in score_evidence.items()
    }
    return matches, observed, observation_ids


def _candidate_strength(candidate: TechniqueCandidate) -> float:
    explicit = candidate.evidence_for("explicit_cue")
    signal = candidate.evidence_for("structured_signal")
    dense = candidate.evidence_for("dense")
    lexical = candidate.evidence_for("lexical")
    hierarchy = candidate.evidence_for("hierarchy_parent")
    has_cue = explicit is not None or signal is not None
    if has_cue and dense is not None:
        return min(1.0, max(0.0, dense.score) + 0.1)
    if has_cue and lexical is not None:
        return 0.85
    if has_cue and dense is None and lexical is None:
        return 0.85
    if hierarchy is not None:
        return 0.5
    if dense is not None and dense.score is not None:
        return min(1.0, max(0.0, dense.score))
    if lexical is not None:
        return 0.5
    return 0.0


def _field_weight(hit: VectorHit) -> float:
    return {
        "name": 0.8,
        "description": 1.0,
        "procedure": 0.9,
        "synthetic": 0.9,
        "detection": 0.3,
    }.get(str(hit.metadata.get("field") or ""), 1.0)


_NAME_TOKEN_STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "via",
    "with",
    "from",
    "by",
    "at",
    "is",
    "as",
    "be",
    "was",
    "may",
    "can",
    "using",
    "used",
    "use",
    "into",
    "over",
    "across",
    "system",
    "windows",
    "adversary",
    "adversaries",
    "technique",
    "based",
    "other",
    "such",
    "this",
    "that",
    "these",
    "those",
}


def _name_token_bonus(kb: Any, technique_id: str, observation_lower: str) -> float:
    """Boost score when distinctive technique name tokens appear in observation.

    Returns a bonus to add to the fused score. Tokens that are generic
    security terms are excluded. Only distinctive tokens (e.g., "powershell",
    "lsass", "rundll32", "smb", "keylogging") produce a bonus. Exact tool
    name matches (e.g., "mshta" in observation and technique name "Mshta")
    get a stronger boost.
    """
    technique = getattr(kb, "techniques", {}).get(technique_id)
    if technique is None:
        return 0.0
    name = _technique_name(kb, technique_id).casefold()
    raw_tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.:/\\-]{2,}", name)
        if token not in _NAME_TOKEN_STOPWORDS and len(token) >= 4
    ]
    tokens = list(dict.fromkeys(raw_tokens))
    if len(tokens) < 2:
        return 0.0
    matches = sum(1 for token in tokens if token in observation_lower)
    if matches == 0:
        return 0.0
    match_ratio = matches / len(tokens)
    if match_ratio < 0.34:
        return 0.0
    exact_tool_bonus = 0.0
    tool_patterns = {
        "powershell",
        "mshta",
        "rundll32",
        "regsvr32",
        "certutil",
        "bitsadmin",
        "msiexec",
        "wmic",
        "cmd",
        "bash",
        "python",
        "lsass",
        "schtasks",
        "vssadmin",
        "ntdsutil",
        "sekurlsa",
    }
    for token in tokens:
        if token in tool_patterns and token in observation_lower:
            exact_tool_bonus = max(exact_tool_bonus, 0.015)
    return 0.02 * match_ratio + exact_tool_bonus


_LEXICAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
_LEXICAL_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:/\\-]{1,}")


def _lexical_tokens(value: str) -> list[str]:
    return [
        token
        for token in _LEXICAL_TOKEN_RE.findall(value.casefold())
        if token not in _LEXICAL_STOPWORDS
    ]


def _record_value(record: Any, name: str, default: Any = "") -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def build_classifier_context(
    kb: Any,
    ledger: EvidenceLedger,
    retrieval: RetrievalResult,
    *,
    telemetry_profile: Any = None,
    soft_min_probability: float = 0.0,
    soft_weight: float = 1.0,
    soft_max_groups: int = 0,
    contrast_min_probability: float = 0.0,
    contrast_weight: float = 1.0,
) -> ClassifierContext:
    observed = dict(retrieval.observed_techniques)
    technique_observations = dict(retrieval.technique_observations)
    negative = _validated_negative_techniques(ledger, kb, telemetry_profile)
    soft, soft_observations = soft_technique_evidence(
        retrieval,
        observed=observed,
        negative=negative,
        min_probability=soft_min_probability,
        weight=soft_weight,
        max_groups=soft_max_groups,
        kb=kb,
    )
    software_ids = {
        str(observation.metadata["software_id"])
        for observation in ledger.observations
        if observation.status is EvidenceStatus.POSITIVE and observation.metadata.get("software_id")
    }
    narrative = " ".join(
        observation.text
        for observation in ledger.observations
        if observation.status is EvidenceStatus.POSITIVE
    )
    software_ids.update(
        match.software_id for match in match_software_aliases(narrative, kb) if match.software_id
    )
    context = ClassifierContext(
        kb=kb,
        observed_techniques=observed,
        technique_observations=technique_observations,
        negative_techniques=negative,
        semantic_profile_scores=retrieval.semantic_profile_scores,
        group_text_scores=retrieval.group_text_scores,
        software_ids=software_ids,
        narrative=narrative,
        parent_policy=str(getattr(retrieval, "parent_policy", "ancestor_compatible")),
    )
    context.soft_techniques = soft
    context.soft_technique_observations = soft_observations
    # Contrastive near misses: same qualification as soft evidence, but they only reach the
    # candidate-relative separation (exclusivity-weighted), never the classifiers.
    context.contrast_techniques, context.contrast_technique_observations = soft_technique_evidence(
        retrieval,
        observed=observed,
        negative=negative,
        min_probability=contrast_min_probability,
        weight=contrast_weight,
        max_groups=0,
        kb=kb,
    )
    context.strategic_context = ledger.strategic_context
    context.specific_behavior_scores = retrieval.specific_behavior_scores
    context.specific_behavior_signals = retrieval.specific_behavior_signals
    context.specific_behavior_matches = tuple(retrieval.specific_behavior_matches or ())
    return context


SOFT_EVIDENCE_MAX_WEIGHT = 0.95


def soft_technique_evidence(
    retrieval: RetrievalResult,
    *,
    observed: dict[str, float],
    negative: set[str],
    min_probability: float,
    weight: float = 1.0,
    max_groups: int = 0,
    kb: Any = None,
) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """Near-miss candidates for the group ranking: NEEDS_REVIEW candidates whose calibrated
    acceptance probability is at least ``min_probability`` count with ``probability * weight``
    (capped below any accepted technique). Accepted, rejected, negative and derived-parent
    candidates never qualify. With ``max_groups`` > 0 only techniques documented for at most
    that many groups qualify (a near-miss generic technique is noise, a near-miss distinctive
    one is information). ``min_probability`` 0 disables the mechanism."""
    if min_probability <= 0:
        return {}, {}
    documented_for: dict[str, int] = {}
    if max_groups > 0 and kb is not None:
        for technique_ids in (getattr(kb, "group_techniques", {}) or {}).values():
            for technique_id in set(technique_ids):
                documented_for[technique_id] = documented_for.get(technique_id, 0) + 1
    soft: dict[str, float] = {}
    ids: dict[str, set[str]] = {}
    for candidate in retrieval.candidates:
        if candidate.state is not TechniqueCandidateState.NEEDS_REVIEW:
            continue
        technique_id = candidate.technique_id
        if technique_id in observed or technique_id in negative:
            continue
        if candidate.evidence_for("hierarchy_parent") is not None:
            continue
        if max_groups > 0 and documented_for.get(technique_id, 0) > max_groups:
            continue
        evidence = candidate.evidence_for(ACCEPTANCE_EVIDENCE_SOURCE)
        if evidence is None or evidence.score is None:
            continue
        probability = float(evidence.score)
        if probability < min_probability:
            continue
        value = min(SOFT_EVIDENCE_MAX_WEIGHT, probability * float(weight))
        if value > soft.get(technique_id, 0.0):
            soft[technique_id] = value
        ids.setdefault(technique_id, set()).add(candidate.observation_id)
    return soft, {key: tuple(sorted(value)) for key, value in ids.items()}


def refresh_context_answers(
    context: ClassifierContext,
    ledger: EvidenceLedger,
    *,
    kb: Any = None,
    telemetry_profile: Any = None,
) -> ClassifierContext:
    context.negative_techniques = _validated_negative_techniques(ledger, kb, telemetry_profile)
    for answer in ledger.question_answers:
        if not answer.technique_id or answer.status is not EvidenceStatus.POSITIVE:
            continue
        context.observed_techniques[answer.technique_id] = 1.0
        existing = set(context.technique_observations.get(answer.technique_id, ()))
        existing.add(answer.id)
        context.technique_observations[answer.technique_id] = tuple(sorted(existing))
    return context


def _validated_negative_techniques(
    ledger: EvidenceLedger,
    kb: Any,
    telemetry_profile: Any,
) -> set[str]:
    """Collect negative technique IDs, validating telemetry availability.

    A negative answer is treated as evidence only when the relevant telemetry
    is available for the technique. If no telemetry profile is provided or the
    KB lacks detection data, all negative answers are accepted (backward
    compatible with the pre-telemetry behavior). When a telemetry profile
    exists, negative answers without available telemetry are downgraded to
    ``NOT_VISIBLE`` and excluded from the negative set.
    """
    negative = {
        answer.technique_id
        for answer in ledger.question_answers
        if answer.technique_id and answer.status is EvidenceStatus.NEGATIVE
    }
    if telemetry_profile is None or kb is None:
        return negative
    if not getattr(kb, "detection_strategies", None):
        return negative
    try:
        from apt_classifier.telemetry import negative_answer_is_valid
    except ImportError:
        return negative
    return {
        technique_id
        for technique_id in negative
        if negative_answer_is_valid(technique_id, kb, telemetry_profile)
    }


def _specific_behavior_results(
    observations: list[Any],
    hit_lists: list[list[VectorHit]],
    *,
    threshold: float,
    relative_margin: float = 0.10,
    minimum_discrimination: float = 0.05,
    max_groups_per_observation: int = 8,
) -> tuple[dict[str, float], dict[str, tuple[str, ...]], list[BehaviorMatch]]:
    evidence_scores: dict[str, list[float]] = {}
    evidence_observations: dict[str, set[str]] = {}
    signal_ids: dict[str, list[str]] = {}
    matches: list[BehaviorMatch] = []
    for observation, hits in zip(observations, hit_lists, strict=True):
        eligible = [hit for hit in hits if hit.similarity >= threshold]
        if not eligible:
            continue
        cutoff = max(threshold, max(hit.similarity for hit in eligible) - relative_margin)
        best_by_group: dict[str, VectorHit] = {}
        for hit in eligible:
            group_id = str(hit.metadata.get("group_id") or "")
            if not group_id or not _has_specific_behavior_details(observation.text, hit.document):
                continue
            if group_id not in best_by_group or hit.similarity > best_by_group[group_id].similarity:
                best_by_group[group_id] = hit
        ordered = sorted(best_by_group.items(), key=lambda item: (-item[1].similarity, item[0]))
        if len(ordered) >= 4:
            comparison_score = ordered[3][1].similarity
            required_discrimination = minimum_discrimination
        elif len(ordered) >= 2:
            comparison_score = ordered[1][1].similarity
            required_discrimination = minimum_discrimination * 0.6
        else:
            comparison_score = None
            required_discrimination = 0.0
        if (
            comparison_score is not None
            and ordered[0][1].similarity - comparison_score < required_discrimination
        ):
            continue
        selected = [item for item in ordered if item[1].similarity >= cutoff][
            :max_groups_per_observation
        ]
        for group_id, hit in selected:
            technique_id = str(hit.metadata.get("technique_id") or "")
            evidence_scores.setdefault(group_id, []).append(hit.similarity)
            evidence_observations.setdefault(group_id, set()).add(observation.id)
            signal_ids.setdefault(group_id, []).append(hit.id)
            matches.append(
                BehaviorMatch(
                    group_id=group_id,
                    observation_id=observation.id,
                    technique_id=technique_id,
                    score=hit.similarity,
                    matched_text=hit.document,
                    vector_id=hit.id,
                )
            )
    observation_count = max(len(observations), 1)
    scores = {}
    for group_id, values in evidence_scores.items():
        strongest = sorted(values, reverse=True)[:3]
        mean = sum(strongest) / len(strongest)
        coverage = len(evidence_observations[group_id]) / observation_count
        scores[group_id] = (0.65 * mean + 0.35 * strongest[0]) * (0.6 + 0.4 * coverage)
    signals = {group_id: tuple(dict.fromkeys(values)) for group_id, values in signal_ids.items()}
    return scores, signals, matches


def _has_specific_behavior_details(observed: str, documented: str) -> bool:
    observed_tokens = _behavior_tokens(observed)
    documented_tokens = _behavior_tokens(documented)
    details = (observed_tokens & documented_tokens) - _BEHAVIOR_GENERIC_TOKENS
    return len(details) >= 2 or any(len(token) >= 12 for token in details)


def _behavior_tokens(value: str) -> set[str]:
    tokens = {
        token.strip("._:/\\-")
        for token in re.findall(r"[a-z0-9][a-z0-9_.:/\\-]{3,}", value.casefold())
    }
    tokens.discard("")
    tokens.update(
        token.rsplit(".", 1)[0]
        for token in tuple(tokens)
        if token.endswith((".exe", ".dll", ".ps1", ".bat", ".vbs"))
    )
    return tokens


def _best_technique_hits(
    hits: list[VectorHit],
    threshold: float,
    required_platform: str | None = None,
    relative_margin: float = 0.12,
    max_candidates: int = 6,
) -> dict[str, VectorHit]:
    eligible = [
        hit
        for hit in hits
        if hit.similarity >= threshold
        and _platform_matches(hit.metadata.get("platforms"), required_platform)
    ]
    if not eligible:
        return {}
    cutoff = max(threshold, max(hit.similarity for hit in eligible) - relative_margin)
    best: dict[str, VectorHit] = {}
    for hit in sorted(eligible, key=lambda item: (-item.similarity, item.id)):
        technique_id = str(hit.metadata.get("technique_id") or "")
        if not technique_id or hit.similarity < cutoff:
            continue
        if technique_id not in best or hit.similarity > best[technique_id].similarity:
            best[technique_id] = hit
        if len(best) >= max_candidates:
            break
    return best


def _platform_matches(platforms: Any, required_platform: str | None) -> bool:
    if not required_platform or not platforms:
        return True
    values = platforms.split(",") if isinstance(platforms, str) else list(platforms)
    return required_platform.casefold() in {str(value).strip().casefold() for value in values}


def _required_platform(observation: Any) -> str | None:
    explicit = observation.metadata.get("platform") if hasattr(observation, "metadata") else None
    if explicit:
        return str(explicit)
    source = str(getattr(getattr(observation, "provenance", None), "source", ""))
    if getattr(observation, "kind", "") == "winlog" or source == "winlog":
        return "Windows"
    return None


def _source_reliability(observation: Any, hit: VectorHit) -> float:
    return 0.95 if _required_platform(observation) == "Windows" else 1.0


def _aggregate_scores(scores: list[float]) -> float:
    strongest = sorted(scores, reverse=True)[:3]
    maximum = strongest[0]
    mean = sum(strongest) / len(strongest)
    return min(1.0, 0.75 * maximum + 0.25 * mean)


def _apply_positive_answers(
    ledger: EvidenceLedger,
    kb: Any,
    observed: dict[str, float],
    observation_ids: dict[str, set[str]],
    matches: list[TechniqueMatch],
) -> None:
    for answer in ledger.question_answers:
        if not answer.technique_id or answer.status is not EvidenceStatus.POSITIVE:
            continue
        technique_id = answer.technique_id
        observed[technique_id] = 1.0
        observation_ids.setdefault(technique_id, set()).add(answer.id)
        matches.append(
            TechniqueMatch(
                technique_id=technique_id,
                technique_name=_technique_name(kb, technique_id),
                score=1.0,
                observation_ids=[answer.id],
                matched_text=answer.answer_text or answer.question,
                metadata={"source": "analyst_answer"},
            )
        )


def _group_scores(hits: list[VectorHit]) -> dict[str, float]:
    output: dict[str, float] = {}
    for hit in hits:
        group_id = str(hit.metadata.get("group_id") or "")
        if group_id:
            output[group_id] = max(output.get(group_id, -1.0), hit.similarity)
    return output


def _technique_name(kb: Any, technique_id: str) -> str:
    technique = getattr(kb, "techniques", {}).get(technique_id)
    if technique is None:
        return technique_id
    if isinstance(technique, dict):
        return str(technique.get("name") or technique_id)
    return str(getattr(technique, "name", technique_id))
