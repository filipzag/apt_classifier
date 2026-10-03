"""Free-text group model: a TF-IDF word document per ATT&CK group, scored against the case text.

Chosen in the free-text study (``scripts/text_group_study/``): on the 23 single-group ATT&CK
campaigns it beat every ATTACK-BERT embedding model tried, because it keeps exact malware/tool
names and distinctive phrases that sentence embeddings blur.

Training text is ATT&CK only (never benchmark or case data):

- ``group``: the group's procedure examples (``Group.technique_behaviors``);
- ``software``: the procedure examples of every malware/tool (from the raw STIX bundle, the
  knowledge base keeps none), credited in full to every group ATT&CK documents using it;
- ``group_desc``: the sentences of the group description;
- ``campaign``: campaign procedure examples, credited to the campaign's attributed groups.

Every text has its ATT&CK citations stripped, every group name/alias replaced with "the
adversary" (the attribution benchmarks' evidence pattern) and every campaign name replaced with
"the operation"; "adversary" and "operation" are stop words, so neither carries signal.

Model: word 1-2-grams (English stop words, ``min_df`` 2 over the training sentences); per group
the summed counts, log-scaled, times an IDF over the group documents, L2-normalised. A case is
the joined text of its positive observations, scored by cosine to every group document,
z-scored across groups and shifted by ``prior`` x the centred log prominence. Groups with no
shared term get no score (the classifier abstains for them).

The artifact (``data/models/text_group/text_group.json.gz``) stores the training records and
parameters, pinned to the ATT&CK version and integrity-checked by a fingerprint; the TF-IDF is
refitted at load (deterministic, about a second). ``without`` refits on a filtered record set,
which the benchmarks use to hide a fold's held-out behaviour or a removed actor.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import zlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
KIND = "text_group_tfidf"
TEXT_GROUP_NAME = "text_group"
DEFAULT_PARAMS: dict[str, Any] = {
    "ngram_max": 2,
    "min_df": 2,
    "software_weight": 1.0,
    "prior": 0.5,
    "min_words": 3,
    "min_desc_words": 4,
}
_EXTRA_STOP = ("adversary", "operation", "operations")
_ADVERSARY = re.compile(r"\bthe adversary(?:'s|s)?\b|\badversary(?:'s|s)?\b")
_ARTICLE = re.compile(r"\b(?:the|a|an)\s+(the (?:adversary|operation))", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


# ---------------------------------------------------------------------------- corpus
def actor_names(kb: Any) -> tuple[str, ...]:
    """Every ATT&CK group name and alias, longest first (the benchmarks' anonymisation set)."""
    return tuple(
        sorted(
            {
                name.strip()
                for group in (getattr(kb, "groups", {}) or {}).values()
                for name in (group.name, *group.aliases)
                if name and name.strip()
            },
            key=lambda item: (-len(item), item.casefold()),
        )
    )


def evidence_name_pattern(
    names: Sequence[str], *, before_hyphen: bool = False
) -> re.Pattern[str] | None:
    """The attribution benchmarks' evidence pattern (``_evidence_name_pattern``): a name ending a
    sentence is replaced too; a dot followed by a word character protects file/family names.
    ``before_hyphen`` also replaces a name followed by a hyphen ("Sea Turtle-controlled"), which
    the benchmarks' pattern keeps; the corpus uses it so a group's own name cannot become one of
    its document's words."""
    ordered = sorted((n for n in names if n), key=lambda item: (-len(item), item.casefold()))
    if not ordered:
        return None
    ahead = r"(?![\w]|\.\w)" if before_hyphen else r"(?![\w-]|\.\w)"
    return re.compile(
        r"(?<![\w.-])(?:" + "|".join(re.escape(name) for name in ordered) + r")"
        r"(?P<possessive>['’]s)?" + ahead,
        re.IGNORECASE,
    )


def _campaign_pattern(kb: Any) -> re.Pattern[str] | None:
    names = sorted(
        {
            n
            for c in (getattr(kb, "campaigns", {}) or {}).values()
            for n in (c.name, *c.aliases)
            if n and len(n) > 2
        },
        key=lambda s: (-len(s), s.casefold()),
    )
    if not names:
        return None
    return re.compile(
        r"(?<![\w.-])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w-]|\.\w)",
        re.IGNORECASE,
    )


def _attack_id(obj: Mapping[str, Any]) -> str | None:
    return next(
        (
            r["external_id"]
            for r in obj.get("external_references", [])
            if r.get("source_name") == "mitre-attack"
        ),
        None,
    )


def build_corpus(
    kb: Any, bundle: Mapping[str, Any] | None, params: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Anonymised, labelled ATT&CK training texts (see the module docstring).

    ``bundle`` is the raw enterprise STIX bundle (software and campaign procedure examples live
    only there); ``None`` keeps the group procedures and descriptions only."""
    from apt_classifier.attack.parse import strip_attack_citations

    p = {**DEFAULT_PARAMS, **(params or {})}
    actors = evidence_name_pattern(actor_names(kb), before_hyphen=True)
    campaigns = _campaign_pattern(kb)

    def clean(text: str) -> str:
        text = strip_attack_citations(text or "").strip()
        if actors is not None:
            text = actors.sub(lambda m: "the adversary" + (m.group("possessive") or ""), text)
        if campaigns is not None:
            text = campaigns.sub("the operation", text)
        text = _ARTICLE.sub(r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    software_groups: dict[str, set[str]] = {}
    for group_id, software_ids in (getattr(kb, "group_software", {}) or {}).items():
        for software_id in software_ids:
            software_groups.setdefault(software_id, set()).add(group_id)
    records: list[dict[str, Any]] = []
    for group_id, group in sorted((getattr(kb, "groups", {}) or {}).items()):
        for technique_id, values in sorted(group.technique_behaviors.items()):
            for value in values:
                text = clean(value)
                if len(text.split()) >= int(p["min_words"]):
                    records.append(
                        {
                            "text": text,
                            "source": "group",
                            "owner": group_id,
                            "groups": [group_id],
                            "technique_id": technique_id,
                        }
                    )
        for sentence in _SENTENCE.split(strip_attack_citations(group.description or "")):
            text = clean(sentence)
            if len(text.split()) >= int(p["min_desc_words"]):
                records.append(
                    {
                        "text": text,
                        "source": "group_desc",
                        "owner": group_id,
                        "groups": [group_id],
                        "technique_id": None,
                    }
                )
    if bundle is None:
        return records
    live = [
        o
        for o in bundle.get("objects", [])
        if not o.get("revoked") and not o.get("x_mitre_deprecated")
    ]
    ids = {o["id"]: _attack_id(o) for o in live if "id" in o}
    types = {o["id"]: o["type"] for o in live if "id" in o}
    known_groups = set(getattr(kb, "groups", {}) or {})
    campaign_groups: dict[str, set[str]] = {}
    for o in live:
        if o.get("type") == "relationship" and o.get("relationship_type") == "attributed-to":
            c, g = ids.get(o["source_ref"]), ids.get(o["target_ref"])
            if c and g and c.startswith("C") and g in known_groups:
                campaign_groups.setdefault(c, set()).add(g)
    for o in live:
        if o.get("type") != "relationship" or o.get("relationship_type") != "uses":
            continue
        if not o.get("description"):
            continue
        source_type, target_type = types.get(o["source_ref"]), types.get(o["target_ref"])
        source, target = ids.get(o["source_ref"]), ids.get(o["target_ref"])
        if not source or not target:
            continue
        text = clean(o["description"])
        if len(text.split()) < int(p["min_words"]):
            continue
        if source_type in ("malware", "tool") and target_type == "attack-pattern":
            records.append(
                {
                    "text": text,
                    "source": "software",
                    "owner": source,
                    "groups": sorted(software_groups.get(source, ())),
                    "technique_id": target,
                }
            )
        elif source_type == "campaign" and target_type in ("attack-pattern", "malware", "tool"):
            groups = sorted(campaign_groups.get(source, ()))
            if groups:
                records.append(
                    {
                        "text": text,
                        "source": "campaign",
                        "owner": source,
                        "groups": groups,
                        "technique_id": target if target_type == "attack-pattern" else None,
                    }
                )
    return records


def normalize_text(value: str) -> str:
    """Casefolded, punctuation-free form (the benchmarks' ``normalize_procedure_text``)."""
    return re.sub(r"\W+", " ", value or "").casefold().strip()


_KEY_DROP = frozenset({"the", "a", "an", "adversary", "operation"})
_ANY_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def match_key(
    value: str,
    campaigns: re.Pattern[str] | None = None,
    actors: re.Pattern[str] | None = None,
) -> str:
    """Comparison key between corpus texts and a fold's held-out sentences: the corpus replaced
    campaign names with "the operation" (and group names before a hyphen) while the fold texts
    keep them, so both sides lose campaign names, group names, the placeholders and articles
    before the normalised comparison."""
    text = value or ""
    for pattern in (campaigns, actors):
        if pattern is not None:
            text = pattern.sub(" ", text)
    return " ".join(w for w in normalize_text(text).split() if w not in _KEY_DROP)


def _keys(value: str, campaigns: Any, actors: Any) -> set[str]:
    """The record's key and the key of each of its sentences (software and campaign records
    are whole multi-sentence descriptions; a held-out sentence can sit inside one)."""
    keys = {match_key(value, campaigns, actors)}
    keys.update(match_key(part, campaigns, actors) for part in _ANY_SENTENCE.split(value or ""))
    keys.discard("")
    return keys


# ---------------------------------------------------------------------------- model
def _preprocess(text: str) -> str:
    return _ADVERSARY.sub(" ", text.lower())


def prominence_log_prior(kb: Any, group_ids: Sequence[str]) -> np.ndarray:
    prominence = getattr(kb, "group_prominence", {}) or {}
    values = np.log(np.asarray([float(prominence.get(g, 0)) + 1.0 for g in group_ids]))
    return (values - values.mean()).astype(np.float64) if len(values) else values


def _records_digest(
    records: Sequence[Mapping[str, Any]], params: Mapping[str, Any], attack_version: str = ""
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(sorted(params.items())), sort_keys=True).encode())
    digest.update(str(attack_version or "").encode())
    for r in records:
        digest.update(
            json.dumps(
                [r["text"], r["source"], r.get("owner"), list(r["groups"]), r.get("technique_id")]
            ).encode()
        )
    return digest.hexdigest()[:24]


class TextGroupModel:
    """The fitted group documents over ``group_ids`` (the knowledge base's groups)."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        group_ids: Sequence[str],
        log_prior: np.ndarray,
        *,
        params: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        _vectorizer: Any = None,
        _counts: Any = None,
    ) -> None:
        from scipy import sparse
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
        from sklearn.preprocessing import normalize

        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.records = [dict(r) for r in records]
        self.group_ids = tuple(group_ids)
        self.log_prior = np.asarray(log_prior, dtype=np.float64)
        if self.log_prior.shape != (len(self.group_ids),):
            raise ValueError("prior does not match the group list")
        self.metadata = dict(metadata or {})
        self._normalize = normalize
        if _vectorizer is None:
            self._vectorizer = CountVectorizer(
                preprocessor=_preprocess,
                analyzer="word",
                ngram_range=(1, int(self.params["ngram_max"])),
                stop_words=sorted(ENGLISH_STOP_WORDS | set(_EXTRA_STOP)),
                min_df=int(self.params["min_df"]),
                dtype=np.float32,
            )
            counts = self._vectorizer.fit_transform([r["text"] for r in self.records]).tocsr()
        else:  # a ``without`` refit: same vocabulary, the kept records' rows
            self._vectorizer = _vectorizer
            counts = _counts
            if counts.shape[0] != len(self.records):
                raise ValueError("count rows do not match the records")
        self._counts = counts
        self._actor_names: tuple[str, ...] = tuple(self.metadata.get("actor_names") or ())
        index = {g: i for i, g in enumerate(self.group_ids)}
        rows, cols, vals = [], [], []
        weight = float(self.params["software_weight"])
        for i, r in enumerate(self.records):
            w = weight if r["source"] == "software" else 1.0
            if w <= 0:
                continue
            for g in r["groups"]:
                if g in index:
                    rows.append(i)
                    cols.append(index[g])
                    vals.append(w)
        membership = sparse.csr_matrix(
            (np.asarray(vals, np.float32), (rows, cols)),
            shape=(len(self.records), len(self.group_ids)),
        )
        documents = (membership.T @ counts).tocsr()
        documents.eliminate_zeros()
        n_groups, vocabulary = documents.shape
        df = np.bincount(documents.indices, minlength=vocabulary)
        self._idf = (np.log((1.0 + n_groups) / (1.0 + df)) + 1.0).astype(np.float32)
        documents.data = np.log1p(documents.data)
        self._documents_t = normalize(documents @ sparse.diags(self._idf), norm="l2").T.tocsc()
        self._without_cache: dict[Any, TextGroupModel] = {}

    @property
    def runtime(self) -> dict[str, str]:
        """Library versions the TF-IDF was fitted with (it is refitted at every load)."""
        import scipy
        import sklearn

        return {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__}

    @property
    def fingerprint(self) -> str:
        return _records_digest(
            self.records, self.params, str(self.metadata.get("attack_version") or "")
        )

    def score_vector(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """``(cosine, score)`` over ``group_ids``: score = z-scored cosine + prior x log
        prominence (the study's ranking). Both are zeros for a text sharing no term."""
        from scipy import sparse

        empty = np.zeros(len(self.group_ids))
        if not (text or "").strip():
            return empty, empty
        row = self._vectorizer.transform([text]).tocsr()
        if row.nnz == 0:
            return empty, empty
        row.data = np.log1p(row.data)
        row = self._normalize(row @ sparse.diags(self._idf), norm="l2")
        cosine = np.asarray((row @ self._documents_t).todense(), dtype=np.float64)[0]
        if not np.any(cosine > 0):
            return empty, empty
        z = (cosine - cosine.mean()) / (cosine.std() + 1e-9)
        return cosine, z + float(self.params["prior"]) * self.log_prior

    def scores(self, text: str) -> dict[str, float]:
        """Per group sharing at least one term with ``text``: the softmax of the ranking score
        over those groups (positive, same order as ``score_vector``)."""
        cosine, score = self.score_vector(text)
        support = cosine > 0
        if not support.any():
            return {}
        kept = score[support]
        weights = np.exp(kept - kept.max())
        weights /= weights.sum()
        groups = [g for g, s in zip(self.group_ids, support, strict=True) if s]
        return {g: float(w) for g, w in zip(groups, weights, strict=True)}

    def without(
        self,
        kb: Any,
        *,
        group_id: str | None = None,
        techniques: Iterable[str] = (),
        texts: Iterable[str] = (),
        remove_group: bool = False,
    ) -> TextGroupModel:
        """Refit without a fold's evidence, mirroring the benchmarks' ``FoldStore``:

        - a record that matches a ``texts`` sentence (``match_key``, on the whole record or any
          of its sentences) is dropped when it credits ``group_id`` or no group (copies credited
          only to other groups stay, as other groups' identical behaviour documents stay in the
          store);
        - ``group_id``'s own procedure examples for ``techniques`` are dropped;
        - with ``remove_group`` (the unknown-actor arm) every record owned by the group goes and
          the group leaves every other record's groups; ``kb`` is then the reduced knowledge
          base, which sets the group list and the prior.

        Equal to a fresh fit on the kept records: the vocabulary is reused, and the terms whose
        document frequency over the kept records falls below ``min_df`` are zeroed, so a rare
        word kept alive only by a hidden sentence cannot match."""
        from scipy import sparse

        held = frozenset(techniques)
        hidden_raw = frozenset(texts)
        key = (group_id, held, hidden_raw, remove_group, id(kb))
        cached = self._without_cache.get(key)
        if cached is not None:
            return cached
        campaigns = _campaign_pattern(kb)
        actors = evidence_name_pattern(self._actor_names or actor_names(kb), before_hyphen=True)
        hidden = {match_key(t, campaigns, actors) for t in hidden_raw}
        hidden.discard("")
        keep_rows: list[int] = []
        kept: list[dict[str, Any]] = []
        for i, r in enumerate(self.records):
            credits_fold = group_id is None or group_id in r["groups"] or not r["groups"]
            if hidden and credits_fold and _keys(r["text"], campaigns, actors) & hidden:
                continue
            if group_id is not None and r["owner"] == group_id:
                if remove_group:
                    continue
                if r["source"] == "group" and r.get("technique_id") in held:
                    continue
            groups = list(r["groups"])
            if remove_group and group_id in groups:
                groups = [g for g in groups if g != group_id]
                if not groups and r["source"] != "software":
                    continue
            keep_rows.append(i)
            kept.append({**r, "groups": groups})
        counts = self._counts[keep_rows].tocsc()
        df = np.diff(counts.indptr)
        counts = (
            counts @ sparse.diags((df >= int(self.params["min_df"])).astype(np.float32))
        ).tocsr()
        counts.eliminate_zeros()
        group_ids = tuple(sorted(getattr(kb, "groups", {}) or {}))
        model = TextGroupModel(
            kept,
            group_ids,
            prominence_log_prior(kb, group_ids),
            params=self.params,
            metadata={
                **self.metadata,
                "refit": {
                    "group_id": group_id,
                    "remove_group": remove_group,
                    "techniques": len(held),
                    "texts": len(hidden_raw),
                    "dropped_records": len(self.records) - len(kept),
                },
            },
            _vectorizer=self._vectorizer,
            _counts=counts,
        )
        if len(self._without_cache) >= 4:
            self._without_cache.pop(next(iter(self._without_cache)))
        self._without_cache[key] = model
        return model

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "params": self.params,
            "metadata": self.metadata,
            "records": self.records,
            "fingerprint": self.fingerprint,
        }
        tmp = target.with_name(target.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path, kb: Any) -> TextGroupModel | None:
        source = Path(path)
        if not source.is_file():
            return None
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError("not a text-group artifact (no records)")
        if payload.get("kind") != KIND:
            raise ValueError(f"not a text-group artifact: {payload.get('kind')!r}")
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload.get('schema_version')!r}")
        params = {**DEFAULT_PARAMS, **(payload.get("params") or {})}
        records = payload["records"]
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("artifact metadata is not an object")
        version = str(metadata.get("attack_version") or "")
        if _records_digest(records, params, version) != payload.get("fingerprint"):
            raise ValueError("fingerprint mismatch (artifact modified or corrupted)")
        group_ids = tuple(sorted(getattr(kb, "groups", {}) or {}))
        return cls(
            records,
            group_ids,
            prominence_log_prior(kb, group_ids),
            params=params,
            metadata={**metadata, "actor_names": list(actor_names(kb))},
        )

    def compatible_with(self, kb: Any) -> tuple[bool, str]:
        fitted = str(self.metadata.get("attack_version") or "")
        version = str(getattr(kb, "version", "") or "")
        if not fitted:
            return False, "artifact records no ATT&CK version (rebuild: make fit-text-group)"
        if version and fitted != version:
            return (
                False,
                f"built on ATT&CK {fitted}, runtime has {version} (rebuild: make fit-text-group)",
            )
        return True, "ok"


def fit_text_group(
    kb: Any, bundle: Mapping[str, Any] | None, params: Mapping[str, Any] | None = None
) -> TextGroupModel:
    """Build the corpus from ``kb`` and the raw STIX ``bundle`` and fit the model."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    records = build_corpus(kb, bundle, p)
    group_ids = tuple(sorted(getattr(kb, "groups", {}) or {}))
    counts: dict[str, int] = {}
    for r in records:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    return TextGroupModel(
        records,
        group_ids,
        prominence_log_prior(kb, group_ids),
        params=p,
        metadata={
            "attack_version": str(getattr(kb, "version", "") or ""),
            "actor_names": list(actor_names(kb)),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "records_by_source": dict(sorted(counts.items())),
            "software_bundle": bundle is not None,
        },
    )


# ---------------------------------------------------------------------------- pipeline member
def text_group_bound_fingerprint(base: str, model: TextGroupModel) -> str:
    """The combiner pipeline fingerprint when the text model votes: the base fingerprint mixed
    with the model's records/params fingerprint and this module's source, so a combiner fitted
    with one text model never loads against another."""
    digest = hashlib.sha256()
    digest.update(base.encode())
    digest.update(model.fingerprint.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:24]


def text_group_votes(settings: Any) -> bool:
    """The model joins the ensemble only with a positive weight: at 0 it is not loaded, not run
    and adds no weight key, so the fusion (and the combiner fingerprint) are unchanged."""
    return bool(getattr(settings, "text_group_enabled", False)) and (
        float(getattr(settings, "text_group_weight", 0.0) or 0.0) > 0
    )


def load_text_group_for(settings: Any, kb: Any) -> tuple[TextGroupModel | None, str]:
    """The configured artifact for the engine, fail-closed: ``(model, status)``."""
    if not text_group_votes(settings):
        return None, "disabled (weight 0)"
    path = getattr(settings, "text_group_path", "")
    try:
        model = TextGroupModel.load(path, kb)
    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
        OSError,
        EOFError,
        zlib.error,
    ) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path} (run make fit-text-group)"
    ok, reason = model.compatible_with(kb)
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return model, "active"


class TextGroupClassifier:
    """Ensemble member: the text model's score for each eligible group sharing a term with the
    case text (the joined positive observations). It cannot make a group eligible."""

    name = TEXT_GROUP_NAME

    def __init__(self, model: TextGroupModel | None = None) -> None:
        self.model = model

    def classify(self, context: Any) -> Any:
        from apt_classifier.classifiers.base import build_result, group_ids

        narrative = str(getattr(context, "narrative", "") or "")
        if self.model is None or not narrative.strip():
            return build_result(self.name, context, {}, applicable=False)
        allowed = set(group_ids(context))
        scores = {g: s for g, s in self.model.scores(narrative).items() if g in allowed}
        signals = {g: (f"text_group:p={s:.3f}",) for g, s in scores.items()}
        return build_result(self.name, context, scores, applicable=bool(scores), signals=signals)
