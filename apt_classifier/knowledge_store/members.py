"""The knowledge store's two ensemble members (docs/KNOWLEDGE_STORE_DESIGN.md step 5; prereg
``knowledge_store_decisive_2026-11-04``, ``store_arm``).

Both read an artifact that ``scripts/knowledge_store.py members`` writes from the store (or from a
pin's copy, ``pins.py``): the statements of every post labelled to an ATT&CK group, the uses links
that count (rule links that are active, unclear links the local model confirmed), the posts that
introduce a malware name, and each post's dates and statement keys. Each case reads the store as it
stood before its own date: :meth:`StoreMembers.view` keeps only posts available strictly before
``as_of`` and leaves out the case's own post, its copies and, on the ATT&CK folds, the removed
actor and the posts ATT&CK cites for a held-out sentence.

- **Store text** (``store_text``): the free-text member's TF-IDF (:class:`text_group.TextGroupModel`,
  its parameters) over one document per ATT&CK group, built from the anonymised statements of the
  posts labelled to that group, scored against the case narrative. Like the other text members it
  never counts as independent evidence for the naming gate.
- **Store software** (``store_software``): the software member's rarity rule over the store's uses
  links that ATT&CK lacks, for the ATT&CK software the pipeline found in the case and the malware
  names the store knows (MISP lists, names posts introduce) written in the case text. A software's
  users are ATT&CK's and the store's (every actor, ATT&CK group or not); a group's score is the
  rarity-weighted share of the case's software the store links it to. It ranks on every link one
  publisher states without "possible"; for the naming gate it supports a group only through a
  link two publishers state (``naming_groups``), and it shares the software family with the ATT&CK
  member.

At weight 0 neither member is loaded and the combiner fingerprint is unchanged (this module lives
outside ``classifiers/``, which the fingerprint hashes).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import zlib
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.classifiers.software import COMMODITY_TOOL_WEIGHT_CAP
from apt_classifier.knowledge_store.sources import PUBLISHERS
from apt_classifier.text_group import DEFAULT_PARAMS as TEXT_PARAMS
from apt_classifier.text_group import TextGroupModel, prominence_log_prior

SCHEMA_VERSION = 1
KIND = "knowledge_store_members"
STORE_TEXT_NAME = "store_text"
STORE_SOFTWARE_NAME = "store_software"
# a post shares a case's statements when at least this share of the shorter side's keys match
COPY_SHARE = 0.5
COPY_MIN_STATEMENTS = 3
# words a statement key drops besides capitalised tokens (names on either side, masked or not)
_KEY_DROP = frozenset({"the", "a", "an", "adversary", "malware", "operation", "tool", "tools"})
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|\d+")
_SLICES = 8


def statement_key(text: str) -> int | None:
    """A name-insensitive key for the copy rule: the statement's words without capitalised
    tokens (actor, malware and vendor names, as written or masked) and the mask words; ``None``
    when fewer than four words remain."""
    words = [
        word.casefold()
        for word in _WORD.findall(text or "")
        if not word[0].isupper() and word.casefold() not in _KEY_DROP
    ]
    if len(words) < 4:
        return None
    digest = hashlib.blake2b(" ".join(words).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def statement_keys(texts: Iterable[str]) -> frozenset[int]:
    return frozenset(k for k in (statement_key(t) for t in texts) if k is not None)


def _day(value: date | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, date) else str(value)[:10]


@dataclass(frozen=True)
class Case:
    """What one case may read: posts available strictly before ``as_of`` (``None``: all), minus
    ``hidden`` post keys and the links and documents of ``removed_actors``."""

    as_of: str | None = None
    hidden: frozenset[str] = frozenset()
    removed_actors: frozenset[str] = frozenset()


@dataclass
class SoftwareSlice:
    # software id -> actor -> (publishers stating it, publishers stating it without "possible")
    pairs: dict[str, dict[str, tuple[frozenset[str], frozenset[str]]]] = field(default_factory=dict)
    names: frozenset[str] = frozenset()  # non-ATT&CK software ids valid in this slice


class StoreMembers:
    """The artifact's facts, sliced per case (a few slices are cached)."""

    def __init__(self, payload: Mapping[str, Any], kb: Any) -> None:
        if payload.get("kind") != KIND or int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("not a knowledge-store members artifact of this schema")
        self.metadata = dict(payload.get("metadata") or {})
        self.kb = kb
        self.posts: dict[str, dict[str, Any]] = {
            key: {
                "source": row[0],
                "available_from": row[1],
                "attack_cited": bool(row[2]),
                "keys": frozenset(row[3]),
            }
            for key, row in (payload.get("posts") or {}).items()
        }
        self.duplicates: dict[str, str] = dict(payload.get("duplicates") or {})
        self.text_rows: list[dict[str, Any]] = [dict(r) for r in payload.get("text") or ()]
        self.links: list[dict[str, Any]] = [dict(r) for r in payload.get("links") or ()]
        self.name_posts: list[dict[str, Any]] = [dict(r) for r in payload.get("names") or ()]
        self.software: dict[str, tuple[str, str]] = {
            sid: (str(v[0]), str(v[1])) for sid, v in (payload.get("software") or {}).items()
        }
        # every name posts introduce that counts as software at the latest date (casefolded):
        # the behaviour reading masks them in every arm
        self.masking_names: tuple[str, ...] = tuple(payload.get("masking_names") or ())
        self.introduced_share = float(self.metadata.get("introduced_share", 0.25))
        self._fingerprint = str(payload.get("fingerprint") or "")
        self._full_text: TextGroupModel | None = None
        self._text_slices: OrderedDict[Any, TextGroupModel | None] = OrderedDict()
        self._software_slices: OrderedDict[Case, SoftwareSlice] = OrderedDict()
        self._name_pattern = _surface_pattern(
            {sid: name for sid, (name, kind) in self.software.items() if kind in ("misp", "report")}
        )

    # ------------------------------------------------------------------ identity
    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @classmethod
    def load(cls, path: str | Path, kb: Any) -> StoreMembers | None:
        target = Path(path)
        if not target.is_file():
            return None
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        members = cls(payload, kb)
        if members.fingerprint != artifact_fingerprint(payload):
            raise ValueError("the artifact's fingerprint does not match its content")
        return members

    def compatible_with(self, kb: Any) -> tuple[bool, str]:
        wanted = str(self.metadata.get("attack_version") or "")
        have = str(getattr(kb, "version", "") or "")
        if wanted and have and wanted != have:
            return False, f"built on ATT&CK {wanted}, the knowledge base is {have}"
        return True, ""

    # ------------------------------------------------------------------ hiding
    def hidden_for(
        self, keys: Iterable[str] = (), statements: Iterable[str] = ()
    ) -> frozenset[str]:
        """A case's own post(s), every post of their copy group, and every post sharing at least
        ``COPY_SHARE`` of the case's statements (the store's copy rule, name-insensitive)."""
        own = {key for key in keys if key}
        canon = {self.duplicates.get(key, key) for key in own}
        hidden = set(own) | canon
        hidden.update(k for k, target in self.duplicates.items() if target in canon)
        case_keys = statement_keys(statements)
        if len(case_keys) >= COPY_MIN_STATEMENTS:
            for key, post in self.posts.items():
                if len(post["keys"]) < COPY_MIN_STATEMENTS:
                    continue
                shared = len(case_keys & post["keys"])
                if shared and shared / min(len(case_keys), len(post["keys"])) >= COPY_SHARE:
                    hidden.add(key)
        return frozenset(hidden)

    def cited_posts(self) -> frozenset[str]:
        return frozenset(key for key, post in self.posts.items() if post["attack_cited"])

    def _readable(self, key: str, case: Case) -> bool:
        post = self.posts.get(key)
        if post is None or key in case.hidden:
            return False
        return case.as_of is None or str(post["available_from"]) < case.as_of

    # ------------------------------------------------------------------ slices
    def text_slice(self, case: Case, kb: Any | None = None) -> TextGroupModel | None:
        kb = kb if kb is not None else self.kb
        return _cached(self._text_slices, (case, id(kb)), lambda: self._text_slice(case, kb))

    def software_slice(self, case: Case) -> SoftwareSlice:
        return _cached(self._software_slices, case, lambda: self._software_slice(case))

    def _full(self) -> TextGroupModel | None:
        if self._full_text is None and self.text_rows:
            groups = tuple(sorted(getattr(self.kb, "groups", {}) or {}))
            records = [
                {"text": r["text"], "source": "store", "owner": r["key"], "groups": [r["group"]]}
                for r in self.text_rows
            ]
            self._full_text = TextGroupModel(
                records, groups, prominence_log_prior(self.kb, groups), params=TEXT_PARAMS
            )
        return self._full_text

    def _text_slice(self, case: Case, kb: Any) -> TextGroupModel | None:
        """The text documents of this case: equal to a fresh fit on the kept statements (the
        vocabulary is reused and the terms the kept statements hold fewer than ``min_df`` times
        are zeroed, as ``TextGroupModel.without`` does)."""
        from scipy import sparse

        full = self._full()
        if full is None:
            return None
        groups = set(getattr(kb, "groups", {}) or {})
        rows = [
            i
            for i, r in enumerate(self.text_rows)
            if r["group"] in groups
            and r["group"] not in case.removed_actors
            and self._readable(r["key"], case)
        ]
        if not rows:
            return None
        counts = full._counts[rows].tocsc()
        df = np.diff(counts.indptr)
        min_df = int(full.params["min_df"])
        counts = (counts @ sparse.diags((df >= min_df).astype(np.float32))).tocsr()
        counts.eliminate_zeros()
        group_ids = tuple(sorted(groups))
        kept = [full.records[i] for i in rows]
        return TextGroupModel(
            kept,
            group_ids,
            prominence_log_prior(kb, group_ids),
            params=full.params,
            metadata={"slice": {"as_of": case.as_of, "statements": len(kept)}},
            _vectorizer=full._vectorizer,
            _counts=counts,
        )

    def _software_slice(self, case: Case) -> SoftwareSlice:
        names = self.introduced(case)
        found: dict[str, dict[str, tuple[set[str], set[str]]]] = defaultdict(dict)
        for link in self.links:
            if link["actor"] in case.removed_actors or not self._readable(link["key"], case):
                continue
            if link["kind"] == "report" and link["software_id"] not in names:
                continue
            publisher = PUBLISHERS.get(self.posts[link["key"]]["source"])
            publisher = publisher or self.posts[link["key"]]["source"]
            stated, naming = found[link["software_id"]].setdefault(link["actor"], (set(), set()))
            stated.add(publisher)
            if link["hedge"] != "possible":
                naming.add(publisher)
        pairs = {
            sid: {actor: (frozenset(a), frozenset(b)) for actor, (a, b) in actors.items()}
            for sid, actors in found.items()
        }
        valid = {
            sid
            for sid, (_, kind) in self.software.items()
            if kind == "misp" or (kind == "report" and sid in names)
        }
        return SoftwareSlice(pairs=pairs, names=frozenset(valid))

    def introduced(self, case: Case) -> set[str]:
        """Names posts introduced that count as software in this case: of the readable posts
        that mention the name, at least ``introduced_share`` introduce it (the store's rule)."""
        mentions: dict[str, list[int]] = defaultdict(list)
        for row in self.name_posts:
            if self._readable(row["key"], case):
                mentions[row["software_id"]].append(int(row["introduced"]))
        return {
            sid
            for sid, flags in mentions.items()
            if sum(flags) >= 1 and sum(flags) >= self.introduced_share * len(flags)
        }

    # ------------------------------------------------------------------ scoring
    def text_scores(self, case: Case, narrative: str, kb: Any | None = None) -> dict[str, float]:
        model = self.text_slice(case, kb)
        if model is None or not (narrative or "").strip():
            return {}
        return model.scores(narrative)

    def software_names_in(self, case: Case, text: str) -> set[str]:
        """The non-ATT&CK software ids the case text names (their own spelling, whole words)."""
        if self._name_pattern is None or not text:
            return set()
        sliced = self.software_slice(case)
        by_name = self._name_index
        found = {by_name.get(match.group(0)) for match in self._name_pattern.finditer(text)}
        return {sid for sid in found if sid and sid in sliced.names}

    @property
    def _name_index(self) -> dict[str, str]:
        index = getattr(self, "_names_cache", None)
        if index is None:
            index = {}
            for sid, (name, kind) in sorted(self.software.items()):
                if kind in ("misp", "report"):
                    index.setdefault(name, sid)
            self._names_cache = index
        return index

    def software_scores(
        self,
        case: Case,
        attack_ids: Iterable[str],
        text: str,
        kb: Any | None = None,
        *,
        tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    ) -> tuple[dict[str, float], frozenset[str], dict[str, tuple[str, ...]]]:
        """``(scores, naming_groups, signals)`` per ATT&CK group of ``kb``: the rarity-weighted
        share of the case's software the store links the group to where ATT&CK does not."""
        kb = kb if kb is not None else self.kb
        sliced = self.software_slice(case)
        groups = set(getattr(kb, "groups", {}) or {})
        software = getattr(kb, "software", {}) or {}
        observed = {str(s) for s in attack_ids} | self.software_names_in(case, text)
        n_groups = len(groups) or 1
        weights: dict[str, float] = {}
        attack_users: dict[str, set[str]] = {}
        for sid in observed:
            entry = software.get(sid)
            users = {g for g in (getattr(entry, "group_ids", ()) or ()) if g in groups}
            attack_users[sid] = users
            store_users = {
                actor
                for actor, (_, naming) in sliced.pairs.get(sid, {}).items()
                if naming and actor not in case.removed_actors
            }
            rarity = math.log((n_groups + 1) / (len(users | store_users) + 1)) + 1.0
            tool = str(getattr(entry, "software_type", "") or "") == "tool" or (
                self.software.get(sid, ("", ""))[1] == "attack_tool"
            )
            weights[sid] = min(rarity, tool_weight_cap) if tool else rarity
        total = math.fsum(weights.values())
        scores: dict[str, float] = {}
        naming: set[str] = set()
        signals: dict[str, tuple[str, ...]] = {}
        if total <= 0:
            return scores, frozenset(), signals
        for group_id in sorted(groups):
            linked = []
            corroborated = False
            for sid in sorted(observed):
                stated = sliced.pairs.get(sid, {}).get(group_id)
                if not stated or not stated[1] or group_id in attack_users.get(sid, set()):
                    continue
                linked.append(sid)
                corroborated = corroborated or len(stated[1]) >= 2
            if linked:
                scores[group_id] = math.fsum(weights[s] for s in linked) / total
                signals[group_id] = tuple(
                    f"{sid}:{self.software.get(sid, (sid, ''))[0]}" for sid in linked
                )
                if corroborated:
                    naming.add(group_id)
        return scores, frozenset(naming), signals


def _cached(cache: OrderedDict[Any, Any], key: Any, build: Any) -> Any:
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = cache[key] = build()
    if len(cache) > _SLICES:
        cache.popitem(last=False)
    return value


def _surface_pattern(names: Mapping[str, str]) -> re.Pattern[str] | None:
    spellings = sorted({n for n in names.values() if n and len(n) >= 3}, key=lambda n: (-len(n), n))
    if not spellings:
        return None
    return re.compile(
        r"(?<![\w.@/\\-])(?:" + "|".join(re.escape(n) for n in spellings) + r")(?![\w@-])"
    )


def citation_keys(bundle_path: str | Path) -> dict[tuple[str, str], frozenset[str]]:
    """``(G####, T####)`` -> the store keys of the posts ATT&CK cites for the group's procedure
    examples under that technique (the ``uses`` relationships of the raw STIX bundle): an
    ATT&CK fold hides them from the store members with its held-out sentences."""
    from apt_classifier.knowledge_store.listing import normalise_url

    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    ids: dict[str, str] = {}
    for item in bundle.get("objects", []):
        if item.get("type") in ("intrusion-set", "attack-pattern"):
            for ref in item.get("external_references", []) or []:
                if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
                    ids[item["id"]] = ref["external_id"]
    cited: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in bundle.get("objects", []):
        if item.get("type") != "relationship" or item.get("relationship_type") != "uses":
            continue
        source, target = ids.get(item.get("source_ref", "")), ids.get(item.get("target_ref", ""))
        if not (source and target and source.startswith("G") and target.startswith("T")):
            continue
        for ref in item.get("external_references", []) or []:
            if ref.get("url"):
                cited[(source, target)].add(normalise_url(ref["url"]))
    return {pair: frozenset(keys) for pair, keys in cited.items()}


def artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash of the artifact's content (everything but the fingerprint itself)."""
    digest = hashlib.sha256()
    body = {k: v for k, v in payload.items() if k != "fingerprint"}
    digest.update(json.dumps(body, sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()[:24]


def write_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    payload = {**payload, "kind": KIND, "schema_version": SCHEMA_VERSION}
    payload["fingerprint"] = artifact_fingerprint(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return target


# ---------------------------------------------------------------------------- pipeline members
def store_text_votes(settings: Any) -> bool:
    return float(getattr(settings, "store_text_weight", 0.0) or 0.0) > 0


def store_software_votes(settings: Any) -> bool:
    return float(getattr(settings, "store_software_weight", 0.0) or 0.0) > 0


def store_members_votes(settings: Any) -> bool:
    return store_text_votes(settings) or store_software_votes(settings)


def members_path(settings: Any, pin: str | None = None) -> Path:
    from apt_classifier.knowledge_store.pins import pin_root

    root = Path(getattr(settings, "knowledge_store_dir", "data/knowledge_store"))
    return (pin_root(root, pin) if pin else root) / "store_members.json.gz"


def load_store_members_for(
    settings: Any, kb: Any
) -> tuple[dict[str | None, StoreMembers] | None, str]:
    """The members' artifacts for the engine, fail-closed: ``({pin: members}, status)``. The
    store's own artifact serves cases without a date (and the app); with
    ``knowledge_store_pinned`` every pin's artifact loads too, for dated cases."""
    if not store_members_votes(settings):
        return None, "disabled (weight 0)"
    from apt_classifier.knowledge_store.pins import PINS

    wanted: list[str | None] = [None]
    if bool(getattr(settings, "knowledge_store_pinned", False)):
        wanted += sorted(PINS)
    loaded: dict[str | None, StoreMembers] = {}
    for pin in wanted:
        path = members_path(settings, pin)
        try:
            members = StoreMembers.load(path, kb)
        except (
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            OSError,
            EOFError,
            zlib.error,
        ) as error:
            return None, f"invalid artifact {path}: {error}"
        if members is None:
            return None, f"artifact not found: {path} (run scripts/knowledge_store.py members)"
        ok, reason = members.compatible_with(kb)
        if not ok:
            return None, f"incompatible artifact {path}: {reason}"
        loaded[pin] = members
    return loaded, "active"


def store_bound_fingerprint(
    base: str, members: Mapping[str | None, StoreMembers], weights: Mapping[str, float]
) -> str:
    """The combiner pipeline fingerprint when a store member votes: the base fingerprint, every
    loaded artifact's fingerprint, the members' weights and this module's source."""
    digest = hashlib.sha256()
    digest.update(base.encode())
    for pin in sorted(members, key=lambda p: p or ""):
        digest.update(f"{pin}:{members[pin].fingerprint}".encode())
    digest.update(json.dumps(dict(sorted(weights.items())), sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:24]


class StoreState:
    """The case the store members read (shared by both members, set by the engine): which
    artifact (a pin for a dated case) and which :class:`Case`. The app reads the whole store."""

    def __init__(self, members: Mapping[str | None, StoreMembers] | None = None) -> None:
        self.members = dict(members or {})
        self.pin: str | None = None
        self.case = Case()

    def set_case(
        self,
        *,
        as_of: date | str | None = None,
        keys: Sequence[str] = (),
        statements: Sequence[str] = (),
        removed_actors: Iterable[str] = (),
        hidden: Iterable[str] = (),
        pinned: bool = True,
    ) -> Case:
        from apt_classifier.knowledge_store.pins import pin_for

        day = _day(as_of)
        pin = pin_for(day) if (day and pinned and pin_for(day) in self.members) else None
        self.pin = pin
        current = self.current
        extra = current.hidden_for(keys, statements) if current is not None else frozenset()
        self.case = Case(
            as_of=day,
            hidden=frozenset(hidden) | extra,
            removed_actors=frozenset(removed_actors),
        )
        return self.case

    def clear(self) -> None:
        self.pin = None
        self.case = Case()

    @property
    def current(self) -> StoreMembers | None:
        return self.members.get(self.pin) or self.members.get(None)


class StoreTextClassifier:
    """Ensemble member: the store's text documents scored against the case narrative, for the
    eligible groups sharing a term with it. It cannot make a group eligible."""

    name = STORE_TEXT_NAME

    def __init__(self, state: StoreState) -> None:
        self.state = state

    def classify(self, context: Any) -> Any:
        from apt_classifier.classifiers.base import build_result, group_ids

        members = self.state.current
        narrative = str(getattr(context, "narrative", "") or "")
        if members is None or not narrative.strip():
            return build_result(self.name, context, {}, applicable=False)
        allowed = set(group_ids(context))
        scores = members.text_scores(self.state.case, narrative, kb=context.kb)
        scores = {g: s for g, s in scores.items() if g in allowed}
        signals = {g: (f"store_text:p={s:.3f}",) for g, s in scores.items()}
        return build_result(self.name, context, scores, applicable=bool(scores), signals=signals)


class StoreSoftwareClassifier:
    """Ensemble member: the store's uses links that ATT&CK lacks for the case's software. The
    groups it may support for the naming gate (links two publishers state) are left on the
    context as ``store_naming_groups``. It cannot make a group eligible."""

    name = STORE_SOFTWARE_NAME

    def __init__(
        self, state: StoreState, tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP
    ) -> None:
        self.state = state
        self.tool_weight_cap = float(tool_weight_cap)

    def classify(self, context: Any) -> Any:
        from apt_classifier.classifiers.base import build_result, group_ids

        members = self.state.current
        context.store_naming_groups = frozenset()
        if members is None:
            return build_result(self.name, context, {}, applicable=False)
        allowed = set(group_ids(context))
        scores, naming, signals = members.software_scores(
            self.state.case,
            getattr(context, "software_ids", set()) or set(),
            str(getattr(context, "narrative", "") or ""),
            kb=context.kb,
            tool_weight_cap=self.tool_weight_cap,
        )
        scores = {g: s for g, s in scores.items() if g in allowed}
        context.store_naming_groups = frozenset(g for g in naming if g in scores)
        signals = {g: signals[g] for g in scores}
        return build_result(self.name, context, scores, applicable=bool(scores), signals=signals)
