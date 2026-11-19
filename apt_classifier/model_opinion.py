"""A cloud model's answer as a second opinion beside the product's own (display only).

The analyst asks for it with a button. One paid call puts the question that was tested offline
(docs 21.3.28-21.3.31, ``scripts/fresh_llm_attribution.py``) to the model: which ATT&CK group
carried out the activity in the case's statements? The answer is read the way the registered
scorer reads it (``scripts/fresh_h2h_score.py --matching name``): every candidate is matched by
its name alone, because the model writes real actors ATT&CK does not list under invented group
ids. A "confident name" is the registered rule and nothing more: the model decides to name its
first candidate, that candidate is an ATT&CK group and the model states at least 0.99 for it.
The cut-off was chosen for GPT-5.6 Sol's own numbers, so the rule applies to that model at that
effort only.

Before the call every ATT&CK group name, alias and group id in the statements is hidden, as the
prompt tells the model it has been, together with the short forms reports write ("Lazarus" for
Lazarus Group) and spellings with or without a space or hyphen ("APT 28", "Fancy-Bear"). That is
less than the tested text had hidden: vendor names ATT&CK does not list, malware names (unless
ATT&CK lists the same word as a group name, as for Carbanak or Akira), camel-case names split in
two ("Muddy Water") and a report's own attribution prose stay. The offline numbers therefore do
not carry over to a product case as they stand.

Nothing here reaches classifier results, the fusion, the combiner, the naming gate, the
confidence labels or an export, and no engine module imports this file (only ``service.py``
does). It is never answered by a local model and never called during an analysis or a rescore.
The probability shown is the model's own statement; it is not calibrated.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import re
import textwrap
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from apt_classifier.evidence.models import EvidenceStatus
from apt_classifier.llm import OpenRouterError, is_local_model
from apt_classifier.overlap_cluster import SAME_ACTIVITY_LINKS, link_kind

PROMPT_VERSION = 1
REGISTERED_MODEL = "openai/gpt-5.6-sol"
REGISTERED_EFFORT = "medium"
# the registered naming cut-off, chosen on GPT-5.6 Sol's own stated numbers
CONFIDENT_PROBABILITY = 0.99
# a stated probability written as 0.99 is at least 0.99
PROBABILITY_TOLERANCE = 1e-9
# as tested (the largest answer measured took 4,628 tokens, most of them reasoning)
MAX_TOKENS = 16000
SCHEMA_NAME = "attribution"
MAX_STATEMENTS = 400
# the prompt asks for up to three and the tested runner kept three
MAX_CANDIDATES = 3
DECISIONS = ("name", "open", "not_in_attack")

# SYSTEM, PROMPT and SCHEMA are the registered runner's (scripts/fresh_llm_attribution.py),
# copied because that script imports the benchmark engine; a test keeps them equal.
SYSTEM = "You are an experienced cyber threat intelligence analyst."
PROMPT = """Below are the statements of one threat report, in report order. Every threat-actor \
name has been replaced with "the adversary", and statements about the attribution itself may \
have been removed.

Which MITRE ATT&CK Enterprise group (ATT&CK v20, group IDs G0001-G1099) carried out the activity \
described?

Answer with up to three candidate groups, most likely first, each with its ATT&CK group ID, its \
ATT&CK name and your probability, and one decision:
- "name": you would name the first candidate as the responsible group in a report to a customer;
- "open": the evidence does not single out one group;
- "not_in_attack": the activity most likely belongs to an actor ATT&CK does not document as a \
group.

Statements:
{statements}"""
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "group_name": {"type": "string"},
                    "probability": {"type": "number"},
                },
                "required": ["group_id", "group_name", "probability"],
                "additionalProperties": False,
            },
        },
        "decision": {"type": "string", "enum": ["name", "open", "not_in_attack"]},
        "rationale": {"type": "string"},
    },
    "required": ["candidates", "decision", "rationale"],
    "additionalProperties": False,
}
NOTE = (
    "The model's own answer and its own stated probability. It is not calibrated and "
    "is never part of the product's decision."
)

REPLACEMENT = "the adversary"
CLUSTER_STAGE = "known_cluster_hypothesis"
# the registered combination also read the product's first five
SHORTLIST = 5
AGREEMENT_KINDS = (
    "no_candidate",
    "outside_attack",
    "no_product_leader",
    "same",
    "named_pair",
    "same_activity",
    "shortlist",
    "different",
)
_AGREEMENT_TEXT = {
    "no_candidate": "The model gave no candidate group.",
    "outside_attack": (
        "The model's first choice does not match one ATT&CK group by its name, so it cannot be "
        "compared with the product's answer."
    ),
    "no_product_leader": "The product has no first choice for this case to compare with.",
    "same": "Same group as the product's first choice.",
    "named_pair": "One of the two groups the product names together.",
    "same_activity": (
        "Not the product's first choice, but ATT&CK documents the two groups as overlapping "
        "activity."
    ),
    "shortlist": "Not the product's first choice, but among the product's first five.",
    "different": (
        "A different group from the product's first choice, and not among the product's first five."
    ),
}

# Head words of "<X> Group/Team" names that reports write on their own: the tested
# anonymiser's reviewed list (scripts/report_anonymizer.py SHORT_FORMS). Not derived from the
# names: "Cobalt" or "Gold" would wipe ordinary text.
SHORT_FORMS = ("Gamaredon", "Lazarus", "Sandworm", "Winnti", "Gwisin")
# the group ids the prompt names (G0001-G1099); other G#### tokens are model numbers
MAX_GROUP_ID = 1099

_ARTICLE_BEFORE_ADVERSARY_RE = re.compile(r"\b(?:an?|the)\s+(the adversary)", re.IGNORECASE)
# an ATT&CK group id: with its brackets ("APT28 (G0007)") or on its own, also after a slash or a
# hyphen (an ATT&CK link, "APT28/G0007"); never inside a longer token (a file name, a hash)
_GROUP_ID_RE = re.compile(
    r"\s*[(\[]\s*(?P<bracketed>G\d{4})\s*[)\]]"
    r"|\s*(?<![\w.])(?P<bare>G\d{4})(?!\w|\.\w)"
)
# where name_key reads no difference: between a name's words, and between letters and digits
_NAME_JOINT_RE = re.compile(r"[\s-]+|(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_WHITESPACE_RE = re.compile(r"\s+")


class ModelOpinionUnavailable(RuntimeError):
    """The second opinion cannot be asked here: no OpenRouter key, or a local model id."""


# ---------------------------------------------------------------------------- name matching
def name_key(name: str) -> str:
    """Lower-case letters and digits only, so "APT 28", "apt-28" and "APT28" are one name."""
    return "".join(character for character in name.lower() if character.isalnum())


def group_names(groups: Mapping[str, Any]) -> dict[str, str]:
    """Normalised ATT&CK group names and aliases -> group id. A name two groups share is left
    out: it cannot say which of them the model meant."""
    index: dict[str, set[str]] = defaultdict(set)
    for group_id, group in groups.items():
        for name in (group.name, *(group.aliases or ())):
            if name and name_key(name):
                index[name_key(name)].add(group_id)
    return {key: next(iter(ids)) for key, ids in index.items() if len(ids) == 1}


# ---------------------------------------------------------------------------- the case text
def case_statements(ledger: Any) -> list[str]:
    """The statements the product's own ensemble reads: the texts of the positive
    observations, in ledger order (``retrieve.build_classifier_context``)."""
    statements: list[str] = []
    for observation in getattr(ledger, "observations", None) or ():
        if getattr(observation, "status", None) != EvidenceStatus.POSITIVE:
            continue
        text = str(getattr(observation, "text", "") or "").strip()
        if text:
            statements.append(text)
    return statements


def _actor_names(kb: Any) -> tuple[str, ...]:
    """Every ATT&CK group name and alias as ATT&CK writes it, and the short forms of the
    "<X> Group/Team" names in ``SHORT_FORMS``, longest first."""
    groups = getattr(kb, "groups", None) or {}
    names = {
        name.strip()
        for group in groups.values()
        for name in (group.name, *(group.aliases or ()))
        if name and name.strip()
    }
    names |= {
        short
        for short in SHORT_FORMS
        if any(name.casefold().startswith(short.casefold() + " ") for name in names)
    }
    return tuple(sorted(names, key=lambda item: (-len(item), item.casefold())))


def _spelled(name: str) -> str:
    """A name with or without a space or hyphen wherever ``name_key`` reads none: "APT28" also
    as "APT 28" and "APT-28", "Fancy Bear" also as "FancyBear" and "Fancy-Bear"."""
    return "[ -]?".join(re.escape(part) for part in _NAME_JOINT_RE.split(name) if part)


@lru_cache(maxsize=4)
def _name_rule(names: tuple[str, ...]) -> tuple[re.Pattern[str] | None, frozenset[str]]:
    """One pattern over every name (the shape of ``text_group.evidence_name_pattern`` with
    ``before_hyphen``; that module loads numpy, so the pattern is written here) and the
    spellings ATT&CK itself uses. Kept per name list: the pattern has several hundred names."""
    if not names:
        return None, frozenset()
    pattern = re.compile(
        r"(?<![\w.-])(?:" + "|".join(_spelled(name) for name in names) + r")"
        r"(?P<possessive>['’]s)?(?!\w|\.\w)",
        re.IGNORECASE,
    )
    return pattern, frozenset(names)


def _mask_one(
    text: str, pattern: re.Pattern[str] | None, spellings: frozenset[str]
) -> tuple[str, int]:
    hidden = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal hidden
        possessive = match.group("possessive") or ""
        word = match.group(0).removesuffix(possessive)
        # "play", "tick", "silence": an ordinary word, unless ATT&CK writes the name that way
        if word.isalpha() and word.islower() and word not in spellings:
            return match.group(0)
        hidden += 1
        return REPLACEMENT + possessive

    # first, so a name written over a line break or with two spaces is still found
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if pattern is not None:
        text = pattern.sub(replace, text)
        if hidden:
            text = _ARTICLE_BEFORE_ADVERSARY_RE.sub(r"\1", text)
    ids = 0

    def remove_id(match: re.Match[str]) -> str:
        nonlocal ids
        number = int((match.group("bracketed") or match.group("bare"))[1:])
        if not 1 <= number <= MAX_GROUP_ID:
            return match.group(0)  # "Netgear G3100": not a group id
        ids += 1
        return ""

    text = _GROUP_ID_RE.sub(remove_id, text)
    return _WHITESPACE_RE.sub(" ", text).strip(), hidden + ids


def mask_statements(statements: Sequence[str], kb: Any) -> tuple[list[str], int]:
    """The statements with every ATT&CK group name and alias replaced by "the adversary" and
    every ATT&CK group id removed, one masked statement per statement, and how many names and
    ids were hidden.

    A name is not replaced after a word character, a dot or a hyphen, nor before a dot and a
    word character ("APT28.dll", "apt28.example.net"); before a hyphen or a bracket it is
    ("APT28-linked", and so "apt28-c2.example.net"). An all-lower-case single word ("play",
    "tick", "lazarus") is kept unless ATT&CK writes the name that way; names with digits or
    several words are replaced in any case, and so is a capitalised ordinary word that is also
    an ATT&CK name ("Google Play", "Chromium-based"). Group ids G0001-G1099 are removed, also
    in an ATT&CK link or after a slash; other G#### tokens (model numbers) stay."""
    pattern, spellings = _name_rule(_actor_names(kb))
    masked: list[str] = []
    hidden = 0
    for statement in statements:
        text, count = _mask_one(str(statement), pattern, spellings)
        masked.append(text)
        hidden += count
    return masked, hidden


def prompt_for(statements: Sequence[str]) -> str:
    return PROMPT.format(
        statements="\n".join(f"{index}. {text}" for index, text in enumerate(statements, 1))
    )


def text_fingerprint(statements: Sequence[str]) -> str:
    """sha256 of the statements joined by line breaks."""
    return hashlib.sha256("\n".join(statements).encode("utf-8", "surrogatepass")).hexdigest()


def _digest(items: Sequence[str]) -> str:
    # unlike text_fingerprint, a statement holding a line break cannot pass for two statements
    encoded = json.dumps(list(items), ensure_ascii=False).encode("utf-8", "surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


# A case view asks on every read whether the answer is stale, and the name pattern takes about
# 0.1 s on a long case, so the fingerprint of what a call would send is kept per knowledge base
# and case text. The table holds hashes only, never case text.
_SENT_LIMIT = 64
_sent_fingerprints: dict[tuple[str, str], str] = {}
_sent_lock = threading.Lock()


def _sent_fingerprint(ledger: Any, kb: Any) -> str:
    """The fingerprint of the masked statements a call on this ledger would send now."""
    statements = case_statements(ledger)[:MAX_STATEMENTS]
    key = (_digest(_actor_names(kb)), _digest(statements))
    with _sent_lock:
        known = _sent_fingerprints.get(key)
    if known is None:
        known = text_fingerprint(mask_statements(statements, kb)[0])
        with _sent_lock:
            if len(_sent_fingerprints) >= _SENT_LIMIT:
                _sent_fingerprints.clear()
            _sent_fingerprints[key] = known
    return known


# ---------------------------------------------------------------------------- the answer
def _shown(text: str) -> str:
    """The model's text as JSON can carry it: a lone surrogate (a broken \\u escape) would
    fail every later view of the case."""
    return text.encode("utf-8", "replace").decode("utf-8")


def _probability(value: Any) -> float | None:
    """The model's stated probability; None when it is not a number between 0 and 1."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def read_answer(parsed: Any, names: Mapping[str, str], kb: Any) -> dict[str, Any]:
    """The model's answer, checked, with every candidate matched to ATT&CK by its name only.

    The model's own group id is ignored: it writes real actors ATT&CK does not list under
    invented ids (docs 21.3.28). ``names`` is ``group_names(kb.groups)``. At most three
    candidates are kept, in the model's order, so the first one is the model's first choice
    whether ATT&CK lists it or not."""
    if not isinstance(parsed, Mapping):
        raise OpenRouterError("the model's answer is not a JSON object")
    decision = parsed.get("decision")
    if decision not in DECISIONS:
        raise OpenRouterError(f"the model's answer has an unknown decision: {decision!r}")
    raw = parsed.get("candidates")
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise OpenRouterError("the model's answer has no list of candidate groups")
    groups = getattr(kb, "groups", None) or {}
    candidates: list[dict[str, Any]] = []
    for item in raw[:MAX_CANDIDATES]:
        name = str(item.get("group_name") or "").strip()
        group_id = names.get(name_key(name))
        group = groups.get(group_id) if group_id else None
        candidates.append(
            {
                "name": _shown(name),
                "group_id": group_id if group is not None else None,
                "attack_name": str(group.name) if group is not None else None,
                "probability": _probability(item.get("probability")),
            }
        )
    return {
        "decision": str(decision),
        "rationale": _shown(str(parsed.get("rationale") or "").strip()),
        "candidates": candidates,
    }


def _registered(model: str, effort: str) -> bool:
    """The naming cut-off was chosen for one model at one effort; it says nothing of others."""
    return model == REGISTERED_MODEL and effort == REGISTERED_EFFORT


def is_confident_name(
    decision: str, first: Mapping[str, Any] | None, *, model: str, effort: str
) -> bool:
    """The registered rule: the model decides to name its first candidate, that candidate is an
    ATT&CK group and the model states at least 0.99 for it, for the registered model and effort
    only. A later candidate never counts."""
    if not _registered(model, effort) or decision != "name" or not first:
        return False
    probability = _probability(first.get("probability"))
    return (
        bool(first.get("group_id"))
        and probability is not None
        and probability >= CONFIDENT_PROBABILITY - PROBABILITY_TOLERANCE
    )


def _agreement_kind(kb: Any, state: Any, first: Mapping[str, Any] | None) -> str:
    if not first:
        return "no_candidate"
    group_id = first.get("group_id")
    if not group_id:
        return "outside_attack"
    decision = getattr(state, "attribution_decision", None)
    leader = getattr(decision, "leading_group_id", None)
    if not leader:
        return "no_product_leader"
    if group_id == leader:
        return "same"
    stage = getattr(decision, "stage", None)
    named = tuple(getattr(decision, "named_cluster", None) or ())
    if str(getattr(stage, "value", stage)) == CLUSTER_STAGE and group_id in named:
        return "named_pair"
    try:
        linked = link_kind(kb, leader, group_id) in SAME_ACTIVITY_LINKS
    except Exception:  # noqa: BLE001 - an overlap hint never breaks the block
        linked = False
    if linked:
        return "same_activity"
    fusion = getattr(state, "deterministic_fusion", None) or getattr(state, "fusion", None)
    entries = tuple(getattr(fusion, "entries", None) or ())[:SHORTLIST]
    if any(getattr(entry, "group_id", None) == group_id for entry in entries):
        return "shortlist"
    return "different"


def agreement(
    kb: Any, state: Any, first: Mapping[str, Any] | None, *, scope: str = "documented"
) -> dict[str, str]:
    """How the model's first choice relates to the product's answer: one of
    ``AGREEMENT_KINDS`` (the first that holds, in that order) and a sentence for it. A pure
    read of the state.

    ``scope`` is the cluster scope the service runs under. No tier needs it today
    (``overlap_cluster.link_kind`` takes none): a pair the product named under its scope is in
    the decision, and what ATT&CK documents about two groups does not depend on it."""
    kind = _agreement_kind(kb, state, first)
    return {"kind": kind, "text": _AGREEMENT_TEXT[kind]}


@dataclass(frozen=True)
class ModelOpinion:
    asked_at: str  # ISO 8601, UTC
    model: str  # the model that answered
    requested_model: str
    effort: str
    prompt_version: int
    decision: str
    rationale: str
    candidates: tuple[dict[str, Any], ...]
    statements_sent: int
    statements_total: int
    names_masked: int
    fingerprint: str  # text_fingerprint of the masked statements that were sent
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    @property
    def first(self) -> dict[str, Any] | None:
        """The model's own first choice, whether ATT&CK lists it or not."""
        return dict(self.candidates[0]) if self.candidates else None

    @property
    def confident_name(self) -> bool:
        return is_confident_name(self.decision, self.first, model=self.model, effort=self.effort)


def ask(
    client: Any, kb: Any, ledger: Any, *, model: str, effort: str, now: str | None = None
) -> ModelOpinion:
    """One call: the case's statements, ATT&CK group names hidden, with the tested question.

    Raises ``ModelOpinionUnavailable`` without an OpenRouter key or for a local model id (the
    client would otherwise let its local fallback model answer in the cloud model's name),
    ``ValueError`` for a case without statements and ``OpenRouterError`` for a failed call or
    an answer that cannot be read."""
    if client is None or not getattr(client, "api_key", None):
        raise ModelOpinionUnavailable(
            "the model's second opinion needs an OpenRouter key (OPENROUTER_API_KEY)"
        )
    if is_local_model(model):
        raise ModelOpinionUnavailable(
            f"the model's second opinion needs a cloud model; {model} is a local model"
        )
    statements = case_statements(ledger)
    if not statements:
        raise ValueError("the case has no statements to send")
    masked, hidden = mask_statements(statements[:MAX_STATEMENTS], kb)
    if not any(masked):  # nothing but group ids: no paid call for an empty question
        raise ValueError("the case has no statements to send")
    # no seed: the question was tested without one
    response = client.complete(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_for(masked)},
        ],
        model=model,
        effort=effort,
        schema=copy.deepcopy(SCHEMA),
        schema_name=SCHEMA_NAME,
        max_tokens=MAX_TOKENS,
    )
    answered_by = str(getattr(response, "model", "") or "")
    if is_local_model(answered_by):
        raise OpenRouterError(
            f"a local model ({answered_by}) answered in place of {model}; its answer is not shown"
        )
    answer = read_answer(response.parsed, group_names(getattr(kb, "groups", None) or {}), kb)
    usage = getattr(response, "usage", None)
    return ModelOpinion(
        asked_at=now or datetime.now(UTC).isoformat(timespec="seconds"),
        model=answered_by,
        requested_model=model,
        effort=effort,
        prompt_version=PROMPT_VERSION,
        decision=answer["decision"],
        rationale=answer["rationale"],
        candidates=tuple(answer["candidates"]),
        statements_sent=len(masked),
        statements_total=len(statements),
        names_masked=hidden,
        fingerprint=text_fingerprint(masked),
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cost_usd=float(getattr(usage, "cost_usd", 0.0) or 0.0),
    )


def payload(
    opinion: ModelOpinion | None, kb: Any, state: Any, *, scope: str = "documented"
) -> dict[str, Any] | None:
    """The block the API shows for a stored answer, or None when there is none. ``stale`` says
    that a call made now would send other text than the answer was given for."""
    if opinion is None:
        return None
    first = opinion.first
    return {
        "asked_at": opinion.asked_at,
        "model": opinion.model,
        "effort": opinion.effort,
        "prompt_version": opinion.prompt_version,
        "decision": opinion.decision,
        "first": first,
        "candidates": [dict(candidate) for candidate in opinion.candidates],
        "confident_name": opinion.confident_name,
        "confident_rule": {
            "probability": CONFIDENT_PROBABILITY,
            "applies": _registered(opinion.model, opinion.effort),
        },
        "rationale": opinion.rationale,
        "agreement": agreement(kb, state, first, scope=scope),
        "statements_sent": opinion.statements_sent,
        "statements_total": opinion.statements_total,
        "names_masked": opinion.names_masked,
        "cost_usd": opinion.cost_usd,
        "prompt_tokens": opinion.prompt_tokens,
        "completion_tokens": opinion.completion_tokens,
        "stale": _sent_fingerprint(getattr(state, "ledger", None), kb) != opinion.fingerprint,
        "note": NOTE,
    }


# the rules the registration pins with the question: what is sent and how the answer is read
_PINNED_RULES = (
    "case_statements",
    "_actor_names",
    "_spelled",
    "_name_rule",
    "_mask_one",
    "mask_statements",
    "prompt_for",
    "name_key",
    "group_names",
    "_probability",
    "read_answer",
    "_registered",
    "is_confident_name",
)


def _rule_code(name: str) -> str:
    """A function's code without its comments and docstring, so only a change of rule counts."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(globals()[name])))
    function = tree.body[0]
    body = getattr(function, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        function.body = body[1:] or [ast.Pass()]  # type: ignore[attr-defined]
    return ast.dump(function)


def registration_fingerprint() -> str:
    """One hash over what the registration pins (``model_second_opinion_2026-11-11``): the
    prompt, the schema, the model, the effort, the naming cut-off, the token limit, the statement
    cap, and the rules that build the text and read the answer (masking, name matching, the
    confident-name rule). A change to any of them changes it; comments and docstrings do not."""
    pinned = {
        "prompt_version": PROMPT_VERSION,
        "system": SYSTEM,
        "prompt": PROMPT,
        "schema": SCHEMA,
        "schema_name": SCHEMA_NAME,
        "model": REGISTERED_MODEL,
        "effort": REGISTERED_EFFORT,
        "confident_probability": CONFIDENT_PROBABILITY,
        "max_tokens": MAX_TOKENS,
        "max_statements": MAX_STATEMENTS,
        "short_forms": list(SHORT_FORMS),
        "max_group_id": MAX_GROUP_ID,
        "replacement": REPLACEMENT,
        "patterns": [
            _GROUP_ID_RE.pattern,
            _NAME_JOINT_RE.pattern,
            _ARTICLE_BEFORE_ADVERSARY_RE.pattern,
        ],
        "rules": {name: _rule_code(name) for name in _PINNED_RULES},
    }
    encoded = json.dumps(pinned, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
