"""Reading facts from a stored post (docs/KNOWLEDGE_STORE_DESIGN.md, step 2).

What a post is about (the registered label rule of the fresh sets), which software each actor
uses, and what it says about actor names. Everything here is rules. A sentence the rules cannot
settle (two actors, a comparison, a negation) is kept as *unclear* and never counts until the
local model or a person has read it.

Actor names come from the caller: the fresh-set builder's dictionary (ATT&CK names and aliases,
MISP threat-actor synonyms) and its vendor cluster codes, so the store and the test sets name
actors alike. Software names come from three places:

- ATT&CK software names and aliases, under the app's own matching rules
  (``evidence.iocs``: generic aliases skipped; an alias ATT&CK's prose uses as an ordinary word
  matches only in its exact spelling, and a title-case one never at the start of a sentence);
- the MISP galaxy's malware and tool lists (CC0 / BSD; Malpedia's list is non-commercial and is
  not read);
- names a post gives a malware ("the HOOKEDGE backdoor", "a loader dubbed PteroSand"), collected
  over the whole store first (``introduced_names``) and then matched in their own spelling.

A name that is an ordinary English word, a word of ATT&CK's technique text (PowerShell, LSASS),
a publisher or an actor never counts as software. A name a post introduces counts, as of a date,
only when at least ``INTRODUCED_SHARE`` of the earlier posts that mention it introduce it as a
malware: technology words (Golang, MITRE, WhatsApp, Log4j) are introduced in a few of the many
posts that mention them, new malware in most (``KnowledgeStore.introduced_software``).

ATT&CK names that look like words match more strictly than in ``evidence.iocs``: a malware name
with inner capitals (PS1, OSInfo, MEDUSA) never in all-lowercase text, so not on ".ps1" (the
app's rule of 2026-10-02, docs 5.5); a name that is an ordinary word (Royal, Epic) only in its
own spelling and never title case at a sentence start; a command name that is a word (route)
only as ``route.exe``; a country or capital (Cuba), or a group alias that is also a software alias
(Sofacy, Snake), only before a malware noun; a tool name of three letters or fewer (Reg, Arp) only
as a command ("reg save", "arp -a", "cmd.exe"); never before a trademark sign ("Aurora®").
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EXTRACTOR_VERSION = 4
MISP_SOFTWARE_FILES = ("tool", "rat", "backdoor", "botnet", "stealer", "ransomware")
# a single-actor post links its actor to software it names at least this often
REPORT_LINK_MIN_MENTIONS = 2
# share of the posts mentioning a name a post introduced that must introduce it (see above)
INTRODUCED_SHARE = 0.25

_TOKEN_RE = re.compile(r"(?<![\w.@/\\])[A-Za-z][A-Za-z0-9]*(?![\w@])")
_NAME_TOKEN_RE = re.compile(r"(?<![\w.@/\\-])[A-Za-z][A-Za-z0-9_-]*[A-Za-z0-9](?![\w@-])")
_MALWARE_NOUNS = (
    r"backdoors?|loaders?|malware|implants?|RATs?|stealers?|droppers?|downloaders?|variants?|"
    r"famil(?:y|ies)|frameworks?|toolkits?|payloads?|trojans?|wipers?|ransomware|webshells?|"
    r"rootkits?|keyloggers?|infostealers?|botnets?|worms?|utility|utilities|tools?"
)
_NAME_BEFORE_NOUN_RE = re.compile(
    r"\b(?P<name>[A-Z][\w-]*[A-Za-z0-9])(?:\s+(?:" + _MALWARE_NOUNS + r"))\b"
)
_NAME_AFTER_CUE_RE = re.compile(
    r"\b(?:" + _MALWARE_NOUNS + r")\s+(?:(?:that|which)\s+)?(?:we\s+(?:have\s+)?)?"
    r"(?:called|named|dubbed|known as|tracked as)\s+[\"“']?(?P<name>[A-Z][\w-]*[A-Za-z0-9])"
)
_MALWARE_NOUN_RE = re.compile(r"(?:" + _MALWARE_NOUNS + r")", re.IGNORECASE)
# a place name is a malware name only before a malware noun ("Cuba ransomware")
_NOUN_AFTER_RE = re.compile(r"\s+(?:" + _MALWARE_NOUNS + r")\b", re.IGNORECASE)
# a short name before "file" or "script" is a file type ("a PS1 script", "the HTA file")
_FILE_AFTER_RE = re.compile(r"\s+(?:files?|scripts?|extensions?)\b", re.IGNORECASE)
_EXE_AFTER_RE = re.compile(r"\.exe\b", re.IGNORECASE)
# a short tool name is the command only before ".exe", a switch or a lower-case argument
_COMMAND_AFTER_RE = re.compile(r"\.(?:exe|EXE)\b|\s+(?:[/-]\w|[a-z]{2,}\b)")
_PRODUCT_AFTER_RE = re.compile(r"[®™]")
# an actor name before a malware noun or inside a detection name ("Silence.Downloader")
_NOT_ACTOR_AFTER_RE = re.compile(r"\.[A-Z]\w|\s+(?:" + _MALWARE_NOUNS + r")\b", re.IGNORECASE)
# ATT&CK tactic ids have the shape of Proofpoint cluster codes
_TACTIC_ID_RE = re.compile(r"TA00\d\d")
# the store's own sources and other security publishers: never a software name
PUBLISHER_WORDS = frozenset(
    word.casefold()
    for word in [
        "ESET",
        "Kaspersky",
        "Securelist",
        "Mandiant",
        "Microsoft",
        "Talos",
        "Cisco",
        "Proofpoint",
        "Zscaler",
        "Trend",
        "TrendMicro",
        "SentinelOne",
        "SentinelLabs",
        "Sekoia",
        "Symantec",
        "Broadcom",
        "Fortinet",
        "FortiGuard",
        "Rapid7",
        "Volexity",
        "Infoblox",
        "Intrinsec",
        "CISA",
        "FBI",
        "NSA",
        "JPCERT",
        "AhnLab",
        "ASEC",
        "Bitdefender",
        "Avast",
        "Gen",
        "Resecurity",
        "ThreatBook",
        "Unit42",
        "Palo",
        "Alto",
        "CrowdStrike",
        "Google",
        "GTIG",
        "Check",
        "Point",
        "Seqrite",
        "Quick",
        "Heal",
        "Arctic",
        "Wolf",
        "Silent",
        "Push",
        "Hunt",
        "StrikeReady",
        "Recorded",
        "Future",
        "Insikt",
        "Group-IB",
        "Elastic",
        "Huntress",
        "Sophos",
        "Morphisec",
        "Cybereason",
        "Secureworks",
        "Volexity",
        "Netskope",
        "Deep",
        "Instinct",
        "Lookout",
        "Kroll",
        "Dragos",
        "Claroty",
        "Nozomi",
        "Tenable",
        "Qualys",
        "Avira",
        "McAfee",
        "Trellix",
        "FireEye",
    ]
)

USE_RE = re.compile(
    r"\b(?:us(?:e|es|ed|ing)|utili[sz](?:e|es|ed|ing)|leverag(?:e|es|ed|ing)|employ(?:s|ed|ing)?"
    r"|deploy(?:s|ed|ing|ment)?|drop(?:s|ped|ping)?|deliver(?:s|ed|ing|y)?|install(?:s|ed|ing)?"
    r"|execut(?:e|es|ed|ing)|launch(?:es|ed|ing)?|load(?:s|ed|ing)?|distribut(?:e|es|ed|ing)"
    r"|spread(?:s|ing)?|push(?:es|ed|ing)?|rel(?:y|ies|ied|ying) on|turn(?:s|ed)? to"
    r"|adopt(?:s|ed)?|develop(?:s|ed|ing)?|wield(?:s|ed)?|arsenal|toolkit|toolset|tooling"
    r"|attributed to|tied to|belong(?:s|ing)? to|exclusive(?:ly)? to|unique to"
    r"|hallmark|signature (?:malware|tool|backdoor|implant))\b",
    re.IGNORECASE,
)
# a sentence that compares, denies or spreads the software over several actors is not a use
COMPARISON_RE = re.compile(
    r"\b(?:similar(?:ities)?|resembl\w*|overlap\w*|unlike|in contrast|whereas|previously"
    r" (?:attributed|reported|associated|linked|tied|documented|tracked)|also (?:used|seen|"
    r"deployed|observed) by|shared (?:by|with|among|between)|borrow\w*|cop(?:y|ied|ycat)\w*"
    r"|imitat\w*|mimic\w*|false flag|no (?:evidence|link|connection|overlap)|not (?:linked|"
    r"attributed|associated|related|tied|connected)|unrelated|other (?:actors|groups|threat "
    r"actors)|multiple (?:actors|groups|threat actors)|several (?:actors|groups)"
    r"|similarit(?:y|ies)|the same (?:[\w-]+ ){0,3}as|same time|ties (?:to|with)|tying"
    r"|connect(?:s|ed|ing)? (?:[\w-]+ ){0,3}(?:with|to)|another (?:cluster|actor|group|threat "
    r"actor|campaign)|other (?:extortion|ransomware|cybercrime|criminal) (?:groups|gangs|actors)"
    r"|later (?:seen|observed|used)|together with|operating together|collaborat\w*"
    r"|partner(?:ed|ing|ship)? with)\b",
    re.IGNORECASE,
)
# a sentence about a lure or a disguise names software the actor imitates, not uses
LURE_RE = re.compile(
    r"\b(?:spoof\w*|impersonat\w*|masquerad\w*|disguised as|posing as|pretend\w*|lures?|decoys?"
    r"|fake)\b",
    re.IGNORECASE,
)
# a sentence about leaked or published tools ("exploits released by the Shadow Brokers") names
# whoever disclosed them, not who uses them
LEAK_RE = re.compile(
    r"\b(?:leak(?:s|ed|ing)?|made public|dumped by|dump of|publish(?:es|ed|ing)?"
    r"|releas(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_HEDGE_RANK = {"confirmed": 0, "stated": 1, "likely": 2, "possible": 3}
_POSSIBLE_RE = re.compile(
    r"\b(?:possibl[ey]|plausibl[ey]|suspected|potentially|perhaps|tentativ\w*|unconfirmed"
    r"|low[- ]confidence"
    r"|cannot (?:rule out|confirm)|not (?:yet )?(?:confirmed|clear)|unclear)\b",
    re.IGNORECASE,
)
_POSSIBLE_CASED_RE = re.compile(r"\b(?:may|might|could)\b")  # "May" is a month
_CONFIRMED_RE = re.compile(
    r"\b(?:high[- ]confidence|confirm(?:s|ed)|definitive(?:ly)?|conclusive(?:ly)?)\b",
    re.IGNORECASE,
)
_LIKELY_RE = re.compile(
    r"\b(?:likely|probabl[ey]|moderate[- ]confidence|we (?:believe|assess|suspect)|assess(?:ed|es)?"
    r"|appears? to|suggest(?:s|ed|ing)?|consistent with)\b",
    re.IGNORECASE,
)
# how a sentence relates two actors (stored as a claim; never merges actors on its own)
CLAIM_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "same_as",
        re.compile(
            r"\b(?:also (?:known|tracked|referred to) as|(?:known|tracked|referred to|reported)"
            r" as|track(?:s|ed)? as|a\.?k\.?a\.?|alias(?:ed)?(?: as)?|same (?:actor|group|"
            r"activity) as|corresponds? (?:to|with)|equivalent (?:to|of))\b",
            re.IGNORECASE,
        ),
    ),
    ("part_of", re.compile(r"\b(?:sub-?groups? of|part of|subset of|division of|unit of)\b", re.I)),
    (
        "overlaps",
        re.compile(
            r"\b(?:overlap\w*|linked to|associated with|related to|tied to|connected to|"
            r"affiliated with|ties to|links to)\b",
            re.IGNORECASE,
        ),
    ),
)


def hedge_of(text: str) -> str:
    """confirmed, likely, possible, or stated (no hedging word)."""
    if _POSSIBLE_RE.search(text) or _POSSIBLE_CASED_RE.search(text):
        return "possible"
    if _CONFIRMED_RE.search(text):
        return "confirmed"
    if _LIKELY_RE.search(text):
        return "likely"
    return "stated"


@dataclass(frozen=True)
class Software:
    id: str  # "S1234" (ATT&CK), "misp:<uuid>" or "name:<name>"
    name: str
    kind: str  # attack_malware, attack_tool, misp, report


# ------------------------------------------------------------------ words that are never names


def protected_words(kb: Any) -> frozenset[str]:
    """Words of ATT&CK's technique and data-component text and of its non-malware software
    names (PowerShell, OneDrive, LSASS, PsExec): technology, never a malware name."""
    texts = [
        text
        for technique in kb.techniques.values()
        for text in (technique.name, technique.description, technique.detection)
        if text
    ]
    texts += [
        text
        for component in (getattr(kb, "data_components", {}) or {}).values()
        for text in (getattr(component, "name", ""), getattr(component, "description", ""))
        if text
    ]
    words = {token.casefold() for text in texts for token in _TOKEN_RE.findall(text)}
    for software in kb.software.values():
        if software.software_type != "malware":
            for name in (software.name, *software.aliases):
                words.update(token.casefold() for token in _TOKEN_RE.findall(name or ""))
    return frozenset(words)


class NameRules:
    """Whether a token can be a software name: at least four characters, not a number, not an
    ordinary English word, a place, a technology word, a publisher, an actor name or a malware
    noun, and not a description ("PowerShell-based", "Base64-encoded")."""

    def __init__(
        self,
        ordinary: Iterable[str],
        protected: Iterable[str],
        actor_names: Iterable[str],
        places: Iterable[str] = (),
    ) -> None:
        self.ordinary = frozenset(word.casefold() for word in ordinary)
        self.protected = frozenset(protected)
        self.actors = frozenset(name.casefold() for name in actor_names)
        self.places = frozenset(places)

    def word_like(self, name: str) -> bool:
        return name.casefold() in self.ordinary or name in self.places

    def allowed(self, name: str) -> bool:
        key = name.casefold()
        parts = [part for part in re.split(r"[-_]", key) if part]
        tail = name.rsplit("-", 1)[1] if "-" in name else ""
        return (
            len(name) >= 4
            and not name.isdigit()
            and key not in self.ordinary
            and name not in self.places
            and key not in self.protected
            and key not in PUBLISHER_WORDS
            and key not in self.actors
            and not _MALWARE_NOUN_RE.fullmatch(name)
            and not (tail.isalpha() and tail.islower())
            and not all(part in self.ordinary or part.isdigit() for part in parts)
        )


def introduced_names(statements: Sequence[str], rules: NameRules) -> set[str]:
    """Names the post gives a malware noun ("the HOOKEDGE backdoor", "a loader dubbed X"), in
    their own spelling."""
    found: set[str] = set()
    for text in statements:
        for pattern in (_NAME_BEFORE_NOUN_RE, _NAME_AFTER_CUE_RE):
            for match in pattern.finditer(text):
                name = match.group("name")
                if rules.allowed(name):
                    found.add(name)
    return found


# ------------------------------------------------------------------ software matching


def _inner_capitals(alias: str) -> bool:
    """A word with a capital after its first letter (PS1, OSInfo, PlugX, MEDUSA)."""
    return any(any(char.isupper() for char in word[1:]) for word in alias.split())


class SoftwareMatcher:
    """Finds software names in a sentence: ATT&CK first, then MISP, then names posts introduced;
    the longest name wins an overlap."""

    def __init__(
        self,
        kb: Any,
        rules: NameRules,
        *,
        misp_dir: Path | None = None,
        introduced: Iterable[str] = (),
    ) -> None:
        from apt_classifier.evidence.iocs import (
            _GENERIC_ALIASES,
            _aliases,
            _external_id,
            _software_records,
            ambiguous_software_aliases,
        )

        self.rules = rules
        ambiguous = ambiguous_software_aliases(kb)
        group_names = {
            alias.casefold()
            for group in (getattr(kb, "groups", {}) or {}).values()
            for alias in (group.name, *group.aliases)
            if alias
        }
        by_alias: dict[tuple[str, bool], set[Software]] = defaultdict(set)
        attack_keys: set[str] = set()
        # folded malware aliases with inner capitals; exact aliases that need ".exe" or a noun
        self._no_lowercase: set[str] = set()
        self._exe_only: set[str] = set()
        self._noun_only: set[str] = set()  # casefolded
        self._command_only: dict[str, str] = {}  # casefolded -> the record's own spelling
        for record in _software_records(kb):
            software_id = _external_id(record)
            if not software_id:
                continue
            kind = str(getattr(record, "software_type", "") or "").casefold()
            item = Software(
                software_id,
                str(record.name),
                "attack_malware" if kind == "malware" else "attack_tool",
            )
            for alias in _aliases(record):
                if len(alias) < 3 or alias.casefold() in _GENERIC_ALIASES or alias.isdecimal():
                    continue
                word = rules.word_like(alias)
                exact = alias in ambiguous or word
                if word and alias == alias.lower():
                    self._exe_only.add(alias)
                elif alias in rules.places or alias.casefold() in group_names:
                    self._noun_only.add(alias.casefold())
                elif item.kind == "attack_tool" and len(alias) <= 3:
                    self._command_only[alias.casefold()] = alias
                elif not exact and item.kind == "attack_malware" and _inner_capitals(alias):
                    self._no_lowercase.add(alias.casefold())
                by_alias[(alias if exact else alias.casefold(), exact)].add(item)
                attack_keys.add(alias.casefold())
        # an alias two ATT&CK entries share names neither
        self._attack = {
            key: next(iter(items)) for key, items in by_alias.items() if len(items) == 1
        }
        self._attack_exact = self._alternation([k for (k, exact) in self._attack if exact], False)
        self._attack_folded = self._alternation(
            [k for (k, exact) in self._attack if not exact], True
        )
        self._misp_single: dict[str, Software] = {}
        misp_multi: dict[str, Software] = {}
        for path in (
            sorted((misp_dir / f"{name}.json") for name in MISP_SOFTWARE_FILES) if misp_dir else ()
        ):
            if not path.is_file():
                continue
            cluster = json.loads(path.read_text(encoding="utf-8"))
            for value in cluster.get("values", []):
                uuid = value.get("uuid")
                if not uuid:
                    continue
                item = Software(f"misp:{uuid}", str(value.get("value", "")).strip(), "misp")
                for label in (value.get("value", ""), *value.get("meta", {}).get("synonyms", [])):
                    label = str(label).strip()
                    tokens = _TOKEN_RE.findall(label)
                    key = label.casefold()
                    if not tokens or label != " ".join(tokens) or key in attack_keys:
                        continue
                    if len(tokens) == 1 and rules.allowed(label):
                        self._misp_single.setdefault(key, item)
                    elif len(tokens) > 1 and any(rules.allowed(token) for token in tokens):
                        misp_multi.setdefault(key, item)
        self._misp_multi = misp_multi
        self._misp_multi_re = self._alternation(list(misp_multi), True)
        self._introduced: dict[str, Software] = {}
        for name in introduced:
            key = name.casefold()
            if key in attack_keys or key in self._misp_single or not rules.allowed(name):
                continue
            self._introduced.setdefault(name, Software(f"name:{key}", name, "report"))

    @staticmethod
    def _alternation(names: list[str], fold: bool) -> re.Pattern[str] | None:
        if not names:
            return None
        ordered = sorted(set(names), key=lambda item: (-len(item), item))
        return re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(name) for name in ordered) + r")(?!\w)",
            re.IGNORECASE if fold else 0,
        )

    def _attack_context(self, spelled: str, text: str, end: int) -> bool:
        """The rules a matched ATT&CK alias must pass in its context (see the module notes)."""
        key = spelled.casefold()
        if key in self._noun_only and not _NOUN_AFTER_RE.match(text, end):
            return False
        own = self._command_only.get(key)
        if own is not None:
            shouted = spelled.isupper() and not own.isupper()
            return bool(_COMMAND_AFTER_RE.match(text, end)) and not (
                shouted and not _EXE_AFTER_RE.match(text, end)
            )
        return True

    @property
    def introduced(self) -> dict[str, Software]:
        return dict(self._introduced)

    def find(self, text: str) -> list[tuple[int, int, Software]]:
        from apt_classifier.evidence.iocs import _sentence_initial, _title_case

        found: list[tuple[int, int, Software]] = []
        if self._attack_folded is not None:
            for match in self._attack_folded.finditer(text):
                key = match.group(0).casefold()
                if key in self._no_lowercase and match.group(0) == key:
                    continue
                item = self._attack.get((key, False))
                if item and self._attack_context(match.group(0), text, match.end()):
                    found.append((match.start(), match.end(), item))
        if self._attack_exact is not None:
            for match in self._attack_exact.finditer(text):
                alias = match.group(0)
                if _title_case(alias) and _sentence_initial(text, match.start()):
                    continue
                if alias in self._exe_only and not _EXE_AFTER_RE.match(text, match.end()):
                    continue
                item = self._attack.get((alias, True))
                if item and self._attack_context(alias, text, match.end()):
                    found.append((match.start(), match.end(), item))
        if self._misp_multi_re is not None:
            for match in self._misp_multi_re.finditer(text):
                item = self._misp_multi.get(match.group(0).casefold())
                if item:
                    found.append((match.start(), match.end(), item))
        for match in _NAME_TOKEN_RE.finditer(text):
            word = match.group(0)
            item = self._introduced.get(word)
            if item is None and "-" not in word and "_" not in word:
                item = self._misp_single.get(word.casefold())
            if item is not None:
                found.append((match.start(), match.end(), item))
        found = [
            hit
            for hit in found
            if (hit[1] - hit[0] > 4 or not _FILE_AFTER_RE.match(text, hit[1]))
            and not _PRODUCT_AFTER_RE.match(text, hit[1])
        ]
        # longest first, then earlier sources (ATT&CK before MISP before introduced names)
        rank = {"attack_malware": 0, "attack_tool": 0, "misp": 1, "report": 2}
        found.sort(key=lambda hit: (-(hit[1] - hit[0]), rank[hit[2].kind], hit[0]))
        kept: list[tuple[int, int, Software]] = []
        for start, end, item in found:
            if any(start < other_end and other_start < end for other_start, other_end, _ in kept):
                continue
            kept.append((start, end, item))
        return sorted(kept, key=lambda hit: hit[0])


# ------------------------------------------------------------------ actors


@dataclass
class ActorNames:
    """The fresh-set builder's actor dictionary and name pattern, plus its cluster-code pattern."""

    surface: Mapping[str, str]
    pattern: re.Pattern[str] | None
    cluster_codes: re.Pattern[str]
    software_pattern: re.Pattern[str] | None = None

    def spans(self, text: str) -> list[tuple[int, int, str]]:
        """Actor mentions with their spans (the rules of ``actor_mentions``: dictionary names,
        case-sensitive whole words, a name inside an ATT&CK software name is the software;
        cluster codes that are no dictionary name count as ``code:<code>``). Stricter than the
        label rule: a name before a malware noun or in a detection name ("Karma ransomware",
        "Silence.Downloader") is the software, and an ATT&CK tactic id (TA0010) is no code."""
        hits: list[tuple[int, int, str]] = []
        software = (
            [match.span() for match in self.software_pattern.finditer(text)]
            if self.software_pattern is not None
            else []
        )
        if self.pattern is not None:
            for match in self.pattern.finditer(text):
                inside = any(
                    start <= match.start() and match.end() <= end
                    for start, end in software
                    if (start, end) != match.span()
                )
                if not inside and not _NOT_ACTOR_AFTER_RE.match(text, match.end()):
                    hits.append((match.start(), match.end(), self.surface[match.group(0)]))
        for match in self.cluster_codes.finditer(text):
            if _TACTIC_ID_RE.fullmatch(match.group(0)):
                continue
            if any(start <= match.start() < end for start, end, _ in hits):
                continue
            hits.append((match.start(), match.end(), f"code:{match.group(0)}"))
        return sorted(hits)


# ------------------------------------------------------------------ reading a post


@dataclass
class Link:
    actor: str
    software_id: str
    software_name: str
    software_kind: str
    rule: str  # sentence, report
    hedge: str
    statement: int
    status: str  # active, unclear
    reason: str = ""


@dataclass
class Claim:
    actor_a: str
    actor_b: str
    relation: str
    hedge: str
    statement: int


@dataclass
class Reading:
    label: str
    actor: str | None
    reason: str | None
    mentions: dict[str, int]
    links: list[Link] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    # names posts introduce that this post mentions: software id -> 1 when it introduces it here
    names: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _distinct(actors: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(actors))


def read_post(
    title: str,
    statements: Sequence[str],
    *,
    actors: ActorNames,
    software: SoftwareMatcher,
    label_fn: Callable[[str, list[str]], dict[str, Any]],
) -> Reading:
    """One post: its label (known, open or excluded), its uses links and its actor claims."""
    labelled = label_fn(title, list(statements))
    label = labelled["label"]
    lead = labelled.get("group_id") or labelled.get("actor")
    reading = Reading(label, lead, labelled.get("reason"), dict(labelled.get("mentions") or {}))
    per_software: dict[str, list[int]] = defaultdict(list)
    # sentences that say the software is used, with no other actor and no comparison or lure
    use_sentences: dict[str, list[tuple[int, str]]] = defaultdict(list)
    names: dict[str, Software] = {}
    troubled: set[str] = set()
    known_names = software.introduced
    here = {
        known_names[name].id
        for name in introduced_names(statements, software.rules)
        if name in known_names
    }
    for index, text in enumerate(statements):
        actor_hits = actors.spans(text)
        software_hits = [
            hit
            for hit in software.find(text)
            if not any(hit[0] < end and start < hit[1] for start, end, _ in actor_hits)
        ]
        present = _distinct(actor for _, _, actor in actor_hits)
        compare = "comparison" if COMPARISON_RE.search(text) else ""
        compare = compare or ("lure" if LURE_RE.search(text) else "")
        compare = compare or ("leak" if LEAK_RE.search(text) else "")
        hedge = hedge_of(text)
        uses = bool(USE_RE.search(text))
        for _, _, item in software_hits:
            per_software[item.id].append(index)
            names[item.id] = item
            if item.kind == "report":
                reading.names[item.id] = int(item.id in here)
            if compare or any(actor != lead for actor in present):
                troubled.add(item.id)
            elif uses:
                use_sentences[item.id].append((index, hedge))
        if software_hits and present and uses:
            unclear = compare or ("several_actors" if len(present) > 1 else "")
            for actor in present:
                for item in {hit[2].id: hit[2] for hit in software_hits}.values():
                    reading.links.append(
                        Link(
                            actor,
                            item.id,
                            item.name,
                            item.kind,
                            "sentence",
                            hedge,
                            index,
                            "unclear" if unclear else "active",
                            unclear,
                        )
                    )
        if len(present) >= 2:
            for relation, pattern in CLAIM_RES:
                if pattern.search(text):
                    for a, b in zip(present, present[1:], strict=False):
                        reading.claims.append(Claim(a, b, relation, hedge, index))
                    break
    if label in ("known", "open") and lead:
        for software_id, indexes in per_software.items():
            if len(indexes) < REPORT_LINK_MIN_MENTIONS:
                continue
            item = names[software_id]
            if software_id in troubled:
                status, reason = "unclear", "other_actor_or_comparison"
                index, hedge = indexes[0], "stated"
            elif use_sentences[software_id]:
                # the most certain of the sentences that say it is used
                index, hedge = min(use_sentences[software_id], key=lambda use: _HEDGE_RANK[use[1]])
                status, reason = "active", ""
            else:
                continue  # named twice, never said to be used
            reading.links.append(
                Link(
                    lead, software_id, item.name, item.kind, "report", hedge, index, status, reason
                )
            )
    return reading


def summarise(readings: Iterable[Reading]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    links: Counter[str] = Counter()
    for reading in readings:
        labels[reading.label] += 1
        for link in reading.links:
            links[f"{link.rule}:{link.status}"] += 1
    return {"labels": dict(labels), "links": dict(links)}
