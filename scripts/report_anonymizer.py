"""Label-blind anonymization of real threat-report statements for the attribution benchmarks.

Real reports name their own actor, its vendor cluster names and its sponsor state, and the
name-sensitive signals can match those words without reading any tradecraft: group-description
similarity embeds the report's opening (where vendors put attribution) against the ATT&CK group
descriptions, and actor-behaviour retrieval shares name tokens with ATT&CK procedure sentences.
Three cumulative levels:

- ``names``: every ATT&CK group name and alias -> "the adversary". This is the held-out folds'
  replacement and the CTI-HAL headline; ``scripts/attribution_benchmark.py`` keeps it
  byte-identical (``evidence_name_pattern`` moved here unchanged).
- ``strict``: names, plus the report's own attribution prose (a sentence that refers to an
  actor *and* argues attribution, a paragraph with two such sentences, a section about
  attribution), vendor cluster ids and actor names that ATT&CK does not list, ATT&CK object ids,
  and a country / nationality / sponsor / region / language gazetteer. Ordinary-word aliases
  ("Play", "Chromium", "Tick") are replaced only in their ATT&CK spelling. Victim geography goes
  too: strict is a no-victimology regime.
- ``tradecraft``: strict, plus every ATT&CK *malware* name -> "the malware" (tools stay, they are
  technique evidence), so only behaviour is left.

Every rule reads the text, never the label, so labelled and open-set reports are cleaned alike.
Malware names stay at strict on purpose (an analyst reads them); the per-report records say
which ATT&CK software survived so a lookup is never mistaken for tradecraft.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from apt_classifier.attack.parse import strip_attack_citations
from apt_classifier.evidence.iocs import match_software_aliases

LEVELS = ("names", "strict", "tradecraft")
REPLACEMENT = "the adversary"
MALWARE_REPLACEMENT = "the malware"
_DELETED = "\x00"  # marks a deletion point until the article and spacing repair


def actor_names(kb: Any) -> tuple[str, ...]:
    """Every ATT&CK group name and alias, longest first, as the replacement pattern needs."""
    return tuple(
        sorted(
            {
                name.strip()
                for group in kb.groups.values()
                for name in (group.name, *group.aliases)
                if name and name.strip()
            },
            key=lambda item: (-len(item), item.casefold()),
        )
    )


def evidence_name_pattern(names: tuple[str, ...]) -> re.Pattern[str] | None:
    """The actor-name pattern for *evidence* text (fold statements, real-report statements).

    Like ``vectorstore._actor_name_pattern``, but a name that ends a sentence ("the backdoor known
    as Carbanak.") is replaced too; a dot followed by a word character still protects file and
    family names (Patchwork.dll, Silence.Downloader). The indexed procedure sentences keep the
    index's pattern: changing it there would need a re-index and a combiner refit.
    """
    ordered = sorted(
        (name for name in names if name), key=lambda item: (-len(item), item.casefold())
    )
    if not ordered:
        return None
    return re.compile(
        r"(?<![\w.-])(?:" + "|".join(re.escape(name) for name in ordered) + r")"
        r"(?P<possessive>['’]s)?(?![\w-]|\.\w)",
        re.IGNORECASE,
    )


# ATT&CK group aliases that are ordinary English words (every single-token alias found in
# /usr/share/dict/words, generated once from ATT&CK 19.2). At strict they are replaced only in
# their ATT&CK spelling or its upper case, so "the Chromium extension API" or "play" stay text.
GUARDED_ALIASES = frozenset(
    {
        "ACTINIUM",
        "Armageddon",
        "Axiom",
        "BARIUM",
        "BISMUTH",
        "BITTER",
        "BROMINE",
        "Buckeye",
        "CHIMBORAZO",
        "CHROMIUM",
        "Chimera",
        "Cicada",
        "Cleaver",
        "Confucius",
        "Dragonfly",
        "ELBRUS",
        "Elfin",
        "Equation",
        "GALLIUM",
        "HAFNIUM",
        "IRIDIUM",
        "Inception",
        "Karma",
        "Krypton",
        "Leviathan",
        "Lyceum",
        "MANGANESE",
        "MERCURY",
        "MONSOON",
        "Machete",
        "Mantis",
        "Mirage",
        "NEODYMIUM",
        "NICKEL",
        "Newscaster",
        "PLATINUM",
        "PLUTONIUM",
        "POLONIUM",
        "POTASSIUM",
        "Patchwork",
        "Phosphorus",
        "Play",
        "RADIUM",
        "Rancor",
        "Rattlesnake",
        "Reaper",
        "SILICON",
        "STRONTIUM",
        "Silence",
        "Snake",
        "Swallowtail",
        "THALLIUM",
        "Tick",
        "WebMasters",
        "YTTRIUM",
        "ZINC",
        "ZIRCONIUM",
    }  # fmt: skip
)
# Head words of "<X> Group/Team" aliases that reports use bare. Not derived automatically:
# "Shell" (Shell Crew) or "Cobalt" (Cobalt Group) would wipe ordinary text and Cobalt Strike.
SHORT_FORMS = ("Gamaredon", "Lazarus", "Sandworm", "Winnti", "Gwisin")
# Actor and cluster names that ATT&CK 19.2 does not list as aliases (vendor naming for the
# actors in the Elastic and CTI-HAL reports, and common DPRK/PRC cluster names). Applied to every
# report alike.
VENDOR_ACTOR_NAMES = (
    "DangerousPassword", "TraderTraitor", "CryptoCore", "SnatchCrypto", "DeceptiveDevelopment",
    "Tenacious Pungsan", "PurpleBravo", "ChamelGang", "Red Menshen", "Dragon Breath",
    "DragonBreath", "RudePanda", "Sordeal Group", "Gwisin Gang", "Famous Chollima",
)  # fmt: skip
_DASH = "-‐‑–"
_CLUSTER_RE = re.compile(
    r"(?<![\w./\\-])(?P<name>"
    r"REF[ -]?\d{4}|UNC\d{3,5}|(?:Storm|DEV)-\d{4}|TA(?!0)\d{3,4}|UAT-?\d{3,5}|UAC-\d{4}"
    r"|TAG-\d{2,3}|APT-[A-Z]{1,2}-\d{1,3}|APT[ -]?\d{1,3}|CL-(?:STA|CRI|UNK)-\d{4}|Hive\d{4}"
    r"|TEMP\.[A-Z][a-z]+|DEV#[A-Z]+"
    r"|[A-Z][a-z]+ (?:Typhoon|Sleet|Blizzard|Sandstorm|Cyclone|Tempest|Hail|Panda|Bear|Kitten"
    r"|Chollima|Spider|Pisces|Taurus|Ursa|Serpens|Libra|Scorpius)"
    r"|(?:Earth|Water|Void) [A-Z][a-z]+"
    r"|"
    + "|".join(re.escape(name) for name in sorted(VENDOR_ACTOR_NAMES, key=len, reverse=True))
    + r")(?P<possessive>['’]s)?(?!\w|\.\w)"
)
_SHORT_FORM_RE = re.compile(
    r"(?<![\w.-])(?:" + "|".join(SHORT_FORMS) + r")(?P<possessive>['’]s)?(?![\w-]|\.\w)",
    re.IGNORECASE,
)
_ATTACK_ID_RE = re.compile(r"(?<![\w/.-])(?:[GSC]\d{4}|TA00\d\d)(?![\w-])")
_NON_LATIN_RE = re.compile(r"[Ѐ-ӿ֐-ۿᄀ-ᇿ぀-ヿ㐀-鿿가-힯]+")

# --- gazetteer --------------------------------------------------------------------------
# Case-sensitive: title-case entries match only as written and acronyms only in capitals, so
# Elastic's ALL-CAPS codenames (CUBA, KREMLIN, CHINA...) stay and are counted as kept. Left
# out on purpose: English ("plain English"), bare Latin (Latin-1), bare Gulf, PLA (the Windows
# "pla" folder), RGB (pixel values).
_COUNTRIES: dict[str, tuple[str, ...]] = {
    "Afghanistan": ("Afghan",), "Albania": ("Albanian",), "Algeria": ("Algerian",),
    "Andorra": (), "Angola": ("Angolan",), "Argentina": ("Argentine", "Argentinian"),
    "Armenia": ("Armenian",), "Australia": ("Australian",), "Austria": ("Austrian",),
    "Azerbaijan": ("Azerbaijani", "Azeri"), "Bahamas": (), "Bahrain": ("Bahraini",),
    "Bangladesh": ("Bangladeshi",), "Barbados": (), "Belarus": ("Belarusian",),
    "Belgium": ("Belgian",), "Belize": (), "Benin": (), "Bhutan": (), "Bolivia": ("Bolivian",),
    "Bosnia": ("Bosnian",), "Botswana": (), "Brazil": ("Brazilian",), "Brunei": (),
    "Bulgaria": ("Bulgarian",), "Burkina Faso": (), "Burundi": (), "Cambodia": ("Cambodian",),
    "Cameroon": ("Cameroonian",), "Canada": ("Canadian",), "Chad": ("Chadian",),
    "Chile": ("Chilean",), "China": ("Chinese",), "Colombia": ("Colombian",), "Congo": ("Congolese",),
    "Costa Rica": ("Costa Rican",), "Croatia": ("Croatian",), "Cuba": ("Cuban",),
    "Cyprus": ("Cypriot",), "Czech Republic": (), "Czechia": ("Czech",), "Denmark": ("Danish",),
    "Djibouti": (), "Dominican Republic": ("Dominican",), "Ecuador": ("Ecuadorian",),
    "Egypt": ("Egyptian",), "El Salvador": ("Salvadoran",), "Eritrea": ("Eritrean",),
    "Estonia": ("Estonian",), "Ethiopia": ("Ethiopian",), "Fiji": (), "Finland": ("Finnish",),
    "France": ("French",), "Gabon": (), "Gambia": (), "Georgia": ("Georgian",),
    "Germany": ("German",), "Ghana": ("Ghanaian",), "Greece": ("Greek",), "Guatemala": (),
    "Guinea": (), "Guyana": (), "Haiti": ("Haitian",), "Honduras": (), "Hungary": ("Hungarian",),
    "Iceland": ("Icelandic",), "India": ("Indian",), "Indonesia": ("Indonesian",),
    "Iran": ("Iranian",), "Iraq": ("Iraqi",), "Ireland": ("Irish",), "Israel": ("Israeli",),
    "Italy": ("Italian",), "Ivory Coast": (), "Jamaica": ("Jamaican",), "Japan": ("Japanese",),
    "Jordan": ("Jordanian",), "Kazakhstan": ("Kazakh",), "Kenya": ("Kenyan",), "Kosovo": (),
    "Kuwait": ("Kuwaiti",), "Kyrgyzstan": ("Kyrgyz",), "Laos": ("Laotian",), "Latvia": ("Latvian",),
    "Lebanon": ("Lebanese",), "Liberia": ("Liberian",), "Libya": ("Libyan",),
    "Liechtenstein": (), "Lithuania": ("Lithuanian",), "Luxembourg": (), "Macau": (),
    "Madagascar": (), "Malawi": (), "Malaysia": ("Malaysian",), "Maldives": (), "Mali": ("Malian",),
    "Malta": ("Maltese",), "Mauritania": (), "Mauritius": (), "Mexico": ("Mexican",),
    "Moldova": ("Moldovan",), "Monaco": (), "Mongolia": ("Mongolian",), "Montenegro": (),
    "Morocco": ("Moroccan",), "Mozambique": (), "Myanmar": ("Burmese",), "Burma": (),
    "Namibia": (), "Nepal": ("Nepali", "Nepalese"), "Netherlands": ("Dutch",),
    "Nicaragua": (), "Niger": (), "Nigeria": ("Nigerian",), "North Macedonia": ("Macedonian",),
    "Norway": ("Norwegian",), "Oman": ("Omani",), "Pakistan": ("Pakistani",),
    "Palestine": ("Palestinian",), "Panama": ("Panamanian",), "Paraguay": (), "Peru": ("Peruvian",),
    "Philippines": ("Filipino", "Philippine"), "Poland": ("Polish",), "Portugal": ("Portuguese",),
    "Qatar": ("Qatari",), "Romania": ("Romanian",), "Russia": ("Russian",), "Rwanda": (),
    "Senegal": (), "Serbia": ("Serbian",), "Sierra Leone": (), "Singapore": ("Singaporean",),
    "Slovakia": ("Slovak",), "Slovenia": ("Slovenian",), "Somalia": ("Somali",),
    "South Africa": ("South African",), "South Sudan": (), "Spain": ("Spanish",),
    "Sudan": ("Sudanese",), "Suriname": (), "Sweden": ("Swedish",), "Switzerland": ("Swiss",),
    "Syria": ("Syrian",), "Taiwan": ("Taiwanese",), "Tajikistan": ("Tajik",),
    "Tanzania": ("Tanzanian",), "Thailand": ("Thai",), "Togo": (), "Tunisia": ("Tunisian",),
    "Turkey": ("Turkish",), "Türkiye": (), "Turkmenistan": ("Turkmen",), "Uganda": ("Ugandan",),
    "Ukraine": ("Ukrainian",), "Uruguay": ("Uruguayan",), "Uzbekistan": ("Uzbek",),
    "Venezuela": ("Venezuelan",), "Vietnam": ("Vietnamese",), "Viet Nam": (),
    "Yemen": ("Yemeni",), "Zambia": (), "Zimbabwe": (), "Korea": ("Korean",),
    "North Korea": ("North Korean",), "South Korea": ("South Korean",), "Hong Kong": (),
    "Saudi Arabia": ("Saudi",), "Sri Lanka": ("Sri Lankan",), "New Zealand": (),
    "Papua New Guinea": (), "United States": ("American",), "United Kingdom": ("British",),
    "Britain": (), "United Arab Emirates": ("Emirati",), "Soviet Union": ("Soviet",),
    "Tibet": ("Tibetan",), "Xinjiang": (), "Crimea": ("Crimean",), "Donbas": (),
    "Chechnya": ("Chechen",), "Kurdistan": ("Kurdish",),
}  # fmt: skip
_SPONSOR_STATES = (
    "Democratic People's Republic of North Korea", "Democratic People’s Republic of North Korea",
    "Democratic People's Republic of Korea", "Democratic People’s Republic of Korea",
    "People's Republic of China", "People’s Republic of China", "Republic of Korea",
    "Russian Federation", "Islamic Republic of Iran", "United States of America",
)  # fmt: skip
_REGIONS: dict[str, tuple[str, ...]] = {
    "Southeast Asia": ("Southeast Asian",), "South East Asia": ("South East Asian",),
    "East Asia": ("East Asian",), "Central Asia": ("Central Asian",), "South Asia": ("South Asian",),
    "Asia-Pacific": (), "Asia Pacific": (), "Indo-Pacific": (), "Asia": ("Asian",),
    "Middle East": ("Middle Eastern",), "Latin America": ("Latin American",),
    "South America": ("South American",), "North America": ("North American",),
    "Central America": ("Central American",), "Eastern Europe": ("Eastern European",),
    "Western Europe": ("Western European",), "Europe": ("European",), "Africa": ("African",),
    "Persian Gulf": (), "Caucasus": (), "Balkans": ("Balkan",), "Baltics": ("Baltic",),
    "Scandinavia": ("Scandinavian",), "Nordics": ("Nordic",), "Caribbean": (), "Oceania": (),
    "Levant": (), "Commonwealth of Independent States": (), "Five Eyes": (),
}  # fmt: skip
_ACRONYMS = {
    "DPRK": "country", "PRC": "country", "ROK": "country", "USSR": "country", "US": "country",
    "USA": "country", "UK": "country", "UAE": "country", "EU": "region", "CIS": "region",
    "ASEAN": "region", "APAC": "region", "EMEA": "region", "LATAM": "region", "NATO": "agency",
    "IRGC": "agency", "MOIS": "agency", "GRU": "agency", "FSB": "agency", "SVR": "agency",
    "MSS": "agency", "KGB": "agency", "GCHQ": "agency",
}  # fmt: skip
_AGENCIES = (
    "Reconnaissance General Bureau", "Ministry of State Security", "Federal Security Service",
    "Main Intelligence Directorate", "Islamic Revolutionary Guard Corps",
    "Ministry of Intelligence", "People's Liberation Army", "People’s Liberation Army", "Kremlin",
)  # fmt: skip
_CAPITALS = (
    "Pyongyang", "Beijing", "Moscow", "Tehran", "Hanoi", "Minsk", "Kyiv", "Kiev", "Seoul",
    "Taipei", "Islamabad", "New Delhi", "Tel Aviv", "Ankara", "Riyadh", "Kuala Lumpur", "Jakarta",
    "Manila", "Bangkok", "Shanghai", "Shenzhen", "Guangzhou", "Saint Petersburg", "St. Petersburg",
)  # fmt: skip
_LANGUAGES = (
    "Cyrillic", "Hangul", "Mandarin", "Cantonese", "Farsi", "Persian", "Arabic", "Hebrew", "Hindi",
    "Urdu", "Pashto", "Uyghur", "Uighur",
)  # fmt: skip
# compound suffixes that make the stem an origin claim ("DPRK-aligned", "Chinese-speaking")
_ORIGIN_SUFFIXES = frozenset(
    [
        "based",
        "nexus",
        "aligned",
        "linked",
        "affiliated",
        "backed",
        "sponsored",
        "speaking",
        "language",
        "origin",
        "attributed",
        "operated",
        "related",
        "led",
        "run",
        "controlled",
        "directed",
    ]
)
_ORIGIN_ROLE_SUFFIXES = frozenset(
    [
        "nexus",
        "aligned",
        "linked",
        "affiliated",
        "backed",
        "sponsored",
        "attributed",
        "operated",
        "controlled",
        "directed",
        "origin",
        "speaking",
    ]
)
_ORIGIN_CONTEXT_RE = re.compile(
    r"attribut|nexus|state|sponsor|government|speaker|origin|actor|group|threat|regime",
    re.IGNORECASE,
)


def _gazetteer_kinds() -> dict[str, str]:
    kinds: dict[str, str] = {}
    for country, demonyms in _COUNTRIES.items():
        kinds[country] = "country"
        for demonym in demonyms:
            kinds[demonym] = "demonym"
    for state in _SPONSOR_STATES:
        kinds[state] = "country"
    for region, adjectives in _REGIONS.items():
        kinds[region] = "region"
        for adjective in adjectives:
            kinds[adjective] = "region_adjective"
    for agency in _AGENCIES:
        kinds[agency] = "agency"
    for capital in _CAPITALS:
        kinds[capital] = "capital"
    for language in _LANGUAGES:
        kinds.setdefault(language, "language")
    for acronym, kind in _ACRONYMS.items():
        kinds[acronym] = "acronym_" + kind
    return kinds


GAZETTEER_KINDS = _gazetteer_kinds()
_TERM_ALTERNATION = "|".join(
    re.escape(term) for term in sorted(GAZETTEER_KINDS, key=lambda item: (-len(item), item))
)
# misspelling-tolerant stems (the Gamaredon report writes "Ukranian" and "Ukraninan")
_STEMS = r"Ukr[a-z]+|Belarus[a-z]*|Unit \d{5}"
_GAZETTEER_RE = re.compile(
    r"(?:(?<![\w.\[\]/\\@#$])|(?<=\bpro-)|(?<=\banti-)|(?<=\bnon-))"
    r"(?P<term>" + _TERM_ALTERNATION + "|" + _STEMS + r")"
    r"(?:(?P<origin>[" + _DASH + r"](?P<suffix>[A-Za-z]+))(?![\w-]))?"
    r"(?P<possessive>['’]s)?(?!\w|\.\w)"
)
_SPONSOR_RE = re.compile(
    r"(?<![\w-])(?:(?:state|government|nation|regime)[" + _DASH + r"](?:sponsored|backed|"
    r"affiliated|aligned|linked|nexus|controlled|directed|level)|nation[" + _DASH + r" ]states?)"
    r"(?![\w-])",
    re.IGNORECASE,
)
# attribution cues; a sentence is attribution prose only when it also refers to an actor
_CUE_RE = re.compile(
    r"\battribut(?:ed|ion|ing|able)\b|\bwe attribute\b|associated with|\bbelieved\b"
    r"|behind (?:this|the)\b|\blink(?:ed|ing|s)? (?:to|this|it|with)\b|\boverlap"
    # "align" as a verb, not inside an origin compound ("DPRK-aligned" is the gazetteer's)
    r"|(?<![" + _DASH + r"])\balign"
    r"|tied to|known (?:publicly )?as|\baka\b|tracked as|consistent with|correlat"
    r"|previous(?:ly)? report|prior reporting|\breporting\b|operat(?:ed|ing) by|behest|sub-unit"
    r"|umbrella|exclusively|only group|\bbelong|notorious|infamous|well-known|in origin"
    r"|suspected|\bassess|confidence|connect(?:ed|ing|ion)|active since|since at least"
    r"|conducted by|observed previously",
    re.IGNORECASE,
)
_HEADING_CUE_RE = re.compile(
    r"attribut|connection|\blinks?\b|associat|intersection|overlap|assessment", re.IGNORECASE
)
_ARTICLE_BEFORE_ADVERSARY_RE = re.compile(r"\b(?:an?|the)\s+(the adversary)", re.IGNORECASE)
_ARTICLE_BEFORE_MALWARE_RE = re.compile(r"\b(?:an?|the)\s+(the malware)", re.IGNORECASE)
_ARTICLE_BEFORE_UNDISCLOSED_RE = re.compile(r"\b(the|an?)\s+an undisclosed\b", re.IGNORECASE)
_A_BEFORE_UNDISCLOSED_RE = re.compile(r"\b([Aa]) undisclosed\b")
# words after a demonym that make it a noun use ("Russian in origin"), not an adjective
_FUNCTION_WORDS = frozenset(
    [
        "in",
        "of",
        "and",
        "or",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "at",
        "on",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "that",
        "which",
    ]
)
_ARTICLE_AT_DELETION_RE = re.compile(r"\b(an?)\s*" + _DELETED + r"[\s" + _DELETED + r"]*(?=(\w))")


def _article_for(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _repair(text: str) -> str:
    """Article agreement at deletion points, article collapse, spacing and punctuation."""
    text = _ARTICLE_AT_DELETION_RE.sub(
        lambda match: (
            _article_for(match.group(2)).capitalize()
            if match.group(1)[0].isupper()
            else _article_for(match.group(2))
        )
        + " ",
        text,
    )
    text = text.replace(_DELETED, " ")
    text = _ARTICLE_BEFORE_ADVERSARY_RE.sub(r"\1", text)
    text = _ARTICLE_BEFORE_MALWARE_RE.sub(r"\1", text)
    text = _ARTICLE_BEFORE_UNDISCLOSED_RE.sub(
        lambda match: ("the" if match.group(1).lower() == "the" else "an") + " undisclosed"
        if match.group(1)[0].islower()
        else ("The" if match.group(1).lower() == "the" else "An") + " undisclosed",
        text,
    )
    text = _A_BEFORE_UNDISCLOSED_RE.sub(lambda match: match.group(1) + "n undisclosed", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return text


def _joined(texts: Sequence[str]) -> tuple[str, list[int]]:
    """The statements joined as the engine's classifier context joins them, with offsets."""
    offsets: list[int] = []
    position = 0
    for text in texts:
        offsets.append(position)
        position += len(text) + 1
    return " ".join(texts), offsets


def _spans_per_statement(
    texts: Sequence[str], spans: list[tuple[int, int, Any]]
) -> list[list[tuple[int, int, Any]]]:
    _, offsets = _joined(texts)
    output: list[list[tuple[int, int, Any]]] = [[] for _ in texts]
    for start, end, payload in spans:
        for index, offset in enumerate(offsets):
            limit = offset + len(texts[index])
            if offset <= start and end <= limit:
                output[index].append((start - offset, end - offset, payload))
                break
    return output


def _merge_spans(spans: list[tuple[int, int, Any]]) -> list[tuple[int, int, Any]]:
    merged: list[tuple[int, int, Any]] = []
    for start, end, payload in sorted(spans, key=lambda item: (item[0], -item[1])):
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end, merged[-1][2])
            continue
        merged.append((start, end, payload))
    return merged


class ReportAnonymizer:
    """The strict and tradecraft passes, built once per knowledge base.

    ``names`` stays in ``attribution_benchmark.anonymize_report_folds`` so the CTI-HAL headline
    is byte-identical; this class adds the guard for ordinary-word aliases and everything after.
    """

    def __init__(self, kb: Any, level: str = "strict") -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown anonymization level {level!r}")
        self.kb = kb
        self.level = level
        self.names = actor_names(kb)
        self.name_pattern = evidence_name_pattern(self.names)
        self._spellings: dict[str, set[str]] = {}
        for name in self.names:
            self._spellings.setdefault(name.casefold(), set()).add(name)
        self._guarded = {alias.casefold() for alias in GUARDED_ALIASES}

    # -- names, guarded ----------------------------------------------------------------
    def keeps_name_match(self, match: re.Match[str]) -> bool:
        """True when a names-pattern match is a group name, False for an ordinary word use."""
        word = match.group(0).removesuffix(match.group("possessive") or "")
        key = word.casefold()
        if key not in self._guarded:
            return True
        spellings = self._spellings.get(key, set())
        return word in spellings or any(word == spelling.upper() for spelling in spellings)

    def replace_names(self, text: str) -> str:
        """``vectorstore._sanitize_procedure`` with the ordinary-word guard."""
        cleaned = strip_attack_citations(text).strip()
        if self.name_pattern is not None:
            cleaned = self.name_pattern.sub(
                lambda match: (REPLACEMENT + (match.group("possessive") or ""))
                if self.keeps_name_match(match)
                else match.group(0),
                cleaned,
            )
        cleaned = _ARTICLE_BEFORE_ADVERSARY_RE.sub(r"\1", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def name_matches(self, text: str) -> list[re.Match[str]]:
        if self.name_pattern is None:
            return []
        return [match for match in self.name_pattern.finditer(text) if self.keeps_name_match(match)]

    # -- attribution prose --------------------------------------------------------------
    def actor_reference(self, text: str) -> bool:
        """Does the text refer to an actor, a cluster, a state or a place (label-blind)?"""
        return bool(
            self.name_matches(text)
            or _SHORT_FORM_RE.search(text)
            or _CLUSTER_RE.search(text)
            or _ATTACK_ID_RE.search(text)
            or _GAZETTEER_RE.search(text)
            or _SPONSOR_RE.search(text)
        )

    def heading_is_attribution(self, heading: str) -> bool:
        """A section heading about attribution: an actor reference plus an attribution word."""
        return bool(_HEADING_CUE_RE.search(heading) and self.actor_reference(heading))

    def attribution_flags(
        self,
        texts: Sequence[str],
        *,
        paragraphs: Sequence[Any] | None = None,
        section_flags: Sequence[bool] | None = None,
    ) -> list[str | None]:
        """Per statement: "section", "sentence", "paragraph" or None (kept).

        A sentence is attribution prose when it refers to an actor *and* carries a cue; a
        paragraph with two or more such sentences goes entirely (vendors put the actor-less
        evidence, "a loader ... tied to the same domains", next to the claim); a statement
        under an attribution section goes too. Read on the original text, before any masking.
        """
        flags: list[str | None] = [None] * len(texts)
        for index, text in enumerate(texts):
            if section_flags is not None and section_flags[index]:
                flags[index] = "section"
            elif _CUE_RE.search(text) and self.actor_reference(text):
                flags[index] = "sentence"
        if paragraphs is not None:
            flagged = Counter(
                paragraphs[index]
                for index, flag in enumerate(flags)
                if flag == "sentence" and paragraphs[index] is not None
            )
            for index, paragraph in enumerate(paragraphs):
                if flags[index] is None and paragraph is not None and flagged[paragraph] >= 2:
                    flags[index] = "paragraph"
        return flags

    # -- strict and tradecraft masking ---------------------------------------------------
    def _software_spans(self, texts: Sequence[str]) -> list[list[tuple[int, int, Any]]]:
        joined, _ = _joined(texts)
        spans = [
            (match.start, match.end, match) for match in match_software_aliases(joined, self.kb)
        ]
        return _spans_per_statement(texts, spans)

    def mask(self, texts: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
        """Strict (and tradecraft) masking of statements whose group names are already replaced.

        Returns the statements (duplicates merged, order kept) and a record of what changed.
        """
        counts: Counter[str] = Counter()
        roles: Counter[str] = Counter()
        codenames: Counter[str] = Counter()
        protected: Counter[str] = Counter()
        texts = list(texts)
        inputs = list(texts)
        # names-level output may still hold bare short forms, cluster ids and object ids
        stage: list[str] = []
        software = self._software_spans(texts)
        for index, text in enumerate(texts):
            spans = software[index]

            def inside_software(start: int, end: int, spans=spans) -> bool:
                return any(start < s_end and s_start < end for s_start, s_end, _ in spans)

            def short_form(match: re.Match[str], inside=inside_software) -> str:
                if inside(match.start(), match.end()):
                    protected[match.group(0)] += 1
                    return match.group(0)
                counts["short_forms"] += 1
                return REPLACEMENT + (match.group("possessive") or "")

            text = _SHORT_FORM_RE.sub(short_form, text)
            stage.append(text)
        texts = stage
        stage = []
        for text in texts:

            def cluster(match: re.Match[str]) -> str:
                counts["cluster_names"] += 1
                return REPLACEMENT + (match.group("possessive") or "")

            def attack_id(match: re.Match[str]) -> str:
                counts["attack_ids"] += 1
                return _DELETED

            text = _CLUSTER_RE.sub(cluster, text)
            text = _ATTACK_ID_RE.sub(attack_id, text)
            stage.append(text)
        texts = stage
        # gazetteer, protected by software spans on the text as it now reads (Cuba, S0625)
        software = self._software_spans([text.replace(_DELETED, " ") for text in texts])
        stage = []
        for index, text in enumerate(texts):
            spans = software[index]
            for word in re.findall(r"\b[A-Z]{4,}\b", text):
                title = word.title()
                if title in GAZETTEER_KINDS and word not in GAZETTEER_KINDS:
                    codenames[word] += 1

            def gazetteer(match: re.Match[str], text=text, spans=spans) -> str:
                term = match.group("term")
                start, end = match.span()
                if any(start < s_end and s_start < end for s_start, s_end, _ in spans):
                    protected[term] += 1
                    return match.group(0)
                kind = GAZETTEER_KINDS.get(term)
                if kind is None:  # a stem
                    if term.startswith("Unit "):
                        kind = "agency"
                    elif term in ("Ukraine", "Belarus"):
                        kind = "country"
                    else:
                        kind = "demonym"
                acronym = kind.startswith("acronym_")
                if acronym and start > 0 and text[start - 1] in _DASH:
                    return match.group(0)  # en-US, zh-CN style locale codes
                suffix = (match.group("suffix") or "").lower()
                window = text[max(0, start - 60) : end + 60]
                origin_role = suffix in _ORIGIN_ROLE_SUFFIXES or bool(
                    _ORIGIN_CONTEXT_RE.search(window)
                )
                roles["origin" if origin_role else "victim_or_other"] += 1
                possessive = match.group("possessive") or ""
                if match.group("origin"):
                    if suffix in _ORIGIN_SUFFIXES:
                        counts["origin_compounds"] += 1
                        return _DELETED
                    if acronym:
                        return match.group(0)  # US-ASCII
                    counts[kind] += 1  # "Malaysian-focused" -> "undisclosed-focused"
                    suffix_term = match.group("suffix")
                    if suffix_term in GAZETTEER_KINDS:  # "Iraqi-Kurdistan", "Russia-Ukraine"
                        counts[GAZETTEER_KINDS[suffix_term]] += 1
                        return "undisclosed"
                    return "undisclosed" + match.group("origin")
                next_word = re.match(r" ([A-Za-z][\w-]*)", text[end:])
                adjectival = (
                    not possessive
                    and next_word is not None
                    and next_word.group(1).lower() not in _FUNCTION_WORDS
                )
                base = kind.removeprefix("acronym_")
                counts[kind] += 1
                if base in ("demonym", "region_adjective", "language") and adjectival:
                    return _DELETED
                if acronym and adjectival and next_word.group(1)[0].islower():
                    return _DELETED  # "DPRK operatives", "US companies"
                if base == "country":
                    return "an undisclosed country" + possessive
                if base == "region":
                    return "an undisclosed region" + possessive
                if base == "agency":
                    return "an undisclosed agency" + possessive
                if base == "capital":
                    return "an undisclosed city" + possessive
                return "undisclosed" + possessive

            def sponsor(match: re.Match[str]) -> str:
                counts["sponsor_terms"] += 1
                roles["origin"] += 1
                return _DELETED

            changed = _GAZETTEER_RE.sub(gazetteer, text)
            changed = _SPONSOR_RE.sub(sponsor, changed)

            def non_latin(match: re.Match[str]) -> str:
                counts["non_latin_runs"] += 1
                return "undisclosed"

            changed = _NON_LATIN_RE.sub(non_latin, changed)
            stage.append(changed)
        texts = stage
        malware_masked: Counter[str] = Counter()
        tools_kept: Counter[str] = Counter()
        if self.level == "tradecraft":
            software = self._software_spans([text.replace(_DELETED, " ") for text in texts])
            stage = []
            for index, text in enumerate(texts):
                spans = _merge_spans(
                    [span for span in software[index] if span[2].software_type == "malware"]
                )
                for _, _, match in software[index]:
                    if match.software_type != "malware":
                        tools_kept[match.software_id] += 1
                for start, end, match in reversed(spans):
                    malware_masked[match.software_id] += 1
                    text = text[:start] + MALWARE_REPLACEMENT + text[end:]
                stage.append(text)
            texts = stage
        output: list[str] = []
        seen: set[str] = set()
        merged = 0
        for original, text in zip(inputs, texts, strict=True):
            repaired = _repair(text) if text != original else text
            key = repaired.casefold()
            if key in seen:
                merged += 1
                continue
            seen.add(key)
            output.append(repaired)
        record: dict[str, Any] = {
            "level": self.level,
            "replacements": dict(sorted(counts.items())),
            "gazetteer_roles": dict(sorted(roles.items())),
            "codenames_kept": dict(sorted(codenames.items())),
            "protected_by_software": dict(sorted(protected.items())),
            "duplicates_merged": merged,
        }
        if self.level == "tradecraft":
            record["malware_masked"] = dict(sorted(malware_masked.items()))
            record["tools_kept"] = dict(sorted(tools_kept.items()))
        return output, record

    # -- residual audit ------------------------------------------------------------------
    def residuals(self, texts: Sequence[str]) -> list[dict[str, str]]:
        """Boundary-free search for names and places that survived masking (for review).

        Lists every group name/alias of five or more characters, gazetteer term and cluster id
        found anywhere in the statements, except ordinary-word aliases used as words and
        ALL-CAPS codenames (both kept on purpose) and software names that contain a term.
        """
        found: list[dict[str, str]] = []
        terms = [name for name in self.names if len(name) >= 5] + [
            term for term in GAZETTEER_KINDS if len(term) >= 5
        ]
        joined, _ = _joined(texts)
        software = [(match.start, match.end) for match in match_software_aliases(joined, self.kb)]
        lowered = joined.casefold()
        for term in sorted(set(terms), key=lambda item: (-len(item), item)):
            key = term.casefold()
            start = lowered.find(key)
            while start >= 0:
                end = start + len(term)
                word = joined[start:end]
                ordinary = key in self._guarded and word not in self._spellings.get(key, {word})
                codename = word.isupper() and word != term and term in GAZETTEER_KINDS
                in_software = any(start < s_end and s_start < end for s_start, s_end in software)
                inside_word = (start > 0 and joined[start - 1].isalnum()) or (
                    end < len(joined) and joined[end].isalnum()
                )
                if not (ordinary or codename or in_software or inside_word):
                    found.append({"term": term, "context": joined[max(0, start - 40) : end + 40]})
                start = lowered.find(key, end)
        for match in _CLUSTER_RE.finditer(joined):
            found.append(
                {
                    "term": match.group("name"),
                    "context": joined[match.start() - 40 : match.end() + 40],
                }
            )
        return found
