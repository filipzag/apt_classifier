"""The model's second opinion (``apt_classifier/model_opinion.py``), with fakes only.

No model is called, no key is needed and no real knowledge base is read. The registered runner
and scorer are never imported (the runner loads the benchmark engine): their prompt, schema and
matching functions are read from the source with ``ast`` and compared with the module's.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from apt_classifier import model_opinion as mo
from apt_classifier.attribution import AttributionDecision, AttributionStage
from apt_classifier.evidence.models import EvidenceLedger, EvidenceStatus, Observation
from apt_classifier.llm import LLMResponse, LLMUsage, OpenRouterError

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "fresh_llm_attribution.py"
SCORER = ROOT / "scripts" / "fresh_h2h_score.py"
PREREG = ROOT / "docs" / "preregistration" / "model_second_opinion_2026-11-11.json"
MODEL = mo.REGISTERED_MODEL
EFFORT = mo.REGISTERED_EFFORT
NOW = "2026-11-11T12:00:00+00:00"


# ----- reading the registered scripts without importing them


def _literals(path: Path, *names: str) -> dict[str, Any]:
    """Module-level constants of a script, read from its source."""
    found: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    assert sorted(found) == sorted(names)
    return found


def _extracted(path: Path, *names: str) -> dict[str, Any]:
    """The named module-level functions and constants of a script, compiled on their own."""
    body: list[ast.stmt] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        wanted = isinstance(node, ast.FunctionDef) and node.name in names
        wanted = wanted or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
        if wanted:
            body.append(node)
    # the names the functions use from their module's imports
    namespace: dict[str, Any] = {"defaultdict": defaultdict, "Any": Any}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    assert all(name in namespace for name in names)
    return namespace


@pytest.fixture(scope="module")
def scorer() -> dict[str, Any]:
    return _extracted(
        SCORER,
        "PROBABILITY_TOLERANCE",
        "name_key",
        "group_names",
        "_probability",
        "llm_rows",
        "confident",
    )


# ----- fakes


def _group(group_id: str, name: str, *aliases: str) -> SimpleNamespace:
    return SimpleNamespace(id=group_id, name=name, aliases=list(aliases))


def _kb() -> SimpleNamespace:
    groups = (
        _group("G0007", "APT28", "Fancy Bear", "Sednit", "STRONTIUM"),
        _group("G0016", "APT29", "Cozy Bear", "Midnight Blizzard"),
        _group("G0034", "Sandworm Team", "Sandworm", "Voodoo Bear"),
        _group("G1040", "Play"),
        _group("G0060", "BRONZE BUTLER", "Tick"),
        _group("G0091", "Silence", "Whisper Spider"),
        # two groups share a name, as Thrip does on ATT&CK 20.0
        _group("G0030", "Lotus Blossom", "Thrip"),
        _group("G0076", "Thrip"),
        _group("G0018", "admin@338"),
        # an invented group whose name is written in lower case
        _group("G9001", "greywolf"),
    )
    return SimpleNamespace(groups={group.id: group for group in groups}, version="test")


def _ledger(*texts: str, extra: tuple[Any, ...] = ()) -> SimpleNamespace:
    observations = [SimpleNamespace(text=text, status=EvidenceStatus.POSITIVE) for text in texts]
    return SimpleNamespace(observations=[*observations, *extra])


class Client:
    """A duck client: an OpenRouter key and ``complete``, which answers with what it was given."""

    def __init__(
        self, parsed: Any, *, api_key: str | None = "key-for-tests", answered_by: str | None = None
    ) -> None:
        self.api_key = api_key
        self.parsed = parsed
        self.answered_by = answered_by
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append((messages, kwargs))
        return LLMResponse(
            "{}",
            self.parsed,
            LLMUsage(3625, 1808, 0.0543),
            self.answered_by or kwargs["model"],
            {},
        )


def _candidate(name: str, probability: Any, group_id: str = "") -> dict[str, Any]:
    return {"group_id": group_id, "group_name": name, "probability": probability}


def _answer(decision: str, *candidates: dict[str, Any], rationale: str = "") -> dict[str, Any]:
    return {"candidates": list(candidates), "decision": decision, "rationale": rationale}


ANSWER = _answer(
    "name",
    _candidate("APT28", 0.99, "G0007"),
    _candidate("APT29", 0.01, "G0016"),
    rationale=" The loader and the lures point to one group. ",
)
APT28 = {"name": "APT28", "group_id": "G0007", "attack_name": "APT28", "probability": 0.99}
APT29 = {"name": "APT29", "group_id": "G0016", "attack_name": "APT29", "probability": 0.01}
RANKING = ("G0007", "G0016", "G0034", "G1040", "G0060", "G0091")


def _state(
    leader: str | None = "G0007",
    *,
    stage: str = "known_group_shortlist",
    named: tuple[str, ...] = (),
    ranking: tuple[str, ...] = RANKING,
    ledger: Any = None,
) -> SimpleNamespace:
    fusion = SimpleNamespace(entries=tuple(SimpleNamespace(group_id=gid) for gid in ranking))
    return SimpleNamespace(
        attribution_decision=AttributionDecision(
            stage=AttributionStage(stage),
            cluster_id=None,
            leading_group_id=leader,
            named_cluster=named,
        ),
        deterministic_fusion=fusion,
        fusion=fusion,
        ledger=ledger if ledger is not None else _ledger("APT28 sent spearphishing emails."),
    )


def _in_attack(group_id: str) -> dict[str, Any]:
    return {"name": group_id, "group_id": group_id, "attack_name": group_id, "probability": 0.6}


# ----- 1. the question is the registered one


def test_settings_are_the_registered_ones(scorer):
    assert mo.PROMPT_VERSION == 1
    assert (mo.REGISTERED_MODEL, mo.REGISTERED_EFFORT) == ("openai/gpt-5.6-sol", "medium")
    assert mo.CONFIDENT_PROBABILITY == 0.99
    assert mo.PROBABILITY_TOLERANCE == scorer["PROBABILITY_TOLERANCE"] == 1e-9
    assert (mo.MAX_TOKENS, mo.SCHEMA_NAME, mo.MAX_STATEMENTS) == (16000, "attribution", 400)
    assert mo.NOTE == (
        "The model's own answer and its own stated probability. It is not calibrated and "
        "is never part of the product's decision."
    )


def test_prompt_and_schema_equal_the_registered_runner():
    registered = _literals(RUNNER, "SYSTEM", "PROMPT", "SCHEMA", "DEFAULT_MODEL")
    assert registered["SYSTEM"] == mo.SYSTEM
    assert registered["PROMPT"].encode("utf-8") == mo.PROMPT.encode("utf-8")
    assert registered["SCHEMA"] == mo.SCHEMA
    assert registered["DEFAULT_MODEL"] == mo.REGISTERED_MODEL
    # the request settings the runner writes into its call
    source = RUNNER.read_text(encoding="utf-8")
    assert f'"json_schema": {{"name": "{mo.SCHEMA_NAME}", "strict": True' in source
    assert f'"max_tokens": {mo.MAX_TOKENS},' in source
    assert f'parser.add_argument("--effort", default="{mo.REGISTERED_EFFORT}")' in source


def test_statements_are_numbered_as_the_registered_runner_numbers_them():
    runner = _extracted(RUNNER, "PROMPT", "prompt_for")
    statements = ["The adversary ran {whoami}.", "A second statement.", "A third: 100% done."]
    assert mo.prompt_for(statements) == runner["prompt_for"](statements)
    assert mo.prompt_for(statements).endswith(
        "Statements:\n1. The adversary ran {whoami}.\n2. A second statement.\n3. A third: 100% done."
    )


# ----- 2. names are matched as the registered scorer matches them

NAME_TABLE = (
    "APT28",
    "apt28",
    "APT 28",
    "apt-28",
    "Fancy Bear",
    "FANCY  BEAR",
    "fancy_bear",
    "Sednit",
    "Thrip",
    "thrip",
    "Lotus Blossom",
    "admin@338",
    "Admin 338",
    "Sandworm",
    "sandworm team",
    "Velvet Otter",
    "Storm-9999",
    "G0007",
    "",
    " - ",
)


def test_name_matching_equals_the_registered_scorer(scorer):
    kb = _kb()
    names = mo.group_names(kb.groups)
    assert names == scorer["group_names"](kb.groups)
    for name in NAME_TABLE:
        assert mo.name_key(name) == scorer["name_key"](name), name
        assert names.get(mo.name_key(name)) == scorer["group_names"](kb.groups).get(
            scorer["name_key"](name)
        ), name

    def matched(name: str) -> str | None:
        return names.get(mo.name_key(name))

    assert matched("APT28") == matched("APT 28") == matched("apt-28") == "G0007"
    assert matched("FANCY  BEAR") == matched("fancy_bear") == "G0007"
    assert matched("admin@338") == matched("Admin 338") == "G0018"
    assert matched("sandworm team") == matched("Sandworm") == "G0034"
    # a name two groups share says nothing; the other name of one of them still does
    assert matched("Thrip") is None and "thrip" not in names
    assert matched("Lotus Blossom") == "G0030"
    # an invented name, an id written as a name and an empty name are no ATT&CK group
    assert matched("Velvet Otter") is matched("Storm-9999") is matched("G0007") is None
    assert matched("") is None and matched(" - ") is None


def test_the_models_group_id_is_ignored():
    kb = _kb()
    answer = mo.read_answer(
        _answer(
            "name",
            # the right name under another group's id: the name decides
            _candidate("Fancy Bear", 0.7, "G0016"),
            # a name ATT&CK does not list under a real id: no ATT&CK group
            _candidate("Velvet Otter", 0.2, "G0007"),
            _candidate("APT 29", 0.1, "G9999"),
        ),
        mo.group_names(kb.groups),
        kb,
    )
    assert answer["candidates"] == [
        {"name": "Fancy Bear", "group_id": "G0007", "attack_name": "APT28", "probability": 0.7},
        {"name": "Velvet Otter", "group_id": None, "attack_name": None, "probability": 0.2},
        {"name": "APT 29", "group_id": "G0016", "attack_name": "APT29", "probability": 0.1},
    ]


# ----- 3. reading the answer


def test_read_answer_keeps_the_models_order_and_text():
    kb = _kb()
    answer = mo.read_answer(ANSWER, mo.group_names(kb.groups), kb)
    assert answer == {
        "decision": "name",
        "rationale": "The loader and the lures point to one group.",
        "candidates": [APT28, APT29],
    }
    empty = mo.read_answer(_answer("not_in_attack"), mo.group_names(kb.groups), kb)
    assert empty == {"decision": "not_in_attack", "rationale": "", "candidates": []}


def test_read_answer_keeps_text_json_can_carry():
    """A broken \\u escape in the model's text must not fail every later view of the case."""
    kb = _kb()
    broken = _answer("name", _candidate("APT\ud83d28", 0.5))
    broken["rationale"] = "The lure matches \ud83d earlier activity. Größe, 日本, 🙂."
    answer = mo.read_answer(broken, mo.group_names(kb.groups), kb)
    json.dumps(answer, ensure_ascii=False).encode("utf-8")  # raises on a lone surrogate
    assert answer["candidates"][0]["group_id"] == "G0007"  # matched on the name as written
    assert answer["rationale"].endswith("Größe, 日本, 🙂.")


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        [],
        "name",
        _answer("maybe", _candidate("APT28", 0.9)),
        {"candidates": [], "rationale": ""},
        {"decision": "name", "rationale": ""},
        {"decision": "name", "candidates": "APT28", "rationale": ""},
        {"decision": "name", "candidates": {"group_name": "APT28"}, "rationale": ""},
        {"decision": "name", "candidates": [_candidate("APT28", 0.9), "APT29"], "rationale": ""},
    ],
)
def test_read_answer_refuses_an_answer_it_cannot_read(parsed):
    kb = _kb()
    with pytest.raises(OpenRouterError):
        mo.read_answer(parsed, mo.group_names(kb.groups), kb)


@pytest.mark.parametrize(
    ("stated", "read"),
    [
        (0.99, 0.99),
        (1, 1.0),
        (0, 0.0),
        (1.7, None),
        (-0.1, None),
        (99, None),
        ("0.99", None),
        (None, None),
        (True, None),
        (float("nan"), None),
    ],
)
def test_a_probability_outside_zero_to_one_is_no_probability(stated, read):
    kb = _kb()
    answer = mo.read_answer(
        _answer("name", _candidate("APT28", stated)), mo.group_names(kb.groups), kb
    )
    assert answer["candidates"][0]["probability"] == read
    assert answer["candidates"][0]["group_id"] == "G0007"


def test_read_answer_keeps_three_candidates():
    kb = _kb()
    names = ("APT28", "Velvet Otter", "APT29", "Play", "Silence")
    answer = mo.read_answer(
        _answer("open", *(_candidate(name, 0.2) for name in names)), mo.group_names(kb.groups), kb
    )
    assert [candidate["name"] for candidate in answer["candidates"]] == list(names[:3])
    assert [candidate["group_id"] for candidate in answer["candidates"]] == ["G0007", None, "G0016"]


# ----- 4. the confident name


def test_confident_name_is_the_registered_rule():
    def confident(decision: str = "name", first: Any = APT28, **kwargs: str) -> bool:
        return mo.is_confident_name(decision, first, **{"model": MODEL, "effort": EFFORT, **kwargs})

    assert confident() is True
    assert confident(first={**APT28, "probability": 0.98}) is False
    assert confident(first={**APT28, "probability": 0.9899999999995}) is True  # written as 0.99
    assert confident(first={**APT28, "probability": 1.0}) is True
    assert confident(first={**APT28, "probability": None}) is False
    assert confident(decision="open") is False
    assert confident(decision="not_in_attack") is False
    outside = {"name": "Velvet Otter", "group_id": None, "attack_name": None, "probability": 0.99}
    assert confident(first=outside) is False
    assert confident(first=None) is False
    # the cut-off was chosen for one model at one effort
    assert confident(model="anthropic/claude-opus-5.5") is False
    assert confident(effort="high") is False


def test_a_later_attack_candidate_never_makes_a_confident_name():
    kb = _kb()
    parsed = _answer("name", _candidate("Velvet Otter", 0.99), _candidate("APT28", 0.99, "G0007"))
    opinion = mo.ask(Client(parsed), kb, _ledger("A statement."), model=MODEL, effort=EFFORT)
    assert opinion.first == {
        "name": "Velvet Otter",
        "group_id": None,
        "attack_name": None,
        "probability": 0.99,
    }
    assert opinion.candidates[1]["group_id"] == "G0007"
    assert opinion.confident_name is False
    assert mo.payload(opinion, kb, _state())["agreement"]["kind"] == "outside_attack"


CONFIDENT_TABLE = (
    ("name", [_candidate("APT28", 0.99, "G0007")], True),
    ("name", [_candidate("APT28", 0.98, "G0007")], False),
    ("name", [_candidate("APT28", 0.9899999999995, "G0007")], True),
    ("name", [_candidate("APT28", 1, "G0007")], True),
    ("open", [_candidate("APT28", 0.99, "G0007")], False),
    ("not_in_attack", [_candidate("APT28", 0.99, "G0007")], False),
    ("name", [_candidate("Velvet Otter", 0.99, "G1048"), _candidate("APT28", 0.99)], False),
    ("name", [_candidate("APT 28", 0.995, "G0016")], True),
    ("name", [_candidate("Velvet Otter", 0.99, "G0007")], False),
    ("name", [_candidate("Thrip", 0.99, "G0030")], False),
    ("name", [], False),
    ("name", [_candidate("APT28", "0.99", "G0007")], False),
    ("name", [_candidate("fancy-bear", 0.999), _candidate("APT29", 0.001)], True),
)


@pytest.mark.parametrize(("decision", "raw", "expected"), CONFIDENT_TABLE)
def test_confident_name_equals_the_registered_scorer(scorer, decision, raw, expected):
    kb = _kb()
    row = {
        "report_id": "r",
        "arm": "known",
        "group_id": None,
        "candidates": [],
        "raw_candidates": raw,
        "decision": decision,
    }
    rows = scorer["llm_rows"]({"rows": [row]}, scorer["group_names"](kb.groups))
    registered = scorer["confident"](rows["r"], mo.CONFIDENT_PROBABILITY)

    answer = mo.read_answer(_answer(decision, *raw), mo.group_names(kb.groups), kb)
    first = answer["candidates"][0] if answer["candidates"] else None
    ours = mo.is_confident_name(answer["decision"], first, model=MODEL, effort=EFFORT)

    assert ours is registered["named"] is expected
    if expected:
        assert [first["group_id"]] == registered["named_members"]


# ----- 5. masking


def test_masking_hides_names_aliases_and_group_ids():
    masked, hidden = mo.mask_statements(
        [
            "APT28 sent spearphishing emails.",
            "The Fancy Bear operators reused Sednit's loader.",
            "An APT28-linked server hosted the payload.",
            "Sandworm’s wiper ran on the domain controllers.",
            "The group (G0007) is tracked as G0016 by others.",
            "See https://attack.mitre.org/groups/G0034/ for the profile.",
            "apt28 came back a week later.",
            "Midnight   Blizzard\nstaged the data.",
        ],
        _kb(),
    )
    assert masked == [
        "the adversary sent spearphishing emails.",
        "the adversary operators reused the adversary's loader.",
        "the adversary-linked server hosted the payload.",
        "the adversary’s wiper ran on the domain controllers.",
        "The group is tracked as by others.",
        "See https://attack.mitre.org/groups// for the profile.",
        "the adversary came back a week later.",
        "the adversary staged the data.",
    ]
    # seven names and three ids
    assert hidden == 10


def test_masking_leaves_ordinary_words_and_longer_tokens():
    kept = [
        "Users were asked to play the video and tick a box.",
        "The implant ran in silence until a sandworm sample was found.",
        "The loader apt28.dll was dropped next to sednit_loader.exe.",
        "Traffic went to cdn-apt28.example.com and play.example.net.",
        "The hash ends in 9fG0007a and the file is G0007.tmp.",
    ]
    assert mo.mask_statements(kept, _kb()) == (kept, 0)


def test_masking_replaces_names_as_attack_writes_them_and_names_with_digits_or_words():
    masked, hidden = mo.mask_statements(
        [
            "Play encrypted the file servers.",
            "TICK and SILENCE were both ruled out.",
            # ATT&CK writes these two in lower case
            "The greywolf operators and admin@338 used the same lure.",
            # several words: replaced in any case
            "the sandworm team and voodoo bear overlap.",
        ],
        _kb(),
    )
    assert masked == [
        "the adversary encrypted the file servers.",
        "the adversary and the adversary were both ruled out.",
        "the adversary operators and the adversary used the same lure.",
        "the adversary and the adversary overlap.",
    ]
    assert hidden == 7


def test_masking_hides_the_spellings_reports_use():
    """Short forms of "<X> Group/Team" names and spellings with or without a space or hyphen,
    on groups shaped like ATT&CK's (Lazarus Group lists no bare "Lazarus")."""
    kb = SimpleNamespace(
        groups={
            "G0032": _group("G0032", "Lazarus Group", "HIDDEN COBRA"),
            "G0047": _group("G0047", "Gamaredon Group"),
            "G0007": _group("G0007", "APT28", "Fancy Bear"),
            "G1015": _group("G1015", "Scattered Spider"),
            "G0069": _group("G0069", "MuddyWater"),
        }
    )
    masked, hidden = mo.mask_statements(
        [
            "Lazarus used a trojanised PDF reader.",
            "Gamaredon’s loader came back.",
            "Attributed to APT-28 (APT 28), also FancyBear and Fancy-Bear.",
            "Scattered-Spider and Lazarus-Group were ruled out.",
            "Tracked as APT28/G0007, see attack.mitre.org/versions/v16/groups/G0007/.",
        ],
        kb,
    )
    assert masked == [
        "the adversary used a trojanised PDF reader.",
        "the adversary’s loader came back.",
        "Attributed to the adversary (the adversary), also the adversary and the adversary.",
        "the adversary and the adversary were ruled out.",
        "Tracked as the adversary/, see attack.mitre.org/versions/v16/groups//.",
    ]
    # nine names and two ids
    assert hidden == 11
    kept = [
        "lazarus.dll and gamaredon.example.net stay, and so does the lazarus sample.",
        "APT 281 and APT28.dll are other tokens.",
        # a camel-case name split in two, and model numbers outside the group ids
        "Muddy Water used a Netgear G3100 router and a Pentium G4560 host.",
        "Short forms come only from names that have them: Cobalt stays.",
    ]
    assert mo.mask_statements(kept, kb) == (kept, 0)


def test_masking_gives_one_statement_per_statement():
    statements = ["G0007", "  Nothing to hide here.  ", "APT28"]
    assert mo.mask_statements(statements, _kb()) == (
        ["", "Nothing to hide here.", "the adversary"],
        2,
    )
    # a knowledge base without groups hides nothing but ids
    assert mo.mask_statements(["APT28 (G0007)"], SimpleNamespace(groups={})) == (["APT28"], 1)
    assert mo.mask_statements([], _kb()) == ([], 0)


def test_text_fingerprint():
    assert mo.text_fingerprint(["a", "b"]) == (
        "7e18f737311b2dc3b2f269dd78396b0351f14fb66efa879f768cb23181883c78"
    )
    assert mo.text_fingerprint(["a", "b"]) != mo.text_fingerprint(["b", "a"])


# ----- 6. the statements of a case


def test_case_statements_are_the_positive_observations_in_order():
    ledger = EvidenceLedger(
        observations=[
            Observation(text="First action.", status="positive"),
            Observation(text="Nothing of the kind was seen.", status="negative"),
            Observation(text="Second action.", status="positive"),
            Observation(text="Nobody looked.", status="not_visible"),
            Observation(text="Unclear.", status="unknown"),
        ]
    )
    assert mo.case_statements(ledger) == ["First action.", "Second action."]
    duck = _ledger(
        "  Padded.  ",
        "",
        extra=(
            SimpleNamespace(text="Negative.", status=EvidenceStatus.NEGATIVE),
            SimpleNamespace(text="Plain status.", status="positive"),
            SimpleNamespace(text="   ", status=EvidenceStatus.POSITIVE),
        ),
    )
    assert mo.case_statements(duck) == ["Padded.", "Plain status."]
    assert mo.case_statements(SimpleNamespace(observations=[])) == []
    assert mo.case_statements(None) == []


# ----- 7. the call


def test_ask_sends_the_tested_question_with_the_names_hidden():
    kb = _kb()
    client = Client(ANSWER)
    ledger = _ledger(
        "APT28 sent spearphishing emails.",
        "The loader was signed (G0007).",
        extra=(SimpleNamespace(text="No lateral movement.", status=EvidenceStatus.NEGATIVE),),
    )
    opinion = mo.ask(client, kb, ledger, model=MODEL, effort=EFFORT, now=NOW)

    sent = ["the adversary sent spearphishing emails.", "The loader was signed."]
    ((messages, kwargs),) = client.calls
    assert messages == [
        {"role": "system", "content": mo.SYSTEM},
        {"role": "user", "content": mo.prompt_for(sent)},
    ]
    assert "APT28" not in messages[1]["content"] and "G0007" not in messages[1]["content"]
    assert "No lateral movement" not in messages[1]["content"]
    assert kwargs["model"] == MODEL and kwargs["effort"] == EFFORT
    assert kwargs["schema"] == mo.SCHEMA and kwargs["schema"] is not mo.SCHEMA
    assert kwargs["schema_name"] == "attribution"
    assert kwargs["max_tokens"] == 16000
    # tested without a seed
    assert kwargs.get("seed") is None
    assert set(kwargs) <= {"model", "effort", "schema", "schema_name", "max_tokens", "seed"}

    assert opinion == mo.ModelOpinion(
        asked_at=NOW,
        model=MODEL,
        requested_model=MODEL,
        effort=EFFORT,
        prompt_version=mo.PROMPT_VERSION,
        decision="name",
        rationale="The loader and the lures point to one group.",
        candidates=(APT28, APT29),
        statements_sent=2,
        statements_total=2,
        names_masked=2,
        fingerprint=mo.text_fingerprint(sent),
        prompt_tokens=3625,
        completion_tokens=1808,
        cost_usd=0.0543,
    )
    assert opinion.first == APT28
    assert opinion.confident_name is True


def test_ask_stamps_the_time_in_utc():
    opinion = mo.ask(Client(ANSWER), _kb(), _ledger("A statement."), model=MODEL, effort=EFFORT)
    asked = datetime.fromisoformat(opinion.asked_at)
    assert asked.utcoffset() == timedelta(0)
    assert abs(datetime.now(asked.tzinfo) - asked) < timedelta(minutes=1)


def test_ask_refuses_without_an_openrouter_key():
    ledger = _ledger("A statement.")
    with pytest.raises(mo.ModelOpinionUnavailable, match="OpenRouter key"):
        mo.ask(None, _kb(), ledger, model=MODEL, effort=EFFORT)
    for client in (Client(ANSWER, api_key=None), Client(ANSWER, api_key="")):
        with pytest.raises(mo.ModelOpinionUnavailable, match="OpenRouter key"):
            mo.ask(client, _kb(), ledger, model=MODEL, effort=EFFORT)
        assert client.calls == []
    # a client that says nothing about a key has none
    keyless = SimpleNamespace(complete=lambda *args, **kwargs: pytest.fail("called"))
    with pytest.raises(mo.ModelOpinionUnavailable):
        mo.ask(keyless, _kb(), ledger, model=MODEL, effort=EFFORT)
    # the refusal comes before anything else is looked at
    with pytest.raises(mo.ModelOpinionUnavailable):
        mo.ask(None, _kb(), _ledger(), model=MODEL, effort=EFFORT)


def test_ask_refuses_a_local_model():
    client = Client(ANSWER)
    with pytest.raises(mo.ModelOpinionUnavailable, match="cloud model"):
        mo.ask(client, _kb(), _ledger("A statement."), model="local/qwen3.5:4b-text", effort=EFFORT)
    assert client.calls == []


def test_ask_shows_no_answer_a_local_model_gave():
    client = Client(ANSWER, answered_by="local/qwen3.5:4b-text")
    with pytest.raises(OpenRouterError, match="local model") as raised:
        mo.ask(client, _kb(), _ledger("A statement."), model=MODEL, effort=EFFORT)
    assert not isinstance(raised.value, mo.ModelOpinionUnavailable)


def test_ask_sends_at_most_the_statement_cap():
    kb = _kb()
    statements = [f"Statement {index} of the case." for index in range(1, mo.MAX_STATEMENTS + 8)]
    client = Client(ANSWER)
    opinion = mo.ask(client, kb, _ledger(*statements), model=MODEL, effort=EFFORT)
    assert opinion.statements_sent == mo.MAX_STATEMENTS == 400
    assert opinion.statements_total == 407
    assert opinion.fingerprint == mo.text_fingerprint(statements[: mo.MAX_STATEMENTS])
    content = client.calls[0][0][1]["content"]
    assert content.endswith("\n400. Statement 400 of the case.")
    assert "Statement 401 of the case." not in content


def test_ask_needs_statements():
    client = Client(ANSWER)
    negative = SimpleNamespace(text="Not seen.", status=EvidenceStatus.NEGATIVE)
    # a statement that is nothing but a group id masks to nothing: no paid call either
    for ledger in (_ledger(), _ledger(extra=(negative,)), _ledger("   "), _ledger("(G0007)")):
        with pytest.raises(ValueError, match="the case has no statements to send"):
            mo.ask(client, _kb(), ledger, model=MODEL, effort=EFFORT)
    assert client.calls == []


def test_ask_passes_on_an_answer_it_cannot_read():
    client = Client(_answer("maybe", _candidate("APT28", 0.9)))
    with pytest.raises(OpenRouterError, match="decision"):
        mo.ask(client, _kb(), _ledger("A statement."), model=MODEL, effort=EFFORT)


def test_another_model_or_effort_never_gives_a_confident_name():
    kb = _kb()
    ledger = _ledger("A statement.")
    for model, effort in (("anthropic/claude-opus-5.5", EFFORT), (MODEL, "high")):
        opinion = mo.ask(Client(ANSWER), kb, ledger, model=model, effort=effort)
        assert (opinion.model, opinion.requested_model, opinion.effort) == (model, model, effort)
        assert opinion.first == APT28 and opinion.confident_name is False
        block = mo.payload(opinion, kb, _state(ledger=ledger))
        assert block["confident_name"] is False
        assert block["confident_rule"] == {"probability": 0.99, "applies": False}
    # the model that answered counts, not the one that was asked for
    opinion = mo.ask(
        Client(ANSWER, answered_by="openai/another-model"), kb, ledger, model=MODEL, effort=EFFORT
    )
    assert opinion.requested_model == MODEL and opinion.model == "openai/another-model"
    assert opinion.confident_name is False


# ----- 8. how the answer relates to the product's


def test_agreement_without_a_comparison():
    kb = _kb()
    outside = {"name": "Velvet Otter", "group_id": None, "attack_name": None, "probability": 0.9}
    no_leader = _state(None, stage="insufficient_evidence")
    assert mo.agreement(kb, _state(), None)["kind"] == "no_candidate"
    assert mo.agreement(kb, _state(), outside)["kind"] == "outside_attack"
    # a first choice outside ATT&CK is said first, whatever the product has
    assert mo.agreement(kb, no_leader, outside)["kind"] == "outside_attack"
    assert mo.agreement(kb, no_leader, APT28)["kind"] == "no_product_leader"
    for state in (None, SimpleNamespace(), SimpleNamespace(attribution_decision=None)):
        assert mo.agreement(kb, state, APT28)["kind"] == "no_product_leader"


def test_agreement_with_the_products_first_choice_and_named_pair(monkeypatch):
    monkeypatch.setattr(mo, "link_kind", lambda *args: pytest.fail("not needed for these tiers"))
    kb = _kb()
    assert mo.agreement(kb, _state("G0007"), APT28) == {
        "kind": "same",
        "text": "Same group as the product's first choice.",
    }
    pair = _state("G0034", stage="known_cluster_hypothesis", named=("G0034", "G0007"))
    assert mo.agreement(kb, pair, _in_attack("G0034"))["kind"] == "same"
    assert mo.agreement(kb, pair, APT28) == {
        "kind": "named_pair",
        "text": "One of the two groups the product names together.",
    }


def test_agreement_through_documented_overlap(monkeypatch):
    kb = _kb()
    calls = []

    def linked(kind):
        def link_kind(*args):
            calls.append(args)
            return kind

        return link_kind

    outsider = _in_attack("G9001")  # not in the product's ranking
    for kind in ("same_activity", "nesting"):
        monkeypatch.setattr(mo, "link_kind", linked(kind))
        assert mo.agreement(kb, _state("G0034"), APT28)["kind"] == "same_activity"
        assert mo.agreement(kb, _state("G0034"), outsider)["kind"] == "same_activity"
    assert calls[0] == (kb, "G0034", "G0007")
    # a mention or shared malware is no overlap; a pair left over from a stage that names no
    # pair is no named pair either
    for kind in ("description", "shared_software", None):
        monkeypatch.setattr(mo, "link_kind", linked(kind))
        assert mo.agreement(kb, _state("G0034"), APT28)["kind"] == "shortlist"
        assert mo.agreement(kb, _state("G0034"), outsider)["kind"] == "different"
        leftover = _state("G0034", named=("G0034", "G9001"))
        assert mo.agreement(kb, leftover, outsider)["kind"] == "different"


def test_agreement_survives_a_failing_overlap_lookup(monkeypatch):
    kb = _kb()

    def broken(*args):
        raise KeyError("no such group")

    monkeypatch.setattr(mo, "link_kind", broken)
    assert mo.agreement(kb, _state("G0034"), APT28)["kind"] == "shortlist"
    assert mo.agreement(kb, _state("G0034"), _in_attack("G9001"))["kind"] == "different"


def test_agreement_with_the_real_overlap_lookup_on_a_bare_knowledge_base():
    # the fake groups have no descriptions, techniques or software: whatever the lookup makes
    # of that, the block still gets an answer
    kb = _kb()
    assert mo.agreement(kb, _state("G0034"), APT28)["kind"] == "shortlist"
    assert mo.agreement(kb, _state("G0034"), _in_attack("G9001"))["kind"] == "different"


def test_agreement_reads_the_products_first_five(monkeypatch):
    monkeypatch.setattr(mo, "link_kind", lambda *args: None)
    kb = _kb()
    state = _state("G0007")
    assert mo.agreement(kb, state, _in_attack("G0060")) == {
        "kind": "shortlist",
        "text": "Not the product's first choice, but among the product's first five.",
    }
    assert mo.agreement(kb, state, _in_attack("G0091"))["kind"] == "different"
    # the ranking the decision was made on, not the one a later vote reordered
    state.fusion = SimpleNamespace(entries=(SimpleNamespace(group_id="G0091"),))
    assert mo.agreement(kb, state, _in_attack("G0091"))["kind"] == "different"
    state.deterministic_fusion = None
    assert mo.agreement(kb, state, _in_attack("G0091"))["kind"] == "shortlist"
    state.fusion = None
    assert mo.agreement(kb, state, _in_attack("G0091"))["kind"] == "different"


def test_every_agreement_kind_has_one_plain_sentence():
    assert mo.AGREEMENT_KINDS == (
        "no_candidate",
        "outside_attack",
        "no_product_leader",
        "same",
        "named_pair",
        "same_activity",
        "shortlist",
        "different",
    )
    assert set(mo._AGREEMENT_TEXT) == set(mo.AGREEMENT_KINDS)
    for text in mo._AGREEMENT_TEXT.values():
        assert text.endswith(".") and text.count(".") == 1 and "\n" not in text


# ----- 9. the block for the API

PAYLOAD_KEYS = {
    "asked_at",
    "model",
    "effort",
    "prompt_version",
    "decision",
    "first",
    "candidates",
    "confident_name",
    "confident_rule",
    "rationale",
    "agreement",
    "statements_sent",
    "statements_total",
    "names_masked",
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "stale",
    "note",
}


def test_payload_has_the_agreed_shape():
    kb = _kb()
    ledger = _ledger("APT28 sent spearphishing emails.", "The loader was signed (G0007).")
    opinion = mo.ask(Client(ANSWER), kb, ledger, model=MODEL, effort=EFFORT, now=NOW)
    state = _state("G0007", ledger=ledger)
    before = copy.deepcopy(state)

    block = mo.payload(opinion, kb, state)
    assert set(block) == PAYLOAD_KEYS
    assert block == {
        "asked_at": NOW,
        "model": "openai/gpt-5.6-sol",
        "effort": "medium",
        "prompt_version": 1,
        "decision": "name",
        "first": APT28,
        "candidates": [APT28, APT29],
        "confident_name": True,
        "confident_rule": {"probability": 0.99, "applies": True},
        "rationale": "The loader and the lures point to one group.",
        "agreement": {"kind": "same", "text": "Same group as the product's first choice."},
        "statements_sent": 2,
        "statements_total": 2,
        "names_masked": 2,
        "cost_usd": 0.0543,
        "prompt_tokens": 3625,
        "completion_tokens": 1808,
        "stale": False,
        "note": mo.NOTE,
    }
    assert json.loads(json.dumps(block)) == block
    # a pure read: neither the state nor the stored answer changes, also through the block
    block["first"]["group_id"] = "changed"
    block["candidates"][1]["group_id"] = "changed"
    assert state == before
    assert opinion.candidates == (APT28, APT29)
    assert mo.payload(opinion, kb, state)["first"] == APT28


def test_payload_without_an_answer_or_a_candidate():
    kb = _kb()
    assert mo.payload(None, kb, _state()) is None
    ledger = _ledger("A statement.")
    opinion = mo.ask(Client(_answer("not_in_attack")), kb, ledger, model=MODEL, effort=EFFORT)
    block = mo.payload(opinion, kb, _state(ledger=ledger))
    assert set(block) == PAYLOAD_KEYS
    assert block["decision"] == "not_in_attack"
    assert block["first"] is None and block["candidates"] == []
    assert block["confident_name"] is False
    assert block["confident_rule"] == {"probability": 0.99, "applies": True}
    assert block["agreement"]["kind"] == "no_candidate"
    assert block["stale"] is False


def test_payload_is_stale_once_the_text_a_call_would_send_changes():
    kb = _kb()
    ledger = _ledger("APT28 sent spearphishing emails.", "The loader was signed.")
    opinion = mo.ask(Client(ANSWER), kb, ledger, model=MODEL, effort=EFFORT)
    state = _state(ledger=ledger)

    def stale() -> bool:
        return mo.payload(opinion, kb, state)["stale"]

    assert stale() is False and stale() is False
    # what is not sent does not count: a negative observation, another spelling of a hidden name
    ledger.observations.append(SimpleNamespace(text="Not seen.", status=EvidenceStatus.NEGATIVE))
    ledger.observations[0].text = "Fancy Bear sent spearphishing emails."
    assert stale() is False
    ledger.observations[0].text = "APT28 sent spearphishing emails with a new lure."
    assert stale() is True
    ledger.observations[0].text = "APT28 sent spearphishing emails."
    assert stale() is False
    ledger.observations.append(SimpleNamespace(text="One more.", status=EvidenceStatus.POSITIVE))
    assert stale() is True
    # a state without a ledger has none of the text the answer was given for
    assert mo.payload(opinion, kb, SimpleNamespace())["stale"] is True


def test_staleness_tells_one_statement_with_a_line_break_from_two():
    kb = SimpleNamespace(groups={})
    kept = SimpleNamespace(observations=[])
    for texts in (("one", "two"), ("one\ntwo",), ("one two",)):
        kept.observations = [SimpleNamespace(text=text, status="positive") for text in texts]
        masked = mo.mask_statements(mo.case_statements(kept), kb)[0]
        assert mo._sent_fingerprint(kept, kb) == mo.text_fingerprint(masked)
    assert mo._sent_fingerprint(kept, kb) == mo.text_fingerprint(["one two"])


# ----- 10. display only


ENGINE_MODULES = (
    "session.py",
    "attribution.py",
    "fusion.py",
    "combiner.py",
    "separation.py",
    "retrieve.py",
    "report.py",
    "navigator.py",
    "attack_flow.py",
)


def test_engine_modules_never_mention_the_second_opinion():
    package = ROOT / "apt_classifier"
    classifiers = sorted((package / "classifiers").glob("*.py"))
    assert len(classifiers) > 5
    for path in [*(package / name for name in ENGINE_MODULES), *classifiers]:
        assert "model_opinion" not in path.read_text(encoding="utf-8"), path.name


def _imports_the_module(source: str) -> bool:
    """Any import of the module, also wrapped over several lines or inside a function."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [part for alias in node.names for part in alias.name.split(".")]
        elif isinstance(node, ast.ImportFrom):
            names = [*(node.module or "").split("."), *(alias.name for alias in node.names)]
        else:
            continue
        if "model_opinion" in names:
            return True
    return False


def test_the_import_check_sees_every_form():
    seen = (
        "from apt_classifier import model_opinion",
        "from apt_classifier import (\n    cases,\n    model_opinion,\n)",
        "from apt_classifier.model_opinion import (\n    ask,\n)",
        "import apt_classifier.model_opinion as second",
        "from . import model_opinion",
        "def f():\n    from apt_classifier import model_opinion\n",
    )
    assert all(_imports_the_module(source) for source in seen)
    assert not _imports_the_module("model_opinion_enabled = False\npayload['model_opinion'] = 1")


def test_only_the_service_imports_the_module():
    allowed = {Path("apt_classifier/model_opinion.py"), Path("apt_classifier/service.py")}
    for folder in ("apt_classifier", "api", "app"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            relative = path.relative_to(ROOT)
            if relative not in allowed:
                source = path.read_text(encoding="utf-8")
                assert not _imports_the_module(source), str(relative)


def test_importing_the_module_loads_no_model_or_index_library():
    probe = (
        "import sys, apt_classifier.model_opinion; "
        "print(sorted(set(sys.modules) & {'numpy', 'scipy', 'sklearn', 'torch', 'chromadb', "
        "'transformers', 'sentence_transformers'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "OPENROUTER_API_KEY": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


# ----- 11. what the registration pins


def test_registration_fingerprint_follows_every_pinned_setting(monkeypatch):
    before = mo.registration_fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", before)
    changes = {
        "PROMPT_VERSION": 2,
        "SYSTEM": "Another system line.",
        "PROMPT": "Another question.\n{statements}",
        "SCHEMA": {"type": "object"},
        "SCHEMA_NAME": "another",
        "REGISTERED_MODEL": "openai/another-model",
        "REGISTERED_EFFORT": "high",
        "CONFIDENT_PROBABILITY": 0.98,
        "MAX_TOKENS": 8000,
        "MAX_STATEMENTS": 200,
        "SHORT_FORMS": ("Lazarus",),
        "MAX_GROUP_ID": 2000,
        "REPLACEMENT": "the actor",
    }
    for name, value in changes.items():
        with monkeypatch.context() as patch:
            patch.setattr(mo, name, value)
            assert mo.registration_fingerprint() != before, name
    assert mo.registration_fingerprint() == before


def test_registration_fingerprint_follows_the_rules_not_their_comments(monkeypatch):
    before = mo.registration_fingerprint()

    def name_key(name: str) -> str:
        """Another docstring."""
        # another comment
        return "".join(character for character in name.lower() if character.isalnum())

    def other_name_key(name: str) -> str:
        return name.lower()

    with monkeypatch.context() as patch:
        patch.setattr(mo, "name_key", name_key)
        assert mo.registration_fingerprint() == before
        patch.setattr(mo, "name_key", other_name_key)
        assert mo.registration_fingerprint() != before


def test_the_registration_holds_this_fingerprint():
    if not PREREG.exists():
        pytest.skip("the registration is not in this checkout")
    registered = json.loads(PREREG.read_text(encoding="utf-8"))["code"]["module_fingerprint"]
    if registered == "PENDING":
        pytest.skip("the registration's module fingerprint is not filled in yet")
    assert registered == mo.registration_fingerprint()
