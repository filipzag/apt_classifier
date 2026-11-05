"""Knowledge store, step 2 (``apt_classifier.knowledge_store.extract``): hedges, software names,
the sentence and report rules, unclear links, actor claims and the dated, corroborated view.
Synthetic knowledge base, MISP lists and posts; nothing is downloaded."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from apt_classifier.knowledge_store.extract import (
    ActorNames,
    NameRules,
    SoftwareMatcher,
    hedge_of,
    introduced_names,
    protected_words,
    read_post,
)
from apt_classifier.knowledge_store.fetch import FetchResult
from apt_classifier.knowledge_store.store import KnowledgeStore
from scripts import knowledge_store as cli
from scripts import prepare_fresh_reports as fresh

ORDINARY = ("the", "update", "loader", "tool", "gold", "this", "server", "agent", "royal", "route")


def _kb():
    def software(software_id, name, kind, aliases=()):
        return SimpleNamespace(
            id=software_id,
            attack_id=software_id,
            name=name,
            software_type=kind,
            aliases=[name, *aliases],
            description="",
        )

    technique = SimpleNamespace(
        name="OS Credential Dumping",
        description="Adversaries may dump LSASS memory with PowerShell, as a shark bites.",
        detection="",
    )
    return SimpleNamespace(
        version="test",
        groups={
            "G0007": SimpleNamespace(
                name="APT28", aliases=["APT28", "Sednit", "Sofacy"], description=""
            ),
            "G0032": SimpleNamespace(
                name="Lazarus Group", aliases=["Lazarus Group"], description=""
            ),
            "G0080": SimpleNamespace(name="Cobalt Group", aliases=["Cobalt"], description=""),
        },
        software={
            "S0001": software("S0001", "X-Agent", "malware", ["CHOPSTICK"]),
            "S0002": software("S0002", "Mimikatz", "tool"),
            "S0003": software("S0003", "Shark", "malware"),  # an ordinary word in ATT&CK prose
            "S0004": software("S0004", "Gemini", "malware", ["Duplicate"]),
            "S0005": software("S0005", "Castor", "malware", ["Duplicate"]),
            "S0154": software("S0154", "Cobalt Strike", "tool"),
            "S0613": software("S0613", "PS1", "malware"),
            "S0690": software("S0690", "Royal", "malware"),
            "S0103": software("S0103", "route", "tool"),
            "S0625": software("S0625", "Cuba", "malware"),
            "S0075": software("S0075", "Reg", "tool"),
            "S0099": software("S0099", "Arp", "tool"),
            "S0137": software("S0137", "CORESHELL", "malware", ["Sofacy"]),
            "S0203": software("S0203", "Hydraq", "malware", ["Aurora"]),
        },
        techniques={"T1003": technique},
        data_components={},
    )


def _misp_dir(tmp_path):
    directory = tmp_path / "misp"
    directory.mkdir()
    values = [
        {"uuid": "m-gold", "value": "GOLDBACKDOOR"},
        {"uuid": "m-racoon", "value": "Agent Racoon"},
        {"uuid": "m-update", "value": "Update"},  # an ordinary word: never a name
        {"uuid": "m-mimi", "value": "Mimikatz"},  # ATT&CK has it
    ]
    (directory / "backdoor.json").write_text(json.dumps({"values": values}), encoding="utf-8")
    return directory


def _rules(kb):
    actors = ["APT28", "Sednit", "Lazarus", "Cobalt"]
    return NameRules(ORDINARY, protected_words(kb), actors, places=["Cuba", "Moscow"])


def _actors(kb):
    surface = {
        "APT28": "G0007",
        "Sednit": "G0007",
        "Lazarus": "G0032",
        "Cobalt": "G0080",
        "Karma": "G1055",
    }
    software = fresh.name_pattern(
        [name for item in kb.software.values() for name in item.aliases if len(name) > 3]
    )
    return ActorNames(surface, fresh.name_pattern(list(surface)), fresh.CLUSTER_CODE_RE, software)


def test_hedges_and_a_month_is_no_hedge():
    assert hedge_of("APT28 deployed X-Agent.") == "stated"
    assert hedge_of("APT28 deployed X-Agent in May 2024.") == "stated"
    assert hedge_of("APT28 may have deployed X-Agent.") == "possible"
    assert hedge_of("We assess with high confidence that APT28 deployed X-Agent.") == "confirmed"
    assert hedge_of("APT28 likely deployed X-Agent.") == "likely"
    assert hedge_of("A suspected APT28 operation deployed X-Agent.") == "possible"
    assert hedge_of("It is plausible that APT28 deployed X-Agent.") == "possible"


def test_name_rules_and_the_names_posts_introduce():
    rules = _rules(_kb())
    assert rules.allowed("HOOKEDGE")
    for word in ("Update", "PowerShell", "LSASS", "Mimikatz", "ESET", "Sednit", "RAT", "2024"):
        assert not rules.allowed(word), word
    assert not rules.allowed("Gold-Update")  # every part an ordinary word
    # a place, a malware noun, a description
    for word in ("Moscow", "Stealer", "PowerShell-based", "Base64-encoded"):
        assert not rules.allowed(word), word
    assert rules.allowed("TinyTurla-NG")
    statements = [
        "The HOOKEDGE backdoor reads its configuration.",
        "It drops a loader dubbed PteroSand on the host.",
        "It also runs a tool that we named TomCat.",
        "This loader runs the Update tool.",
        "The APT28 backdoor is new.",
    ]
    assert introduced_names(statements, rules) == {"HOOKEDGE", "PteroSand", "TomCat"}


def test_software_names_from_attack_misp_and_posts(tmp_path):
    kb = _kb()
    matcher = SoftwareMatcher(
        kb,
        _rules(kb),
        misp_dir=_misp_dir(tmp_path),
        introduced=["HOOKEDGE", "Mimikatz", "GOLDBACKDOOR"],
    )
    assert set(matcher.introduced) == {"HOOKEDGE"}  # the others are ATT&CK or MISP names

    def ids(text):
        return [item.id for _, _, item in matcher.find(text)]

    assert ids("They ran CHOPSTICK, GOLDBACKDOOR, HOOKEDGE and Agent Racoon.") == [
        "S0001",
        "misp:m-gold",
        "name:hookedge",
        "misp:m-racoon",
    ]
    # a malware name with inner capitals never in lowercase text; a short name before "script"
    # is a file type
    assert ids("They ran chopstick and Test.ps1 from a PS1 script.") == []
    assert ids("They ran Chopstick and PS1.") == ["S0001", "S0613"]
    # an ordinary-word name in its own spelling, a command word only as an .exe, a place only
    # before a malware noun
    assert ids("A royal decree; they deployed Royal and Cuba ransomware.") == ["S0690", "S0625"]
    assert ids("They chose a route via Cuba, then ran route.exe.") == ["S0103"]
    # a short tool name only as a command; a group alias only before a malware noun; never
    # before a trademark sign
    assert ids("Hosted at reg[.]ru, found by ARP-based discovery.") == []
    assert ids("They ran reg save HKLM and arp -a.") == ["S0075", "S0099"]
    assert ids("The Sofacy group returned with Sofacy malware.") == ["S0137"]
    assert ids("Aurora® detections and the Aurora backdoor.") == ["S0203"]
    assert ids("They ran the Update tool and Cobalt Strike.") == ["S0154"]
    # an ATT&CK alias the prose uses as a word: exact spelling, never title case at a start
    assert ids("Shark was deployed.") == []
    assert ids("They deployed Shark.") == ["S0003"]
    assert ids("They deployed a shark.") == []
    assert ids("They deployed Duplicate.") == []  # two ATT&CK entries share the alias


def test_actor_names_are_not_malware_names_or_tactic_ids():
    actors = _actors(_kb())
    text = "Karma ransomware, exfiltration [TA0010] and APT28 with Silence.Downloader."
    assert [actor for _, _, actor in actors.spans(text)] == ["G0007"]


def _read(statements, label=None):
    kb = _kb()
    matcher = SoftwareMatcher(kb, _rules(kb), introduced=["HOOKEDGE"])
    labelled = label or {"label": "known", "group_id": "G0007", "mentions": {"G0007": 5}}
    return read_post(
        "APT28 returns",
        statements,
        actors=_actors(kb),
        software=matcher,
        label_fn=lambda title, texts: labelled,
    )


def _links(reading):
    return {
        (link.rule, link.actor, link.software_id, link.status, link.reason, link.hedge)
        for link in reading.links
    }


def test_sentence_and_report_rules_and_unclear_links():
    statements = [
        "APT28 deployed X-Agent against ministries.",
        "The HOOKEDGE backdoor was likely installed for persistence.",
        "HOOKEDGE beacons to its server every hour.",
        "Lazarus also used Mimikatz, similar to APT28.",
        "Mimikatz dumped credentials from memory.",
        "APT28 may have used Cobalt Strike in May 2024.",
        "APT28 overlaps with UNC1234 activity.",
        "Sednit is also known as APT28.",
        "APT28 and Lazarus both used X-Agent.",
    ]
    reading = _read(statements)
    assert _links(reading) == {
        ("sentence", "G0007", "S0001", "active", "", "stated"),
        ("sentence", "G0032", "S0002", "unclear", "comparison", "stated"),
        ("sentence", "G0007", "S0002", "unclear", "comparison", "stated"),
        ("sentence", "G0007", "S0154", "active", "", "possible"),  # Cobalt is inside the name
        ("sentence", "G0007", "S0001", "unclear", "several_actors", "stated"),
        ("sentence", "G0032", "S0001", "unclear", "several_actors", "stated"),
        ("report", "G0007", "name:hookedge", "active", "", "likely"),
        ("report", "G0007", "S0002", "unclear", "other_actor_or_comparison", "stated"),
        ("report", "G0007", "S0001", "unclear", "other_actor_or_comparison", "stated"),
    }
    # two names of one actor make no claim, nor does a comparison; an actor and a code do
    assert [(c.actor_a, c.actor_b, c.relation, c.statement) for c in reading.claims] == [
        ("G0007", "code:UNC1234", "overlaps", 6)
    ]


def test_comparisons_lures_and_a_report_link_needs_a_use_sentence():
    statements = [
        "APT28 deployed a wiper with the same license key as X-Agent.",
        "APT28 used sites spoofing Mimikatz downloads.",
        "Lazarus used CHOPSTICK, which was released by the Shadow Brokers in 2017.",
        "HOOKEDGE beacons to its server every hour.",
        "HOOKEDGE reads files from the disk.",
    ]
    assert _links(_read(statements)) == {
        ("sentence", "G0007", "S0001", "unclear", "comparison", "stated"),
        ("sentence", "G0007", "S0002", "unclear", "lure", "stated"),
        ("sentence", "G0032", "S0001", "unclear", "leak", "stated"),
        ("report", "G0007", "S0001", "unclear", "other_actor_or_comparison", "stated"),
    }  # HOOKEDGE is named twice but never said to be used


def test_an_excluded_post_keeps_sentence_links_but_makes_no_report_links():
    statements = ["APT28 deployed X-Agent.", "X-Agent beacons home.", "X-Agent reads files."]
    reading = _read(statements, {"label": "excluded", "reason": "too_short", "mentions": {}})
    assert _links(reading) == {("sentence", "G0007", "S0001", "active", "", "stated")}


def _html(title, published, sentences):
    body = "".join(f"<p>{sentence}</p>" for sentence in sentences)
    return (
        f'<html><head><title>{title}</title><meta property="article:published_time" '
        f'content="{published}T10:00:00+00:00"></head><body><article>{body}</article></body></html>'
    )


def test_extraction_over_the_store_and_the_corroborated_view(tmp_path, monkeypatch):
    words = tmp_path / "words"
    words.write_text("\n".join(ORDINARY), encoding="utf-8")
    monkeypatch.setattr(fresh, "WORDS", words)
    store = KnowledgeStore(tmp_path / "store")
    posts = [("securelist", "2024-05-01"), ("kaspersky_ics", "2024-06-01"), ("v", "2024-07-01")]
    store.add_posts(
        [
            {"key": f"{source}.test/p", "source": source, "url": f"https://{source}.test/p"}
            for source, _ in posts
        ]
    )
    for number, (source, published) in enumerate(posts):
        sentences = [
            "APT28 deployed X-Agent against ministries.",
            "APT28 used the HOOKEDGE backdoor for persistence.",
            "APT28 sent spearphishing mail to diplomats.",
            *(f"The operators renamed file {number}-{n} on the host." for n in range(14)),
        ]
        store.record_fetch(
            f"{source}.test/p",
            source,
            FetchResult(
                "fetched", http_status=200, content=_html("APT28", published, sentences).encode()
            ),
            at="2026-11-03T10:00:00+00:00",
        )
    store.parse()
    store.finalise()
    summary = cli.extract_facts(
        store, kb=_kb(), misp={"values": []}, misp_dir=_misp_dir(tmp_path), workers=1
    )
    assert summary["labels"] == {"known": 3}
    assert summary["software_names_kept"] == 1  # HOOKEDGE
    labels = {row["key"]: row["label"] for row in store.db.execute("SELECT * FROM readings")}
    assert labels == {f"{source}.test/p": "known" for source, _ in posts}
    sentence = store.db.execute(
        "SELECT sentence FROM links WHERE software_id = 'S0001' AND rule = 'sentence' LIMIT 1"
    ).fetchone()["sentence"]
    assert sentence == "APT28 deployed X-Agent against ministries."

    def pair(as_of):
        return next(
            item
            for item in store.links_view(as_of)
            if (item["actor"], item["software_id"]) == ("G0007", "S0001")
        )

    # securelist and kaspersky_ics are one publisher: a second one corroborates
    assert pair(date(2024, 6, 15))["publishers"] == ["kaspersky"]
    assert not pair(date(2024, 6, 15))["corroborated"]
    assert pair(date(2024, 8, 1))["corroborated"]
    assert pair(date(2024, 8, 1))["posts"] == 3
    assert store.view(date(2024, 5, 2)).links()[0]["first"] == "2024-05-01"
    assert json.loads((store.root / "EXTRACT.json").read_text())["pairs_corroborated"] >= 1


def test_a_name_counts_only_when_most_posts_mentioning_it_introduce_it(tmp_path):
    store = KnowledgeStore(tmp_path / "store")
    days = ["2024-01-10", "2024-02-10", "2024-03-10", "2024-04-10", "2024-05-10"]
    store.add_posts(
        [{"key": f"v.test/{n}", "source": "v", "url": f"https://v.test/{n}"} for n in range(5)]
    )
    for n, day in enumerate(days):
        store.db.execute(
            "INSERT INTO pages(key, raw_sha256, parser_version, lines, n_lines, available_from) "
            "VALUES(?, '', 1, x'', 0, ?)",
            (f"v.test/{n}", day),
        )
        reading = {
            "label": "known",
            "actor": "G0007",
            "reason": None,
            "mentions": {},
            "claims": [],
            # NEWDOOR introduced in the first two posts; Golang mentioned in all, introduced once
            "names": {"name:newdoor": int(n < 2), "name:golang": int(n == 0)},
            "links": [
                {
                    "actor": "G0007",
                    "software_id": software_id,
                    "software_name": name,
                    "software_kind": "report",
                    "rule": "sentence",
                    "hedge": "stated",
                    "statement": 0,
                    "status": "active",
                    "reason": "",
                }
                for software_id, name in (("name:newdoor", "NEWDOOR"), ("name:golang", "Golang"))
            ],
        }
        store.save_reading(f"v.test/{n}", reading, {0: "APT28 used it."}, 2)
    assert store.introduced_software(date(2024, 1, 10)) == set()  # nothing before the first
    assert store.introduced_software(date(2024, 3, 1)) == {"name:newdoor", "name:golang"}
    # by May, Golang is introduced in 1 of the 4 earlier posts that mention it
    assert store.introduced_software(date(2024, 5, 1)) == {"name:newdoor", "name:golang"}
    assert store.introduced_software() == {"name:newdoor"}  # 1 of 5 is under a quarter
    assert [pair["software_id"] for pair in store.links_view()] == ["name:newdoor"]
    assert len(store.links_view(date(2024, 3, 1))) == 2


def _verdict_payload(verdict: str, p_supported: float) -> dict:
    import math

    head = '{"reason": "The text says so.", "verdict": "'
    return {
        "message": {"content": head + verdict + '"}'},
        "eval_count": 12,
        "logprobs": [
            {"token": head, "logprob": 0.0},
            {
                "token": verdict,
                "logprob": math.log(p_supported if verdict == "supported" else 1 - p_supported),
                "top_logprobs": [
                    {"token": "supported", "logprob": math.log(p_supported)},
                    {"token": "unsupported", "logprob": math.log(1 - p_supported)},
                    {"token": "No", "logprob": -1.0},  # outside the allowed verdicts
                ],
            },
            {"token": '"}', "logprob": 0.0},
        ],
    }


def _ollama(sent, *, share=1.0):
    """A fake Ollama server: answers every question "supported" at 0.99 and reports the model
    with ``share`` of it in GPU memory."""
    import httpx

    def handler(request):
        if request.url.path == "/api/ps":
            size = 1000
            model = {"name": "qwen3.5:4b-text", "size": size, "size_vram": int(size * share)}
            return httpx.Response(200, json={"models": [model]})
        sent.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/generate" or not sent[-1][1]["messages"]:
            return httpx.Response(200, json={"done": True})  # an unload or a load
        return httpx.Response(200, json=_verdict_payload("supported", 0.99))

    return httpx.MockTransport(handler)


def test_the_local_checker_asks_greedily_on_the_gpu_and_reads_p_supported():
    from apt_classifier.knowledge_store.check import LocalChecker

    sent = []
    checker = LocalChecker(url="http://local.test", transport=_ollama(sent))
    result = checker.check("Claim: APT28 itself used X-Agent.")
    assert (result.verdict, round(result.p_supported, 3)) == ("supported", 0.99)
    body = sent[0][1]
    options = body["options"]
    assert (options["num_gpu"], options["temperature"], body["think"]) == (999, 0.0, False)
    assert body["format"]["required"] == ["reason", "verdict"]
    assert body["logprobs"] is True


def test_the_local_checker_stops_unless_all_of_the_model_is_in_gpu_memory():
    import pytest

    from apt_classifier.knowledge_store.check import LocalChecker

    sent = []
    full = LocalChecker(url="http://local.test", transport=_ollama(sent))
    full.load()
    assert sent[0] == ("/api/chat", {**sent[0][1], "messages": []})  # loaded, nothing asked
    assert sent[0][1]["options"]["num_gpu"] == 999
    split = LocalChecker(url="http://local.test", transport=_ollama([], share=0.8))
    with pytest.raises(RuntimeError, match="80% in GPU memory"):
        split.load()
    with pytest.raises(RuntimeError, match="nothing runs on the CPU"):
        split.check("Claim: APT28 itself used X-Agent.")  # the answer is never returned


def test_the_local_checker_reloads_the_model_to_free_the_servers_memory():
    from apt_classifier.knowledge_store.check import LocalChecker

    sent = []
    checker = LocalChecker(url="http://local.test", reload_every=2, transport=_ollama(sent))
    for _ in range(3):
        checker.check("Claim: APT28 itself used X-Agent.")
    paths = [(path, bool(body.get("messages"))) for path, body in sent]
    assert paths == [
        ("/api/chat", True),
        ("/api/chat", True),
        ("/api/generate", False),  # unload
        ("/api/chat", False),  # load
        ("/api/chat", True),
    ]
    assert sent[2][1]["keep_alive"] == 0


def test_unclear_links_count_once_the_model_supports_them(tmp_path, monkeypatch):
    from apt_classifier.knowledge_store.check import Verdict

    words = tmp_path / "words"
    words.write_text("\n".join(ORDINARY), encoding="utf-8")
    monkeypatch.setattr(fresh, "WORDS", words)
    store = KnowledgeStore(tmp_path / "store")
    store.add_posts([{"key": "v.test/p", "source": "v", "url": "https://v.test/p"}])
    sentences = [
        "APT28 and Lazarus both used X-Agent in the intrusion.",
        *(f"The operators renamed file {n} on the host." for n in range(16)),
    ]
    store.record_fetch(
        "v.test/p",
        "v",
        FetchResult(
            "fetched", http_status=200, content=_html("A report", "2024-05-01", sentences).encode()
        ),
        at="2026-11-03T10:00:00+00:00",
    )
    store.parse()
    store.finalise()
    misp_dir = _misp_dir(tmp_path)
    cli.extract_facts(store, kb=_kb(), misp={"values": []}, misp_dir=misp_dir, workers=1)
    assert store.links_view() == []  # two actors in one sentence: unclear

    class Fake:
        model = "fake-model"
        asked: list[str] = []

        def check(self, message):
            self.asked.append(message)
            p = 0.99 if "Actor: APT28" in message else 0.6
            return Verdict("supported", p, "reason", 10)

    summary = cli.check_links(
        store, kb=_kb(), misp={"values": []}, misp_dir=misp_dir, workers=1, checker=Fake()
    )
    assert (summary["checked"], summary["confirmed"]) == (2, 1)
    assert "Text:\n[1] APT28 and Lazarus both used X-Agent in the intrusion." in Fake.asked[0]
    pairs = store.links_view(model="fake-model")
    assert [(p["actor"], p["software_id"], p["rules"]) for p in pairs] == [
        ("G0007", "S0001", ["sentence+model"])
    ]
    assert store.links_view(model=None) == []  # the model's verdicts left out
    assert store.links_view(date(2024, 5, 1), model="fake-model") == []  # dated as the post
    # answered links are not asked again
    again = cli.check_links(
        store, kb=_kb(), misp={"values": []}, misp_dir=misp_dir, workers=1, checker=Fake()
    )
    assert again["checked"] == 0

    # a throttled GPU: the run stops instead of crawling, and the next run resumes
    store.db.execute("DELETE FROM link_checks")
    clock = [0.0]

    class Slow(Fake):
        def check(self, message):
            clock[0] += 60.0
            return super().check(message)

    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cli, "SLOW_IN_A_ROW", 1)
    slow = cli.check_links(
        store, kb=_kb(), misp={"values": []}, misp_dir=misp_dir, workers=1, checker=Slow()
    )
    assert slow["checked"] == 1 and "throttled GPU" in slow["stopped"]
    assert len(store.pending_checks("fake-model", cli.PROMPT_VERSION)) == 1


def test_the_members_artifact_reads_the_store_as_of_each_case(tmp_path, monkeypatch):
    from apt_classifier.knowledge_store.members import Case, StoreMembers

    words = tmp_path / "words"
    words.write_text("\n".join(ORDINARY), encoding="utf-8")
    monkeypatch.setattr(fresh, "WORDS", words)
    store = KnowledgeStore(tmp_path / "store")
    posts = [("securelist", "2024-05-01"), ("kaspersky_ics", "2024-06-01"), ("v", "2024-07-01")]
    store.add_posts(
        [
            {"key": f"{source}.test/p", "source": source, "url": f"https://{source}.test/p"}
            for source, _ in posts
        ]
    )
    for number, (source, published) in enumerate(posts):
        sentences = [
            "APT28 deployed X-Agent against ministries.",
            "APT28 used the HOOKEDGE backdoor for persistence.",
            "APT28 sent spearphishing mail to diplomats.",
            *(f"The operators renamed file {number}-{n} on the host." for n in range(14)),
        ]
        store.record_fetch(
            f"{source}.test/p",
            source,
            FetchResult(
                "fetched", http_status=200, content=_html("APT28", published, sentences).encode()
            ),
            at="2026-11-03T10:00:00+00:00",
        )
    store.parse()
    store.finalise()
    kb = _kb()
    cli.extract_facts(store, kb=kb, misp={"values": []}, misp_dir=_misp_dir(tmp_path), workers=1)
    metadata = cli.build_members(store, kb=kb, misp={"values": []})
    assert metadata["counts"]["labelled_posts"] == 3
    members = StoreMembers.load(store.root / "store_members.json.gz", kb)
    assert members.masking_names == ("hookedge",)
    assert members.text_rows and all("APT28" not in row["text"] for row in members.text_rows)
    # the store text reads only the posts available before the case
    assert members.text_scores(Case(as_of="2024-05-01"), "spearphishing mail to diplomats") == {}
    assert set(members.text_scores(Case(as_of="2024-08-01"), "spearphishing to diplomats")) == {
        "G0007"
    }
    # the name a post introduces links APT28; two publishers only from the third post on
    text = "They used HOOKEDGE."
    scores, naming, _ = members.software_scores(Case(as_of="2024-06-15"), [], text, kb)
    assert set(scores) == {"G0007"} and naming == frozenset()
    scores, naming, _ = members.software_scores(Case(as_of="2024-08-01"), [], text, kb)
    assert set(scores) == {"G0007"} and naming == {"G0007"}
    # a case's own post and its copies are hidden from it
    own = members.hidden_for(keys=["v.test/p"])
    scores, naming, _ = members.software_scores(Case(as_of="2024-08-01", hidden=own), [], text, kb)
    assert naming == frozenset()
