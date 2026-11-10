"""Store text as a ranking-only lead: the registered rule and its offline scoring.

Registration ``docs/preregistration/store_text_lead_rule_2026-11-10.json`` (design doc
``docs/KNOWLEDGE_STORE_DESIGN.md``, "The lead rule"). The dated knowledge store's text member
(``apt_classifier/knowledge_store/members.py``: one TF-IDF document per ATT&CK group from the
posts readable before the report's date) is not fused here and reaches no combiner. The rule
reads it beside a finished product run:

    on a report the product does not name, store text's first choice leads the ranking when its
    margin (its first score minus its second) is at least ``THRESHOLD``; otherwise the product's
    leader stays. A name the product gives is never changed and the rule never names.

The product run aside (``scripts/attribution_benchmark.py`` with the store members off), four
steps, all on the CPU:

* ``store-text``: store text alone for every report of a set (about 10 s per report and about
  1.1 GB of memory), at the behaviour and the strict reading, with the whole store and with every
  post of the report's own publisher hidden. The reports are loaded, cut and masked as the
  benchmark does it.
* ``score``: one product artifact and the store-text file -> the product against the product with
  the rule (right group first, top five, per group, open set; the names are the product's own).
  Inputs that do not belong together are refused; ``--frozen`` adds the registered checks of the
  decisive set (its report files, the registered store files, product arm, threshold and this
  version of the script).
* ``verdict``: the registered reading of the behaviour and the strict score files.
* ``threshold``: how ``THRESHOLD`` follows from the store-text file of the store training posts
  (information only: the registered value is the constant below).

    PYTHONPATH=. .venv/bin/python scripts/store_text_lead.py store-text --reports R.jsonl
        [--open-set O.jsonl] --output S.json
    PYTHONPATH=. .venv/bin/python scripts/store_text_lead.py score PRODUCT.json S.json
        [--frozen docs/preregistration/knowledge_store_decisive_2026-11-04.json] [--output F.json]
    PYTHONPATH=. .venv/bin/python scripts/store_text_lead.py verdict BEHAVIOR.json STRICT.json
    PYTHONPATH=. .venv/bin/python scripts/store_text_lead.py threshold S.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Registered 2026-11-10: the smallest margin in the top half of the margins of the store training
# posts published before 2024-01-01 (known and open set, whole store, behaviour reading; 214 posts
# with a margin), rounded to four decimals. One threshold for every reading and arm.
THRESHOLD = 0.0131
THRESHOLD_POOL_BEFORE = "2024-01-01"
THRESHOLD_SHARE = 0.5
THRESHOLD_READING = "behavior"
THRESHOLD_ARM = "whole_store"

# The registered run (``score --frozen``): the decisive set's registration, the store files its
# reports read (the whole store gives the masking names; every decisive report is dated before
# 2026-03-01 and reads the before_2025 copy) and the product arm (the defaults, local pipeline,
# technique head on).
REGISTERED_SET = "knowledge_store_decisive_2026-11-04"
REGISTERED_STORE = {"whole": "da1da80adca9c75b1331c947", "before_2025": "26f77c2002c00de385d34dd1"}
REGISTERED_PIN = "before_2025"
REGISTERED_ACCEPTANCE = "active + technique head model"
LOCAL_OVERRIDES = {"cyberre_weight": 0.0, "technique_mapping_preset": "local_only"}

NAMED = ("known_group_hypothesis", "known_cluster_hypothesis")
CLUSTER = "known_cluster_hypothesis"
ARMS = ("whole_store", "publisher_hidden")
READINGS = ("behavior", "strict")
SETS = ("known", "open_set")
STORE_WEIGHTS = ("store_text", "store_software")
# the registered reading: gains outnumber losses at this one-sided level (exact sign test)
SIGN_LEVEL = 0.05
SEED = 7
RESAMPLES = 10_000
TOP = 10
SCHEMA = 1
KIND = "store_text_alone"
SCORE_KIND = "store_text_lead_score"
REPO_ROOT = Path(__file__).resolve().parent.parent


# --- store text alone ---------------------------------------------------------------------------


def ranked(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    """Groups by falling score; equal scores in the order of the group ids."""
    return sorted(
        ((str(group), float(value)) for group, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def margin_of(order: Sequence[tuple[str, float]]) -> float | None:
    """The first score minus the second; none with fewer than two scored groups."""
    return order[0][1] - order[1][1] if len(order) > 1 else None


def arm_entry(scores: Mapping[str, float], hidden_posts: int) -> dict[str, Any]:
    order = ranked(scores)
    return {
        "first": order[0][0] if order else None,
        "margin": margin_of(order),
        "scored_groups": len(order),
        "top": [[group, value] for group, value in order[:TOP]],
        # posts the case could read by date and does not (its own post and copies; with the
        # publisher hidden also the publisher's other posts)
        "hidden_posts": hidden_posts,
    }


def publisher_of(source: str, publishers: Mapping[str, str]) -> str:
    """Sources of one publisher count as one (``knowledge_store.sources.PUBLISHERS``)."""
    return publishers.get(source) or source


def hidden_readable(members: Any, case: Any) -> int:
    """How many of the store's posts the case hides although their date would let it read them."""
    return sum(
        1
        for key in case.hidden
        if key in members.posts
        and (case.as_of is None or str(members.posts[key]["available_from"]) < case.as_of)
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sources(*paths: Path | None) -> dict[str, dict[str, str]]:
    """``report_id -> source, published`` from the report files themselves."""
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        if path is None:
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            report_id = str(row.get("report_id") or "")
            if not report_id:
                raise ValueError(f"{path}: a report without report_id")
            if report_id in out:
                raise ValueError(f"{path}: report {report_id} is listed twice")
            out[report_id] = {
                "source": str(row.get("source") or ""),
                "published": str(row.get("published") or "")[:10],
            }
    return out


def store_text_rows(
    items: Iterable[tuple[str, str, Any]],
    sources: Mapping[str, Mapping[str, str]],
    state: Any,
    kb: Any,
    *,
    no_store: str,
    publishers: Mapping[str, str],
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """One row per report and reading: store text alone, whole store and own publisher hidden.

    ``items`` are ``(set, reading, fold)``. Each report reads the store as the benchmark sets it
    (``StoreState.set_case``: the posts dated before the report, its own post and copies hidden,
    the name lists of its date); an undated report reads nothing. A report's readings share one
    cut of the store per arm, so they are scored together. The narrative is the report's
    statements at the reading joined by spaces, as the store text member reads a case.
    """
    by_report: dict[str, list[tuple[str, str, Any]]] = {}
    for set_name, reading, fold in items:
        if not fold.report_id:
            raise ValueError("a report without report_id")
        by_report.setdefault(str(fold.report_id), []).append((set_name, reading, fold))
    rows: list[dict[str, Any]] = []
    for number, (report_id, group) in enumerate(by_report.items(), 1):
        readings = [reading for _, reading, _ in group]
        if len(set(readings)) != len(readings):
            raise ValueError(f"report {report_id} is listed twice at one reading")
        first = group[0][2]
        as_of = first.as_of or no_store
        common = {
            "as_of": as_of,
            "keys": first.store_keys,
            "statements": first.store_statements,
            "pinned": True,
        }
        whole = state.set_case(**common)
        members = state.current
        if members is None:
            raise ValueError("no store members are loaded")
        pin = state.pin
        source = str((sources.get(report_id) or {}).get("source") or "")
        if not source:
            own = next((key for key in first.store_keys if key in members.posts), None)
            source = str(members.posts[own]["source"]) if own is not None else ""
        if not source:
            raise ValueError(
                f"report {report_id}: no source, so its publisher's posts cannot be hidden"
            )
        publisher = publisher_of(source, publishers)
        theirs = [
            key
            for key, post in members.posts.items()
            if publisher_of(str(post["source"]), publishers) == publisher
        ]
        hidden = state.set_case(**common, hidden=theirs)
        cases = {"whole_store": whole, "publisher_hidden": hidden}
        for set_name, reading, fold in group:
            narrative = " ".join(str(text) for text in fold.observations)
            row: dict[str, Any] = {
                "report_id": report_id,
                "reading": reading,
                "set": set_name,
                "group_id": fold.group_id,
                "accept": sorted(fold.accept),
                "published": str((sources.get(report_id) or {}).get("published") or ""),
                "as_of": as_of,
                "pin": pin,
                "publisher": publisher,
                "statements": len(fold.observations),
            }
            for arm in ARMS:
                row[arm] = arm_entry(
                    members.text_scores(cases[arm], narrative, kb=kb),
                    hidden_readable(members, cases[arm]),
                )
            rows.append(row)
        state.clear()
        if progress is not None:
            progress(number, len(by_report))
    return rows


def load_reading(
    bench: Any,
    anonymizer_cls: Any,
    kb: Any,
    reports: Path,
    open_set: Path | None,
    reading: str,
    store_names: Sequence[str],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """The known and open-set reports at one reading, as ``attribution_benchmark.py`` loads them
    for ``--report-observations 0`` and ``--anonymize <reading>`` (attribution prose dropped, then
    the names masked; at the behaviour reading also every software name the store knows, the
    benchmark's ``--mask-store-names``). The third value holds the masking totals of the known
    and the open-set reports, which the benchmark writes into its artifact: the score step
    compares them, so both runs are known to have read the same masked text."""
    dropper = anonymizer_cls(kb, reading)
    loaded = bench.load_report_folds(
        reports, max_observations=None, anonymizer=dropper, sample_seed=None, kb=kb
    )
    known = [fold for fold in loaded if fold.group_id is not None]
    open_folds = [fold for fold in loaded if fold.group_id is None]
    if open_set is not None:
        open_folds += bench.load_report_folds(
            open_set, max_observations=None, anonymizer=dropper, sample_seed=None, kb=kb
        )
    extra = tuple(store_names) if reading == "behavior" else ()
    known, _, known_totals = bench.anonymize_report_folds(
        known, kb, level=reading, extra_names=extra
    )
    open_totals = None
    if open_folds:
        open_folds, _, open_totals = bench.anonymize_report_folds(
            open_folds, kb, level=reading, extra_names=extra
        )
    return known, open_folds, {"known": known_totals, "open_set": open_totals}


def run_store_text(
    reports: Path, open_set: Path | None, readings: Sequence[str] = READINGS
) -> dict[str, Any]:
    """Store text alone for a report set, at every reading and in both arms."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.evaluation import source_fingerprint
    from apt_classifier.knowledge_store.members import StoreMembers, StoreState, members_path
    from apt_classifier.knowledge_store.pins import PINS
    from apt_classifier.knowledge_store.sources import PUBLISHERS
    from scripts import attribution_benchmark as bench
    from scripts.report_anonymizer import ReportAnonymizer

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    members: dict[str | None, Any] = {}
    for pin in (None, *sorted(PINS)):
        path = members_path(settings, pin)
        loaded = StoreMembers.load(path, kb)
        if loaded is None:
            raise ValueError(f"{path}: no store members (scripts/knowledge_store.py members)")
        compatible, reason = loaded.compatible_with(kb)
        if not compatible:
            raise ValueError(f"{path}: {reason}")
        members[pin] = loaded
    # the benchmark masks with the whole store's names (--mask-store-names)
    store_names = tuple(members[None].masking_names)
    sources = read_sources(reports, open_set)
    items: list[tuple[str, str, Any]] = []
    masking: dict[str, Any] = {}
    for reading in readings:
        known, open_folds, masking[reading] = load_reading(
            bench, ReportAnonymizer, kb, reports, open_set, reading, store_names
        )
        items += [("known", reading, fold) for fold in known]
        items += [("open_set", reading, fold) for fold in open_folds]

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  store text {done}/{total} reports", file=sys.stderr, flush=True)

    rows = store_text_rows(
        items,
        sources,
        StoreState(members),
        kb,
        no_store=bench.NO_STORE,
        publishers=PUBLISHERS,
        progress=progress,
    )
    return {
        "kind": KIND,
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # this file's own hash, to compare with the one in the registration
        "script_sha256": file_sha256(Path(__file__)),
        "inputs": {
            str(path): file_sha256(path) for path in (reports, open_set) if path is not None
        },
        "attack_version": kb.version,
        "store": {pin or "whole": loaded.fingerprint for pin, loaded in members.items()},
        "masking_names": len(store_names),
        # what the masking did per reading and the hash of its rules, as the benchmark records
        # them for its own run (``leak_control`` and ``fingerprint_inputs.report_rules``)
        "masking": json.loads(json.dumps(masking)),
        "report_rules": source_fingerprint((REPO_ROOT / "scripts" / "report_anonymizer.py",)),
        "readings": list(readings),
        "arms": list(ARMS),
        "rows": rows,
    }


# --- the rule -----------------------------------------------------------------------------------


def truth_of(case: Mapping[str, Any]) -> set[str]:
    """The groups that count as right: the label and the report's accepted alternatives."""
    return {str(group) for group in (case.get("group_id"), *(case.get("accept") or ())) if group}


def top_five(case: Mapping[str, Any], lead: str | None = None) -> list[str]:
    """The product's first five groups; a lead moves to the front and the rest keep their order."""
    order = [str(item["group_id"]) for item in case.get("topk") or ()]
    if lead is not None:
        order = [lead, *(group for group in order if group != lead)]
    return order[:5]


def product_leader(case: Mapping[str, Any]) -> str | None:
    """The product's first choice. With too little evidence the product reports no leader although
    it ranks the groups: its leader is then the first group of that ranking, as the benchmark
    counts it; none when it ranked nothing."""
    leading = case.get("leading")
    if leading:
        return str(leading)
    return next(iter(top_five(case)), None)


def decide(case: Mapping[str, Any], entry: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    """The registered rule for one report: who leads, and whether store text made it so."""
    named = case.get("stage") in NAMED
    first, margin = entry.get("first"), entry.get("margin")
    leads = not named and first is not None and margin is not None and margin >= threshold
    return {"named": named, "leads": leads, "leader": first if leads else product_leader(case)}


def name_right(case: Mapping[str, Any]) -> bool:
    """A product name is right when it is the truth; a cluster when the truth is a member."""
    truth = truth_of(case)
    if case.get("stage") == CLUSTER:
        return bool(truth & {str(group) for group in case.get("named_cluster") or ()}) or (
            case.get("leading") in truth
        )
    return case.get("leading") in truth


def macro(groups: Sequence[str], flags: Sequence[int]) -> float | None:
    """The share first, averaged over the groups (each group counts once)."""
    per_group: dict[str, list[int]] = defaultdict(list)
    for group, flag in zip(groups, flags, strict=True):
        per_group[group].append(flag)
    if not per_group:
        return None
    return round(sum(sum(v) / len(v) for v in per_group.values()) / len(per_group), 4)


def mcnemar(gained: int, lost: int) -> float:
    """Exact two-sided McNemar p for the reports that changed."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n), 4)


def sign_test(gained: int, lost: int) -> float:
    """Exact one-sided p of at least ``gained`` gains among the changed reports by chance."""
    n = gained + lost
    if n == 0:
        return 1.0
    return round(sum(math.comb(n, i) for i in range(gained, n + 1)) / 2**n, 4)


def beyond_chance(gained: int, lost: int) -> bool:
    """Gains outnumber losses at ``SIGN_LEVEL``: the sign test compared exactly, not rounded."""
    n = gained + lost
    if gained <= lost:
        return False
    tail = sum(math.comb(n, i) for i in range(gained, n + 1))
    return tail * 10_000 <= round(SIGN_LEVEL * 10_000) * 2**n


def bootstrap_diff(after: Sequence[int], before: Sequence[int]) -> list[float] | None:
    """95% interval of the paired difference in the share first (report resamples, fixed seed)."""
    n = len(after)
    if n == 0:
        return None
    rng = random.Random(SEED)
    diffs = []
    for _ in range(RESAMPLES):
        sample = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(after[i] - before[i] for i in sample) / n)
    diffs.sort()
    return [round(diffs[int(0.025 * RESAMPLES)], 4), round(diffs[int(0.975 * RESAMPLES)], 4)]


# --- scoring ------------------------------------------------------------------------------------


def canonical(value: Any) -> Any:
    """A JSON value with its lists of plain values in one order, so the masking totals of two
    runs compare equal whatever order they were counted in."""
    if isinstance(value, Mapping):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        items = [canonical(item) for item in value]
        if all(not isinstance(item, dict | list) for item in items):
            return sorted(items, key=repr)
        return items
    return value


def check_inputs(
    product: Mapping[str, Any],
    store: Mapping[str, Any],
    *,
    threshold: float,
    frozen: Mapping[str, Any] | None = None,
) -> str:
    """The reading the two files share. Raises ``ValueError`` when they do not belong together:
    the rule is scored only on a complete product run with the store off and the combiner active
    over the same report files, masked the same way, on the same ATT&CK release. ``frozen`` (the
    decisive set's registration) adds the registered run's own conditions."""
    if store.get("kind") != KIND or store.get("schema") != SCHEMA:
        raise ValueError("the second file is not a store-text file of this script")
    if list(store.get("arms") or ()) != list(ARMS):
        raise ValueError("the store-text file does not hold both arms")
    parameters = product.get("parameters") or {}
    reading = parameters.get("anonymization")
    if reading not in (store.get("readings") or ()):
        raise ValueError(f"the product run's reading ({reading}) is not in the store-text file")
    for key in ("truncated_at", "sample", "limit", "report_observations", "report_sample_seed"):
        if parameters.get(key) is not None:
            raise ValueError(
                f"the product run does not read every report in full ({key} = {parameters[key]})"
            )
    inputs = product.get("fingerprint_inputs") or {}
    weights = inputs.get("weights") or {}
    voting = [name for name in STORE_WEIGHTS if float(weights.get(name) or 0.0) > 0.0]
    if voting:
        raise ValueError(
            f"the product run has store members voting ({', '.join(voting)}); "
            "the rule is scored against a run with the store off"
        )
    if inputs.get("combiner") != "active":
        raise ValueError(
            f"the product run's combiner is {inputs.get('combiner')!r}: its first five are the "
            "product's own order only with the combiner active"
        )
    masked = inputs.get("mask_store_names")
    if reading == "behavior":
        if not masked or int(masked) != int(store.get("masking_names") or 0):
            raise ValueError(
                f"store names masked: {masked} in the product run, "
                f"{store.get('masking_names')} in the store-text file"
            )
    elif masked:
        raise ValueError("the product run masks store names outside the behaviour reading")
    if not product.get("attack_version") or product.get("attack_version") != store.get(
        "attack_version"
    ):
        raise ValueError(
            f"ATT&CK {product.get('attack_version')} in the product run, "
            f"{store.get('attack_version')} in the store-text file"
        )
    theirs = set((inputs.get("report_inputs") or {}).values())
    ours = set((store.get("inputs") or {}).values())
    if not theirs or theirs != ours:
        raise ValueError("the product run and the store-text file read different report files")
    # the same masked text in both runs: the hash of the masking rules and what the masking did
    # (a store-text file from before these were recorded is compared only by the checks above)
    totals, rules = (store.get("masking") or {}).get(reading), store.get("report_rules")
    if frozen is not None and (totals is None or not rules):
        raise ValueError("the store-text file does not record how its reports were masked")
    if rules and rules != inputs.get("report_rules"):
        raise ValueError(
            "the two runs masked the reports with different rules (report_anonymizer.py)"
        )
    if totals is not None:
        leak = product.get("leak_control")
        leak = leak if isinstance(leak, Mapping) else {}
        for name, key in (
            ("known", "actor_name_replacement"),
            ("open_set", "open_set_actor_name_replacement"),
        ):
            if canonical(totals.get(name)) != canonical(leak.get(key)):
                raise ValueError(
                    f"the two runs masked the {name.replace('_', '-')} reports differently "
                    "(their masking totals differ)"
                )
    if frozen is not None:
        if frozen.get("id") != REGISTERED_SET:
            raise ValueError(f"--frozen is not the registration {REGISTERED_SET}")
        files = (frozen.get("frozen") or {}).get("files_sha256") or {}
        wanted = {files.get("reports.jsonl"), files.get("open_set.jsonl")}
        if None in wanted or len(wanted) != 2:
            raise ValueError("the registration holds no frozen hashes of the report files")
        if ours != wanted:
            raise ValueError("the report files are not the frozen set of the registration")
        if threshold != THRESHOLD:
            raise ValueError(f"the registered threshold is {THRESHOLD}")
        if parameters.get("pipeline") != "local":
            raise ValueError("the registered product arm is the local pipeline")
        if parameters.get("settings_overrides") != LOCAL_OVERRIDES:
            raise ValueError(
                "the registered product arm is the local pipeline at the product's defaults "
                f"(settings overrides {parameters.get('settings_overrides')})"
            )
        if inputs.get("acceptance_model") != REGISTERED_ACCEPTANCE:
            raise ValueError(
                "the registered product arm runs with the technique head "
                f"(acceptance model: {inputs.get('acceptance_model')})"
            )
        if store.get("script_sha256") != file_sha256(Path(__file__)):
            raise ValueError("the store-text file was written by another version of this script")
        read = store.get("store") or {}
        for name, fingerprint in REGISTERED_STORE.items():
            if read.get(name) != fingerprint:
                raise ValueError(
                    f"the store-text file read another {name} store file than the registered one"
                )
        pins = {str(row.get("pin")) for row in store.get("rows") or ()}
        if pins != {REGISTERED_PIN}:
            raise ValueError(
                f"a report of the decisive set reads the {REGISTERED_PIN} copy of the store "
                f"(found {sorted(pins)})"
            )
    return str(reading)


def paired(
    product: Mapping[str, Any], store: Mapping[str, Any], reading: str
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Each product case with its store-text row; both files must hold the same reports."""
    if product.get("unknown"):
        raise ValueError("the product run holds fold cases; the rule is scored on report sets")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in store.get("rows") or ():
        if row.get("reading") != reading:
            continue
        key = (str(row.get("set")), str(row.get("report_id")))
        if key in rows:
            raise ValueError(f"report {key[1]} is twice in the store-text file")
        rows[key] = row
    out: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for set_name in SETS:
        cases = list(product.get(set_name) or ())
        ids = [str(case.get("report_id") or "") for case in cases]
        if "" in ids or len(set(ids)) != len(ids):
            raise ValueError(f"{set_name}: product cases without a report id or listed twice")
        ours = {report_id for name, report_id in rows if name == set_name}
        if set(ids) != ours:
            raise ValueError(
                f"{set_name}: the two files do not hold the same reports "
                f"(only in the store-text file: {sorted(ours - set(ids))[:5]}; "
                f"only in the product run: {sorted(set(ids) - ours)[:5]})"
            )
        pairs = []
        for case, report_id in zip(cases, ids, strict=True):
            row = rows[(set_name, report_id)]
            if case.get("group_id") != row.get("group_id"):
                raise ValueError(f"report {report_id}: the two files disagree on its group")
            if sorted(case.get("accept") or ()) != sorted(row.get("accept") or ()):
                raise ValueError(f"report {report_id}: the two files disagree on accepted groups")
            if case.get("observations") != row.get("statements"):
                raise ValueError(
                    f"report {report_id}: {case.get('observations')} statements in the product "
                    f"run, {row.get('statements')} in the store-text file"
                )
            for arm in ARMS:
                if not isinstance(row.get(arm), Mapping):
                    raise ValueError(f"report {report_id}: no {arm} entry in the store-text file")
            pairs.append((case, row))
        out[set_name] = pairs
    return out


def score_arm(
    known: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    open_set: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    arm: str,
    threshold: float,
) -> dict[str, Any]:
    before: list[int] = []
    after: list[int] = []
    top5_before = top5_after = 0
    cases: list[dict[str, Any]] = []
    for case, row in known:
        truth = truth_of(case)
        entry = row[arm]
        decision = decide(case, entry, threshold)
        leader = product_leader(case)
        was, now = leader in truth, decision["leader"] in truth
        before.append(int(was))
        after.append(int(now))
        top5_before += bool(truth & set(top_five(case)))
        top5_after += bool(
            truth & set(top_five(case, decision["leader"] if decision["leads"] else None))
        )
        cases.append(
            {
                "report_id": case.get("report_id"),
                "group_id": case.get("group_id"),
                "stage": case.get("stage"),
                "named": decision["named"],
                "product_leader": leader,
                "product_first": was,
                "store_first": entry.get("first"),
                "margin": entry.get("margin"),
                "leads": decision["leads"],
                "leader": decision["leader"],
                "first": now,
            }
        )
    gained = [c["report_id"] for c in cases if c["first"] and not c["product_first"]]
    lost = [c["report_id"] for c in cases if c["product_first"] and not c["first"]]
    leads = [c for c in cases if c["leads"]]
    open_cases = []
    for case, row in open_set:
        decision = decide(case, row[arm], threshold)
        open_cases.append(
            {
                "report_id": case.get("report_id"),
                "stage": case.get("stage"),
                "named": decision["named"],
                "product_leader": product_leader(case),
                "store_first": row[arm].get("first"),
                "margin": row[arm].get("margin"),
                "leads": decision["leads"],
                "leader": decision["leader"],
            }
        )
    n = len(cases)
    groups = [str(case.get("group_id")) for case, _ in known]
    return {
        "known": n,
        "not_named": sum(1 for c in cases if not c["named"]),
        "first_before": sum(before),
        "first_after": sum(after),
        "gained": gained,
        "lost": lost,
        "difference": round((sum(after) - sum(before)) / n, 4) if n else None,
        "bootstrap_95": bootstrap_diff(after, before),
        "mcnemar_p": mcnemar(len(gained), len(lost)),
        "sign_test_one_sided_p": sign_test(len(gained), len(lost)),
        "macro_first_before": macro(groups, before),
        "macro_first_after": macro(groups, after),
        "top5_before": int(top5_before),
        "top5_after": int(top5_after),
        "leads": len(leads),
        "leads_right": sum(1 for c in leads if c["first"]),
        "leads_wrong": sum(1 for c in leads if not c["first"]),
        "leader_changes": sum(1 for c in leads if c["leader"] != c["product_leader"]),
        "open_set": {
            "reports": len(open_cases),
            "not_named": sum(1 for c in open_cases if not c["named"]),
            "leads": sum(1 for c in open_cases if c["leads"]),
            "leader_changes": sum(
                1 for c in open_cases if c["leads"] and c["leader"] != c["product_leader"]
            ),
        },
        "cases": cases,
        "open_set_cases": open_cases,
    }


def score(
    product: Mapping[str, Any],
    store: Mapping[str, Any],
    *,
    threshold: float = THRESHOLD,
    frozen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The product against the product with the rule, for the reading of the product run."""
    reading = check_inputs(product, store, threshold=threshold, frozen=frozen)
    pairs = paired(product, store, reading)
    known, open_set = pairs["known"], pairs["open_set"]
    named = [case for case, _ in known if case.get("stage") in NAMED]
    right = sum(1 for case in named if name_right(case))
    return {
        "kind": SCORE_KIND,
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": file_sha256(Path(__file__)),
        "store_text_script_sha256": store.get("script_sha256"),
        "reading": reading,
        "threshold": threshold,
        "registered_threshold": threshold == THRESHOLD,
        "frozen_set_checked": frozen is not None,
        "inputs": {
            "reports_sha256": sorted((store.get("inputs") or {}).values()),
            "attack_version": store.get("attack_version"),
            "store": store.get("store"),
            "masking_names": store.get("masking_names") if reading == "behavior" else 0,
            # true when the masking rules and totals of the two runs were compared
            "masking_compared": (store.get("masking") or {}).get(reading) is not None,
            "product_pipeline_fingerprint": product.get("pipeline_fingerprint"),
        },
        "known": len(known),
        "groups": len({case.get("group_id") for case, _ in known}),
        "open_set": len(open_set),
        # the benchmark's own count of the right group first, beside ``first_before``
        "product_first_by_rank": sum(
            1 for case, _ in known if case.get("lenient_rank", case.get("rank")) == 1
        ),
        # known reports the product gives no leader for (too little evidence): their leader is
        # the first group of the product's ranking
        "known_without_product_leader": sum(1 for case, _ in known if not case.get("leading")),
        # the product's names; the rule changes none of them
        "names": {
            "known": {"named": len(named), "right": right, "wrong": len(named) - right},
            "open_set": {"named": sum(1 for case, _ in open_set if case.get("stage") in NAMED)},
        },
        # known reports whose publisher has no other readable post in the store: for them the
        # two arms are the same reading
        "known_without_other_publisher_posts": sum(
            1
            for _, row in known
            if row["publisher_hidden"]["hidden_posts"] == row["whole_store"]["hidden_posts"]
        ),
        "arms": {arm: score_arm(known, open_set, arm, threshold) for arm in ARMS},
    }


def verdict(behavior: Mapping[str, Any], strict: Mapping[str, Any]) -> dict[str, Any]:
    """The registered reading of the two score files of one report set: the rule is supported
    when, at the behaviour reading with the whole store, gains outnumber losses beyond chance
    (exact one-sided sign test at ``SIGN_LEVEL``) and no other reading or arm loses more reports
    than it gains."""
    for name, result, reading in (("first", behavior, "behavior"), ("second", strict, "strict")):
        if result.get("kind") != SCORE_KIND or result.get("reading") != reading:
            raise ValueError(f"the {name} file is not a score file at the {reading} reading")
        if result.get("threshold") != THRESHOLD:
            raise ValueError(f"the {name} file was not scored at the registered threshold")
    if behavior["inputs"]["reports_sha256"] != strict["inputs"]["reports_sha256"]:
        raise ValueError("the two score files are not of the same report set")
    for key in ("attack_version", "store"):
        if behavior["inputs"].get(key) != strict["inputs"].get(key):
            raise ValueError(f"the two score files differ in what they read ({key})")
    for key in ("script_sha256", "store_text_script_sha256"):
        if behavior.get(key) != strict.get(key):
            raise ValueError("the two score files come from different versions of the script")
    frozen_checked = bool(behavior.get("frozen_set_checked") and strict.get("frozen_set_checked"))
    if frozen_checked and behavior.get("script_sha256") != file_sha256(Path(__file__)):
        raise ValueError("the score files were written by another version of this script")
    primary = behavior["arms"]["whole_store"]
    gained, lost = len(primary["gained"]), len(primary["lost"])
    checks = {
        "behaviour, whole store: gains beyond chance": beyond_chance(gained, lost),
    }
    for label, result, arm in (
        ("behaviour, own publisher hidden", behavior, "publisher_hidden"),
        ("strict, whole store", strict, "whole_store"),
        ("strict, own publisher hidden", strict, "publisher_hidden"),
    ):
        checks[f"{label}: gained at least as many as lost"] = len(
            result["arms"][arm]["gained"]
        ) >= len(result["arms"][arm]["lost"])
    return {
        "supported": all(checks.values()),
        "checks": checks,
        "frozen_set_checked": frozen_checked,
        # what the two score files were made from, so this file can be read on its own
        "threshold": THRESHOLD,
        "script_sha256": behavior.get("script_sha256"),
        "store_text_script_sha256": behavior.get("store_text_script_sha256"),
        "inputs": {
            key: behavior["inputs"].get(key)
            for key in ("reports_sha256", "attack_version", "store")
        },
        "primary": {
            "known": primary["known"],
            "first_before": primary["first_before"],
            "first_after": primary["first_after"],
            "gained": gained,
            "lost": lost,
            "sign_test_one_sided_p": sign_test(gained, lost),
            "bootstrap_95": primary["bootstrap_95"],
        },
    }


# --- the threshold ------------------------------------------------------------------------------


def quantile_threshold(values: Iterable[float], share: float) -> float:
    """The smallest value among the top ``share`` of the values (of an odd number of values the
    middle one belongs to the top half)."""
    ordered = sorted(values, reverse=True)
    if not ordered:
        raise ValueError("no margins to take a threshold from")
    return ordered[max(1, math.ceil(share * len(ordered) - 1e-9)) - 1]


def threshold_pool(
    store: Mapping[str, Any],
    *,
    reading: str = THRESHOLD_READING,
    arm: str = THRESHOLD_ARM,
    before: str = THRESHOLD_POOL_BEFORE,
) -> list[float]:
    """The margins the threshold is taken from: posts published before the cut that have one."""
    return [
        float(row[arm]["margin"])
        for row in store.get("rows") or ()
        if row.get("reading") == reading
        and row.get("published")
        and str(row["published"]) < before
        and row[arm].get("margin") is not None
    ]


def derive_threshold(store: Mapping[str, Any]) -> dict[str, Any]:
    if store.get("kind") != KIND or store.get("schema") != SCHEMA:
        raise ValueError("not a store-text file of this script")
    out: dict[str, Any] = {"registered": THRESHOLD, "share": THRESHOLD_SHARE, "readings": {}}
    for reading in store.get("readings") or ():
        pool = threshold_pool(store, reading=reading)
        value = quantile_threshold(pool, THRESHOLD_SHARE) if pool else None
        out["readings"][reading] = {
            "posts": len(pool),
            "threshold": value,
            "rounded": round(value, 4) if value is not None else None,
        }
    return out


# --- command line -------------------------------------------------------------------------------


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _print_score(result: Mapping[str, Any]) -> None:
    names = result["names"]
    print(
        f"{result['reading']} reading, threshold {result['threshold']}: {result['known']} known "
        f"reports ({result['groups']} groups), {result['open_set']} open set; product names "
        f"{names['known']['named']} on known ({names['known']['right']} right, "
        f"{names['known']['wrong']} wrong), {names['open_set']['named']} on the open set "
        f"(unchanged by the rule)"
    )
    for name, arm in result["arms"].items():
        print(
            f"  {name:17s} first {arm['first_before']} -> {arm['first_after']} of {arm['known']} "
            f"(+{len(arm['gained'])} -{len(arm['lost'])}; interval {arm['bootstrap_95']}, "
            f"sign test p {arm['sign_test_one_sided_p']}) | per group "
            f"{arm['macro_first_before']} -> {arm['macro_first_after']} | top five "
            f"{arm['top5_before']} -> {arm['top5_after']} | leads {arm['leads']} "
            f"({arm['leads_right']} right) of {arm['not_named']} not named | open-set leads "
            f"{arm['open_set']['leads']} of {arm['open_set']['reports']}"
        )
    print(
        f"  known reports whose publisher has no other readable post: "
        f"{result['known_without_other_publisher_posts']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    alone = commands.add_parser("store-text", help="store text alone for a report set (CPU)")
    alone.add_argument("--reports", type=Path, required=True)
    alone.add_argument("--open-set", type=Path)
    alone.add_argument("--output", type=Path, required=True)
    scoring = commands.add_parser("score", help="the product against the product with the rule")
    scoring.add_argument("product", type=Path)
    scoring.add_argument("store_text", type=Path)
    scoring.add_argument("--frozen", type=Path, help="the decisive set's registration")
    scoring.add_argument("--output", type=Path)
    scoring.add_argument(
        "--threshold", type=float, default=THRESHOLD, help="development only (not with --frozen)"
    )
    reading = commands.add_parser("verdict", help="the registered reading of two score files")
    reading.add_argument("behavior", type=Path)
    reading.add_argument("strict", type=Path)
    reading.add_argument("--output", type=Path)
    derive = commands.add_parser("threshold", help="how the registered threshold was derived")
    derive.add_argument("store_text", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "store-text":
            result = run_store_text(args.reports, args.open_set)
            _write(args.output, result)
            print(f"{len(result['rows'])} rows -> {args.output}")
        elif args.command == "score":
            result = score(
                _read(args.product),
                _read(args.store_text),
                threshold=args.threshold,
                frozen=_read(args.frozen) if args.frozen else None,
            )
            _print_score(result)
            _write(args.output, result)
        elif args.command == "verdict":
            result = verdict(_read(args.behavior), _read(args.strict))
            for check, passed in result["checks"].items():
                print(f"  {'yes' if passed else 'no ':3s} {check}")
            print(
                f"the rule is {'supported' if result['supported'] else 'not supported'}"
                + ("" if result["frozen_set_checked"] else " (not the frozen decisive set)")
            )
            _write(args.output, result)
        else:
            result = derive_threshold(_read(args.store_text))
            for name, entry in result["readings"].items():
                print(
                    f"  {name}: {entry['posts']} posts before {THRESHOLD_POOL_BEFORE} with a "
                    f"margin, top-half threshold {entry['threshold']} ({entry['rounded']})"
                )
            print(f"registered: {THRESHOLD} (the {THRESHOLD_READING} reading)")
    except (ValueError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
