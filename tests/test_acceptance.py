"""Tests for the calibrated technique-acceptance model."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from apt_classifier.acceptance import (
    CONTRACT_1_KEYS,
    CONTRACT_RECORD_KEYS,
    EXEMPLAR_CONTRACT_KEYS,
    FEATURE_NAMES,
    FEATURE_NAMES_V4,
    FEATURE_NAMES_V5,
    FEATURE_SCHEMA_VERSION,
    LEGACY_CONTRACT,
    PRIOR_FEATURE_NAMES,
    RETRIEVAL_CONTRACT_KEYS,
    AcceptanceModel,
    candidate_features,
    choose_threshold,
    choose_threshold_runtime,
    compute_technique_priors,
    contract_mismatches,
    feature_vector,
    legacy_contract,
    load_acceptance_model,
    prior_features,
)
from apt_classifier.evidence.models import TechniqueCandidate, TechniqueRetrievalEvidence


def _candidate(
    tid: str,
    rank: int,
    fused: float,
    dense: float | None,
    field: str = "description",
    lexical_rank: int | None = None,
    cue: bool = False,
) -> TechniqueCandidate:
    evidence = []
    if dense is not None:
        evidence.append(
            TechniqueRetrievalEvidence(source="dense", field=field, score=dense, rank=rank)
        )
    if lexical_rank is not None:
        evidence.append(
            TechniqueRetrievalEvidence(source="lexical", field="name", score=2.0, rank=lexical_rank)
        )
    if cue:
        evidence.append(TechniqueRetrievalEvidence(source="explicit_cue", score=0.85, rank=1))
    return TechniqueCandidate(
        observation_id="obs-1",
        technique_id=tid,
        technique_name=tid,
        fused_score=fused,
        fused_rank=rank,
        evidence=evidence,
    )


def test_candidate_features_cover_schema_and_margins():
    top = _candidate("T1059.001", 1, 0.040, 0.86, lexical_rank=1, cue=True)
    second = _candidate("T1003.001", 2, 0.017, 0.71, field="procedure")
    third = _candidate("T1021.002", 3, 0.016, None, lexical_rank=7)
    siblings = [second, third, top]
    features = candidate_features(second, siblings, benign_context=False, sparse_context=True)
    # candidate features plus the per-technique priors (looked up by id) cover the schema (4:
    # the technique head's features too; a schema-3 model reads its own prefix)
    assert set(features) | set(PRIOR_FEATURE_NAMES) == set(FEATURE_NAMES_V5)
    assert set(FEATURE_NAMES) < set(FEATURE_NAMES_V4) < set(FEATURE_NAMES_V5)
    assert features["is_rank_one"] == 0.0
    assert features["fused_margin_to_top"] == pytest.approx(0.023)
    assert features["fused_margin_to_next"] == pytest.approx(0.001)
    assert features["dense_field_procedure"] == 1.0
    assert features["dense_margin"] == pytest.approx(0.15)
    assert features["sparse_context"] == 1.0
    assert features["candidate_count"] == 3.0
    top_features = candidate_features(top, siblings, benign_context=True, sparse_context=False)
    assert top_features["explicit_cue"] == 1.0
    assert top_features["lexical_rank_le3"] == 1.0
    assert top_features["benign_context"] == 1.0
    assert feature_vector(top_features).shape == (len(FEATURE_NAMES),)


def _synthetic_rows(seed: int = 3, n: int = 400):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n):
        dense = float(rng.uniform(0.3, 0.95))
        rank_one = float(rng.integers(0, 2))
        positive = int(dense + 0.2 * rank_one + rng.normal(0, 0.08) > 0.75)
        row = dict.fromkeys(FEATURE_NAMES, 0.0)
        row.update({"dense_score": dense, "is_rank_one": rank_one, "dense_present": 1.0})
        rows.append(row)
        labels.append(positive)
    return rows, labels


def test_fit_learns_separable_signal_and_roundtrips(tmp_path):
    rows, labels = _synthetic_rows()
    model = AcceptanceModel.fit(rows, labels, attack_version="19.2")
    weights = dict(zip(FEATURE_NAMES, model.weights.tolist(), strict=True))
    assert weights["dense_score"] > 0.5
    assert weights["is_rank_one"] > 0.0
    probs = [model.probability(row) for row in rows]
    accuracy = np.mean([(p >= 0.5) == bool(y) for p, y in zip(probs, labels, strict=True)])
    assert accuracy > 0.85
    path = model.save(tmp_path / "model.json")
    loaded = AcceptanceModel.load(path)
    assert loaded is not None
    assert loaded.probability(rows[0]) == pytest.approx(model.probability(rows[0]))
    assert loaded.compatible_with("19.2") == (True, "ok")
    ok, reason = loaded.compatible_with("18.0")
    assert not ok and "ATT&CK" in reason
    loaded.feature_schema_version = FEATURE_SCHEMA_VERSION + 1
    assert loaded.compatible_with("19.2")[0] is False
    assert AcceptanceModel.load(tmp_path / "missing.json") is None


def test_choose_threshold_respects_precision_floor_and_unrecoverable_labels():
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.2, 0.1]
    labels = [1, 1, 0, 1, 0, 0]
    case_ids = ["a", "a", "b", "b", "c", "c"]
    # case "a" has a third expected label that never became a candidate
    expected_counts = {"a": 3, "b": 1, "c": 0}
    choice = choose_threshold(probabilities, labels, case_ids, expected_counts, min_precision=0.9)
    assert choice["precision_floor_met"] is True
    assert choice["threshold"] > 0.7  # excludes the 0.7 false positive
    assert choice["recall"] == pytest.approx(2 / 4)
    relaxed = choose_threshold(probabilities, labels, case_ids, expected_counts, min_precision=0.5)
    assert relaxed["threshold"] <= 0.6
    assert relaxed["recall"] == pytest.approx(3 / 4)
    impossible = choose_threshold(
        probabilities, labels, case_ids, expected_counts, min_precision=1.1
    )
    assert impossible["precision_floor_met"] is False


def test_from_dict_rejects_foreign_artifacts():
    with pytest.raises(ValueError):
        AcceptanceModel.from_dict({"kind": "something_else"})
    payload = AcceptanceModel.fit(*_synthetic_rows(), attack_version="x").to_dict()
    payload["feature_names"] = payload["feature_names"][:-1]
    with pytest.raises(ValueError):
        AcceptanceModel.from_dict(payload)


def test_technique_priors_from_kb_and_train_counts_roundtrip(tmp_path):
    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique

    kb = AttackKnowledgeBase(
        techniques={
            "T1105": Technique(id="T1105", name="Ingress Tool Transfer"),
            "T1003.001": Technique(id="T1003.001", name="LSASS Memory", is_subtechnique=True),
        },
        groups={
            "G0001": Group(
                id="G0001",
                name="A",
                technique_ids=["T1105", "T1003.001"],
                technique_behaviors={"T1105": ["A downloaded tools.", "A fetched a payload."]},
            ),
            "G0002": Group(id="G0002", name="B", technique_ids=["T1105"]),
        },
    )
    priors = compute_technique_priors(kb, {"T1105": 300, "T1003.001": 3})
    assert set(priors) == {"T1105", "T1003.001"}
    assert priors["T1105"][0] == pytest.approx(np.log1p(300))
    assert priors["T1105"][2] == pytest.approx(np.log1p(2))  # two groups document it
    assert priors["T1003.001"][2] == pytest.approx(np.log1p(1))
    assert priors["T1105"][1] > priors["T1003.001"][1]  # more procedure sentences
    assert prior_features(priors, "T9999") == dict.fromkeys(PRIOR_FEATURE_NAMES, 0.0)

    rows, labels = _synthetic_rows()
    for row in rows:
        row.update(prior_features(priors, "T1105"))
    model = AcceptanceModel.fit(rows, labels, attack_version="19.2", technique_priors=priors)
    loaded = AcceptanceModel.load(model.save(tmp_path / "priors.json"))
    assert loaded is not None
    assert loaded.prior_features("T1105") == prior_features(priors, "T1105")
    assert loaded.prior_features("T9999")["technique_prior_log_freq"] == 0.0
    assert loaded.probability(rows[0]) == pytest.approx(model.probability(rows[0]))


# ----- runtime-aligned threshold -----


def test_choose_threshold_runtime_without_guards_equals_the_probability_rule():
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.2, 0.1]
    labels = [1, 1, 0, 1, 0, 0]
    case_ids = ["a", "a", "b", "b", "c", "c"]
    techniques = ["T1", "T2", "T3", "T4", "T5", "T6"]
    expected_counts = {"a": 3, "b": 1, "c": 0}
    for floor in (0.5, 0.9, 1.1):
        plain = choose_threshold(
            probabilities, labels, case_ids, expected_counts, min_precision=floor
        )
        runtime = choose_threshold_runtime(
            probabilities,
            labels,
            case_ids,
            expected_counts,
            blocked=[False] * 6,
            forced=[False] * 6,
            technique_ids=techniques,
            min_precision=floor,
        )
        assert {key: runtime[key] for key in plain} == plain
        assert runtime["rule"] == "runtime"


def test_choose_threshold_runtime_applies_guards_cues_and_counts_techniques_once():
    # case a: a cue-supported row is accepted at any threshold, a blocked row (benign context
    # or a hierarchy parent) never on p; case b: the same technique judged in two sentences
    probabilities = [0.10, 0.95, 0.90, 0.80, 0.85]
    labels = [1, 0, 1, 1, 1]
    case_ids = ["a", "a", "a", "b", "b"]
    techniques = ["T1", "T2", "T3", "T4", "T4"]
    blocked = [False, True, False, False, False]
    forced = [True, False, False, False, False]
    expected = {"a": 2, "b": 1}
    choice = choose_threshold_runtime(
        probabilities,
        labels,
        case_ids,
        expected,
        blocked=blocked,
        forced=forced,
        technique_ids=techniques,
        min_precision=0.5,
        grid=[0.5, 0.99],
    )
    # at 0.5: T1 (forced) + T3 + T4 once; T2 is blocked although p = 0.95
    assert choice["threshold"] == 0.5
    assert choice["accepted"] == 3
    assert choice["precision"] == pytest.approx(1.0)
    assert choice["recall"] == pytest.approx(3 / 3)
    assert choice["forced_rows"] == 1 and choice["blocked_rows"] == 1
    # at 0.99 only the forced row survives
    high = choose_threshold_runtime(
        probabilities,
        labels,
        case_ids,
        expected,
        blocked=blocked,
        forced=forced,
        technique_ids=techniques,
        min_precision=0.5,
        grid=[0.99],
    )
    assert high["accepted"] == 1 and high["recall"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="misaligned"):
        choose_threshold_runtime(
            probabilities,
            labels,
            case_ids,
            expected,
            blocked=blocked[:-1],
            forced=forced,
            technique_ids=techniques,
        )


# ----- retrieval contract -----


def _default_contract(fingerprint: str | None = "fp-1", **changes):
    from apt_classifier.config import Settings
    from apt_classifier.presets import apply_preset
    from apt_classifier.retrieve import retrieval_contract_for

    settings = apply_preset(Settings(_env_file=None)).model_copy(update=changes)
    source = SimpleNamespace(fingerprint=fingerprint, search_class="exact") if fingerprint else None
    return retrieval_contract_for(settings, source)


def test_legacy_contract_is_the_serving_default():
    """LEGACY_CONTRACT must equal what the default serving settings produce (except the exemplar
    fingerprint, which a legacy artifact supplies itself)."""
    runtime = _default_contract("fp-1")
    assert set(runtime) == {"contract_version", *RETRIEVAL_CONTRACT_KEYS, *CONTRACT_RECORD_KEYS}
    # the backend is recorded beside the compared keys, never compared itself
    assert not set(CONTRACT_RECORD_KEYS) & set(RETRIEVAL_CONTRACT_KEYS)
    assert runtime["exemplar_backend"] == "npz"
    assert _default_contract(None)["exemplar_backend"] is None
    assert set(LEGACY_CONTRACT) == set(runtime)
    assert {key: runtime[key] for key in runtime if key != "exemplar_fingerprint"} == {
        key: LEGACY_CONTRACT[key] for key in runtime if key != "exemplar_fingerprint"
    }
    assert (
        contract_mismatches(
            legacy_contract({"exemplar_fingerprint": "fp-1"}), runtime, exemplars=True
        )
        == []
    )


def _saved_legacy_model(tmp_path, name: str, *, fingerprint: str | None, metadata=None):
    """A model saved the way artifacts were before contracts (no retrieval_contract key)."""
    model = AcceptanceModel.fit(*_synthetic_rows(), attack_version="19.2")
    model.metadata = {"exemplar_fingerprint": fingerprint, **(metadata or {})}
    payload = model.to_dict()
    assert "retrieval_contract" not in payload
    path = tmp_path / name
    path.write_text(json.dumps(payload), "utf-8")
    return path


def test_legacy_model_is_held_only_to_the_knobs_new_with_contracts(tmp_path):
    """An artifact from before contracts ran under any value of the older knobs (presets,
    sliders, ablations, exemplar k/tau); only the knobs that did not exist then can refuse it."""
    loaded = AcceptanceModel.load(_saved_legacy_model(tmp_path, "m.json", fingerprint="fp-1"))
    assert loaded.contract_origin == "legacy"
    assert loaded.contract["exemplar_fingerprint"] == "fp-1"
    assert loaded.contract["contract_version"] == 0
    assert set(loaded.contract) == {"contract_version", "exemplar_fingerprint", *CONTRACT_1_KEYS}
    assert loaded.compatible_with("19.2", "fp-1", _default_contract()) == (True, "ok")
    for changes in (
        {"technique_exemplar_k": 128},
        {"technique_exemplar_tau": 0.08},
        {"technique_include_synthetic": True},
        {"technique_platform_from_text": False},
        {"technique_candidate_pool": 16},
        {"retrieval_top_k": 20},
        {"similarity_threshold": 0.5},
        {"technique_retrieval_mode": "dense"},
        {"technique_parent_policy": "exact_only"},
        {"technique_include_detection": True},
    ):
        assert loaded.compatible_with("19.2", "fp-1", _default_contract(**changes)) == (
            True,
            "ok",
        ), changes
    for changes, key in (
        (
            {"technique_dense_relative_margin": 0.16},
            "dense_relative_margin 0.16 (fitted with 0.12)",
        ),
        ({"technique_dense_overfetch": 90}, "dense_overfetch 90 (fitted with 0)"),
        ({"technique_exemplar_vote_norm": "label_split"}, "exemplar_vote_norm"),
        ({"technique_exemplar_segment_votes": "max"}, "exemplar_segment_votes"),
        ({"technique_collection_name": "attack_techniques-copy"}, "dense_collection"),
    ):
        ok, reason = loaded.compatible_with("19.2", "fp-1", _default_contract(**changes))
        assert not ok and key in reason, changes
    # an approximate exemplar search is a different input as well
    approximate = _default_contract()
    approximate["exemplar_search_class"] = "approximate"
    ok, reason = loaded.compatible_with("19.2", "fp-1", approximate)
    assert not ok and "exemplar_search_class" in reason
    # the exemplar fingerprint is still checked as before contracts
    assert not loaded.compatible_with("19.2", "fp-2", _default_contract("fp-2"))[0]
    # without a runtime contract only the old checks apply
    assert loaded.compatible_with("19.2", "fp-1") == (True, "ok")
    # re-saving keeps it a legacy artifact (the contract is not written back)
    assert "retrieval_contract" not in loaded.to_dict()


@pytest.mark.parametrize(
    "preset", ["balanced", "conservative", "exploratory", "local_only", "legacy"]
)
def test_legacy_model_keeps_running_under_every_preset(tmp_path, preset):
    """Regression: the presets used the shipped (contract-less) model before contracts and
    must keep using it; only the rule-based gate ran when it was refused."""
    primary = _saved_legacy_model(tmp_path, "primary.json", fingerprint="fp-1")
    fallback = _saved_legacy_model(tmp_path, "fallback.json", fingerprint=None)
    settings = SimpleNamespace(
        acceptance_model_enabled=True,
        acceptance_model_path=primary,
        acceptance_model_fallback_path=fallback,
    )
    from apt_classifier.config import Settings
    from apt_classifier.presets import apply_preset
    from apt_classifier.retrieve import retrieval_contract_for

    source = SimpleNamespace(fingerprint="fp-1", search_class="exact")
    contract = retrieval_contract_for(apply_preset(Settings(_env_file=None), preset), source)
    if preset in ("conservative", "exploratory", "legacy"):
        assert contract != _default_contract()  # the preset changes retrieval knobs
    model, status = load_acceptance_model(
        settings, SimpleNamespace(version="19.2"), source, contract=contract
    )
    assert status == "active" and model is not None and model.requires_exemplars, preset


def test_recorded_settings_of_a_legacy_artifact_do_not_constrain_it(tmp_path):
    """metadata.exemplars / metadata.retrieval_settings were recorded but never checked before
    contracts, so they do not refuse the model now either."""
    path = _saved_legacy_model(
        tmp_path,
        "m.json",
        fingerprint="fp-1",
        metadata={
            "exemplars": {"exemplar_k": 32, "exemplar_tau": 0.05},
            "retrieval_settings": {"retrieval_top_k": 30, "candidate_pool": 16},
        },
    )
    loaded = AcceptanceModel.load(path)
    assert "exemplar_k" not in loaded.contract and "candidate_pool" not in loaded.contract
    assert loaded.compatible_with("19.2", "fp-1", _default_contract()) == (True, "ok")


def test_stored_contract_round_trips_and_is_enforced(tmp_path):
    contract = _default_contract(technique_exemplar_k=128)
    model = AcceptanceModel.fit(*_synthetic_rows(), attack_version="19.2", contract=contract)
    model.metadata = {"exemplar_fingerprint": "fp-1"}
    loaded = AcceptanceModel.load(model.save(tmp_path / "m.json"))
    assert loaded.contract_origin == "artifact" and loaded.contract == contract
    assert loaded.compatible_with("19.2", "fp-1", contract) == (True, "ok")
    ok, reason = loaded.compatible_with("19.2", "fp-1", _default_contract())
    assert not ok and "exemplar_k 64 (fitted with 128)" in reason
    # a model fitted in memory without a contract is unconstrained
    free = AcceptanceModel.fit(*_synthetic_rows(), attack_version="19.2")
    assert free.contract is None
    assert free.compatible_with("19.2", None, _default_contract(technique_candidate_pool=20))[0]
    # a contract from newer code is refused; a key an older contract lacks counts as legacy
    newer = dict(contract, contract_version=99)
    assert "newer" in contract_mismatches(newer, contract, exemplars=True)[0]
    older = {key: value for key, value in contract.items() if key != "dense_overfetch"}
    assert contract_mismatches(older, contract, exemplars=True) == []
    bad = model.to_dict()
    bad["retrieval_contract"] = ["not", "an", "object"]
    with pytest.raises(ValueError):
        AcceptanceModel.from_dict(bad)
    # version 0 (the loose legacy comparison) never comes from a file
    for version in (0, "0"):
        loose = model.to_dict()
        loose["retrieval_contract"] = dict(contract, contract_version=version)
        with pytest.raises(ValueError, match="version 0"):
            AcceptanceModel.from_dict(loose)
    # a missing or malformed version is refused, never read as version 0
    assert contract_mismatches(dict(contract, contract_version=None), contract, exemplars=True)
    # a malformed value fails closed instead of raising at load
    broken = dict(contract, top_k="many")
    assert contract_mismatches(broken, contract, exemplars=True) == [
        "top_k has an unreadable value"
    ]
    assert contract_mismatches(dict(contract, contract_version="x"), contract, exemplars=True)


def _saved_contract_model(tmp_path, name: str, *, fingerprint: str | None, contract):
    model = AcceptanceModel.fit(*_synthetic_rows(), attack_version="19.2", contract=contract)
    model.metadata = {"exemplar_fingerprint": fingerprint}
    return model.save(tmp_path / name)


@pytest.mark.parametrize("stored", [False, True])
def test_no_exemplar_model_is_compared_without_the_exemplar_keys(tmp_path, stored):
    if stored:
        primary = _saved_contract_model(
            tmp_path, "primary.json", fingerprint="fp-1", contract=_default_contract()
        )
        fallback = _saved_contract_model(
            tmp_path, "fallback.json", fingerprint=None, contract=_default_contract(None)
        )
        exemplar_change = {"technique_exemplar_k": 128}
        retrieval_change = {"technique_candidate_pool": 16}
    else:
        # artifacts from before contracts: only the knobs new with contracts decide
        primary = _saved_legacy_model(tmp_path, "primary.json", fingerprint="fp-1")
        fallback = _saved_legacy_model(tmp_path, "fallback.json", fingerprint=None)
        exemplar_change = {"technique_exemplar_vote_norm": "label_split"}
        retrieval_change = {"technique_dense_relative_margin": 0.16}
    settings = SimpleNamespace(
        acceptance_model_enabled=True,
        acceptance_model_path=primary,
        acceptance_model_fallback_path=fallback,
    )
    kb = SimpleNamespace(version="19.2")
    index = SimpleNamespace(fingerprint="fp-1", search_class="exact")
    model, status = load_acceptance_model(settings, kb, index, contract=_default_contract())
    assert status == "active" and model.requires_exemplars
    # an exemplar knob changed: the primary is refused, the no-exemplar model stands in
    model, status = load_acceptance_model(
        settings, kb, index, contract=_default_contract(**exemplar_change)
    )
    assert model is not None and not model.requires_exemplars
    assert status.startswith("fallback without exemplars (incompatible artifact: retrieval")
    assert next(iter(exemplar_change)).removeprefix("technique_") in status
    # a retrieval knob changed: both are refused
    model, status = load_acceptance_model(
        settings, kb, index, contract=_default_contract(**retrieval_change)
    )
    assert model is None and next(iter(retrieval_change)).removeprefix("technique_") in status
    # exemplar keys never decide for a no-exemplar model
    loaded = AcceptanceModel.load(fallback)
    contract = _default_contract(None, technique_exemplar_tau=0.5)
    assert contract["exemplar_tau"] != LEGACY_CONTRACT["exemplar_tau"]
    assert set(EXEMPLAR_CONTRACT_KEYS) <= set(contract)
    assert loaded.compatible_with("19.2", None, contract) == (True, "ok")


def test_a_legacy_fallback_never_runs_under_a_refit_primarys_tuned_settings(tmp_path):
    """Behind a primary that stores a contract, a contract-less no-exemplar fallback must meet
    the serving defaults it ran under; behind a contract-less primary it stands in as before."""
    fallback = _saved_legacy_model(tmp_path, "fallback.json", fingerprint=None)
    kb = SimpleNamespace(version="19.2")
    for name, primary_contract, runtime, expected in (
        # a refit with another candidate pool; the exemplar index is gone at runtime
        (
            "tuned.json",
            _default_contract(technique_candidate_pool=16),
            {"technique_candidate_pool": 16},
            None,
        ),
        # a refit at the serving defaults: the old fallback may stand in
        ("aligned.json", _default_contract(), {}, "fallback"),
    ):
        primary = _saved_contract_model(
            tmp_path, name, fingerprint="fp-1", contract=primary_contract
        )
        settings = SimpleNamespace(
            acceptance_model_enabled=True,
            acceptance_model_path=primary,
            acceptance_model_fallback_path=fallback,
        )
        model, status = load_acceptance_model(
            settings, kb, None, contract=_default_contract(None, **runtime)
        )
        if expected is None:
            assert (
                model is None and "fallback refused: candidate_pool 16 (fitted with 12)" in status
            )
        else:
            assert model is not None and not model.requires_exemplars
            assert status.startswith("fallback without exemplars")
    # a contract-less primary (the shipped pair): the fallback stands in under any older knob
    legacy_primary = _saved_legacy_model(tmp_path, "legacy.json", fingerprint="fp-1")
    settings = SimpleNamespace(
        acceptance_model_enabled=True,
        acceptance_model_path=legacy_primary,
        acceptance_model_fallback_path=fallback,
    )
    model, status = load_acceptance_model(
        settings, kb, None, contract=_default_contract(None, technique_candidate_pool=16)
    )
    assert model is not None and status.startswith("fallback without exemplars")


def test_shipped_acceptance_artifacts_load_under_the_legacy_contract():
    """The shipped models (read only) keep loading at the default settings."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "models" / "acceptance"
    primary, fallback = root / "acceptance_model.json", root / "acceptance_model_no_exemplars.json"
    if not primary.is_file() or not fallback.is_file():
        pytest.skip("shipped acceptance artifacts are not present")
    model = AcceptanceModel.load(primary)
    assert model.contract_origin == "legacy" and model.requires_exemplars
    fingerprint = model.exemplar_fingerprint
    source = SimpleNamespace(fingerprint=fingerprint, search_class="exact")
    from apt_classifier.config import Settings
    from apt_classifier.presets import apply_preset
    from apt_classifier.retrieve import retrieval_contract_for

    contract = retrieval_contract_for(apply_preset(Settings(_env_file=None)), source)
    assert model.compatible_with(model.attack_version, fingerprint, contract) == (True, "ok")
    no_exemplars = AcceptanceModel.load(fallback)
    assert no_exemplars.contract_origin == "legacy" and not no_exemplars.requires_exemplars
    assert no_exemplars.compatible_with(no_exemplars.attack_version, None, contract) == (
        True,
        "ok",
    )
    # regression: every preset, the Streamlit sliders and the documented ablations ran the
    # shipped model before contracts and still do (the loader call AttributionEngine makes)
    from apt_classifier.presets import PRESETS

    kb = SimpleNamespace(version=model.attack_version)
    variants = [{"technique_mapping_preset": name} for name in PRESETS] + [
        {"technique_platform_from_text": False},
        {"technique_include_synthetic": True},
        {"similarity_threshold": 0.5},
        {"retrieval_top_k": 20},
    ]
    for changes in variants:
        settings = apply_preset(Settings(_env_file=None, **changes))
        loaded, status = load_acceptance_model(
            settings.model_copy(
                update={
                    "acceptance_model_path": primary,
                    "acceptance_model_fallback_path": fallback,
                }
            ),
            kb,
            source,
            contract=retrieval_contract_for(settings, source),
        )
        assert status == "active" and loaded is not None and loaded.requires_exemplars, changes
