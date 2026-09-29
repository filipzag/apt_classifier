from apt_classifier.config import Settings
from apt_classifier.presets import PRESETS, apply_preset, preset_choices


def test_balanced_preset_is_default():
    settings = Settings(technique_mapping_preset="balanced")
    applied = apply_preset(settings)
    assert applied.technique_retrieval_mode == "hybrid"
    assert applied.technique_verifier_mode == "ambiguous"
    assert applied.technique_parent_policy == "ancestor_compatible"


def test_conservative_preset_raises_thresholds():
    settings = Settings(technique_mapping_preset="conservative")
    applied = apply_preset(settings)
    assert applied.technique_verifier_mode == "all"
    assert applied.technique_parent_policy == "exact_only"
    assert applied.similarity_threshold >= 0.55


def test_local_only_preset_disables_verifier():
    settings = Settings(technique_mapping_preset="local_only")
    applied = apply_preset(settings)
    assert applied.technique_verifier_mode == "off"
    assert applied.technique_retrieval_mode == "hybrid"


def test_legacy_preset_uses_dense_only():
    settings = Settings(technique_mapping_preset="legacy")
    applied = apply_preset(settings)
    assert applied.technique_retrieval_mode == "dense"
    assert applied.technique_verifier_mode == "off"
    assert applied.technique_include_detection is True


def test_unknown_preset_falls_back_to_balanced():
    settings = Settings(technique_mapping_preset="nonexistent")
    applied = apply_preset(settings)
    assert applied.technique_retrieval_mode == "hybrid"


def test_apply_preset_does_not_mutate_original():
    settings = Settings(technique_mapping_preset="balanced", similarity_threshold=0.45)
    apply_preset(settings, "conservative")
    assert settings.similarity_threshold == 0.45


def test_preset_choices_contains_all_presets():
    choices = preset_choices()
    assert set(choices) == set(PRESETS)
    assert "balanced" in choices
