"""Mapping presets that bundle retrieval, verification, hierarchy, and gating options.

Presets adjust settings at analysis time without requiring the analyst to edit
individual environment variables. The ``balanced`` preset is the default;
``conservative`` raises acceptance requirements; ``exploratory`` widens the
candidate view without letting unsupported candidates vote; ``local_only``
disables OpenRouter verification; and ``legacy`` reproduces the original
dense-only behavior for comparison and rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MappingPreset:
    name: str
    label: str
    description: str
    overrides: dict[str, object] = field(default_factory=dict)


PRESETS: dict[str, MappingPreset] = {
    "balanced": MappingPreset(
        name="balanced",
        label="Balanced",
        description=(
            "Hybrid dense+lexical retrieval, OpenRouter verification on ambiguous "
            "candidates, ancestor-compatible hierarchy matching, strict positive "
            "group eligibility."
        ),
        overrides={
            "technique_retrieval_mode": "hybrid",
            "technique_verifier_mode": "ambiguous",
            "technique_parent_policy": "ancestor_compatible",
            "technique_include_detection": False,
        },
    ),
    "conservative": MappingPreset(
        name="conservative",
        label="Conservative",
        description=(
            "Stronger acceptance requirements, more abstention, analyst confirmation "
            "for ambiguous and low-confidence candidates."
        ),
        overrides={
            "technique_retrieval_mode": "hybrid",
            "technique_verifier_mode": "all",
            "technique_parent_policy": "exact_only",
            "technique_include_detection": False,
            "similarity_threshold": 0.55,
            "behavior_similarity_threshold": 0.65,
            "technique_acceptance_threshold": 0.70,
            "technique_relative_margin": 0.10,
        },
    ),
    "exploratory": MappingPreset(
        name="exploratory",
        label="Exploratory",
        description=(
            "Wider candidate suggestions shown in a non-voting discovery list; "
            "unsupported candidates do not enter authoritative RRF."
        ),
        overrides={
            "technique_retrieval_mode": "hybrid",
            "technique_verifier_mode": "ambiguous",
            "technique_parent_policy": "ancestor_compatible",
            "technique_candidate_pool": 20,
            "retrieval_top_k": 50,
            "technique_include_detection": False,
        },
    ),
    "local_only": MappingPreset(
        name="local_only",
        label="Local only",
        description=("Hybrid mapping without OpenRouter; uncertain candidates go to review."),
        overrides={
            "technique_retrieval_mode": "hybrid",
            "technique_verifier_mode": "off",
            "technique_parent_policy": "ancestor_compatible",
            "technique_include_detection": False,
        },
    ),
    "legacy": MappingPreset(
        name="legacy",
        label="Legacy (dense-only)",
        description=("Original ATTACK-BERT dense-only retrieval for comparison and rollback."),
        overrides={
            "technique_retrieval_mode": "dense",
            "technique_verifier_mode": "off",
            "technique_parent_policy": "ancestor_compatible",
            "technique_include_detection": True,
        },
    ),
}


def apply_preset(settings, preset_name: str | None = None):
    """Return a new settings object with preset overrides applied.

    The original settings object is not mutated. Unknown preset names fall back
    to ``balanced``.
    """
    if preset_name is None:
        preset_name = getattr(settings, "technique_mapping_preset", "balanced")
    preset = PRESETS.get(preset_name, PRESETS["balanced"])
    if hasattr(settings, "model_copy"):
        return settings.model_copy(update=preset.overrides)
    from dataclasses import replace as dc_replace

    return dc_replace(settings, **preset.overrides)


def preset_choices() -> dict[str, str]:
    """Return a mapping of preset name to label for UI dropdowns."""
    return {name: preset.label for name, preset in PRESETS.items()}
