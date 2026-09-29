"""Evidence normalisation without the LLM.

The shipped extractor asks OpenRouter to turn prose into atomic ATT&CK-style observations.
When the app runs without a key or credit, free text is split into sentences locally so
retrieval, gating and ranking still work; the analyst is told that the split is mechanical.
"""

from __future__ import annotations

from apt_classifier.attack.parse import split_sentences
from apt_classifier.evidence.models import Observation, Provenance

MIN_WORDS = 3


def local_observations(text: str, *, source: str = "free_text", source_name: str = "analyst_input"):
    """One observation per sentence with at least three words, with line provenance."""
    observations: list[Observation] = []
    seen: set[str] = set()
    for line_number, line in enumerate(str(text or "").splitlines(), 1):
        if not line.strip():
            continue
        for sentence in split_sentences(line):
            cleaned = " ".join(sentence.split())
            if len(cleaned.split()) < MIN_WORDS:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            observations.append(
                Observation(
                    text=cleaned,
                    provenance=Provenance(
                        source=source,
                        source_name=source_name,
                        line_numbers=[line_number],
                        raw_excerpt=line.strip(),
                        metadata={"normalization": "local_sentence_split"},
                    ),
                )
            )
    return observations
