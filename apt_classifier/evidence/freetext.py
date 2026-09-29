from typing import Any

from apt_classifier.llm import OpenRouterClient

from .extract import extract_attack_actions
from .models import Observation


def parse_free_text(
    text: str,
    client: OpenRouterClient,
    *,
    source_name: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    seed: int | None = 0,
) -> list[Observation]:
    return extract_attack_actions(
        text,
        client,
        source_type="free_text",
        source_name=source_name,
        model=model,
        effort=effort,
        seed=seed,
    )


def parse_freetext(*args: Any, **kwargs: Any) -> list[Observation]:
    return parse_free_text(*args, **kwargs)


def extract_freetext(*args: Any, **kwargs: Any) -> list[Observation]:
    return parse_free_text(*args, **kwargs)


def extract_free_text(*args: Any, **kwargs: Any) -> list[Observation]:
    return parse_free_text(*args, **kwargs)


def parse(*args: Any, **kwargs: Any) -> list[Observation]:
    return parse_free_text(*args, **kwargs)
