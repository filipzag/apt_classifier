import json
from collections.abc import Iterable, Mapping
from typing import Any

from apt_classifier.llm import OpenRouterClient

from .extract import extract_attack_actions
from .models import Observation


def _alert_text(alerts: Any) -> str:
    if alerts is None:
        return ""
    if isinstance(alerts, str):
        return alerts
    if isinstance(alerts, bytes):
        return alerts.decode("utf-8-sig")
    if isinstance(alerts, Mapping):
        return json.dumps(alerts, ensure_ascii=False, sort_keys=True)
    if isinstance(alerts, Iterable):
        return "\n".join(
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in alerts
        )
    return str(alerts)


def parse_alerts(
    alerts: Any,
    client: OpenRouterClient,
    *,
    source_name: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    seed: int | None = 0,
) -> list[Observation]:
    return extract_attack_actions(
        _alert_text(alerts),
        client,
        source_type="alert",
        source_name=source_name,
        model=model,
        effort=effort,
        seed=seed,
    )


def parse_alert_text(*args: Any, **kwargs: Any) -> list[Observation]:
    return parse_alerts(*args, **kwargs)


def extract_alerts(*args: Any, **kwargs: Any) -> list[Observation]:
    return parse_alerts(*args, **kwargs)


def parse(*args: Any, **kwargs: Any) -> list[Observation]:
    return parse_alerts(*args, **kwargs)
