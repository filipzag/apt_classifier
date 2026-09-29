"""Guarded execution of UI steps.

Every step that can fail for an operational reason (OpenRouter credit, key, rate limit,
network, Chroma down) runs through :func:`run_step` so the analyst sees a plain-language
message and a collapsed technical detail block instead of a raw traceback, and the page
can stop cleanly instead of half-rendering.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import streamlit as st

T = TypeVar("T")


def describe_error(exc: BaseException) -> str:
    """Map common operational failures to a sentence an analyst can act on."""
    text = str(exc)
    lowered = text.lower()
    if "402" in text or "payment required" in lowered:
        return (
            "OpenRouter refused the request because the account has no credit. "
            "Top up at openrouter.ai, then retry. Local retrieval and ranking keep working."
        )
    if "401" in text or "unauthorized" in lowered or "invalid api key" in lowered:
        return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env and restart."
    if "429" in text or "rate limit" in lowered:
        return "OpenRouter rate limit reached. Wait a moment and retry."
    if "timed out" in lowered or "timeout" in lowered:
        return "The call timed out. Retry, or reduce the amount of text sent in one step."
    if "chroma" in lowered or "connection refused" in lowered or "connect" in lowered:
        return (
            "A backing service could not be reached. Check that Chroma is running "
            "(`make up`) and that the network allows outbound calls."
        )
    if "json" in lowered and ("parse" in lowered or "decode" in lowered):
        return "The model returned an unreadable answer. Retry; the step is safe to repeat."
    return "The step failed. Details are below; the case state was not changed."


@dataclass(frozen=True)
class StepResult:
    ok: bool
    value: Any = None
    error: BaseException | None = None


def run_step(label: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> StepResult:
    """Run ``fn`` and render a friendly error on failure.

    Returns ``StepResult(ok=True, value=...)`` or ``StepResult(ok=False, error=...)``. The
    caller decides whether to stop the page. Nothing is written to session state here.
    """
    try:
        return StepResult(ok=True, value=fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - the whole point is to catch everything
        st.error(f"**{label} did not complete.** {describe_error(exc)}")
        with st.expander("Technical details", expanded=False):
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            st.code(detail[:2000] or type(exc).__name__)
        return StepResult(ok=False, error=exc)
