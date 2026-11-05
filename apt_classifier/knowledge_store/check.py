"""Checking unclear uses links with the local model (docs/KNOWLEDGE_STORE_DESIGN.md, step 2).

A link the rules leave unclear (a comparison, a lure, a leak, several actors in one sentence, or a
report link whose software also appears next to another actor) is put to the local model as one
question: does the text say that this actor itself used this software? The model (Ollama, every
layer on the GPU, never the CPU) writes a short reason first, then a verdict: supported,
unsupported or uncertain. P(supported) is read from the token probabilities and renormalised
over the allowed verdicts, as the technique verifier does (``cyberre._verdict_confidences``).

Verdicts live in their own table (``KnowledgeStore.save_check``), keyed by the link, so a later
extraction keeps them. ``KnowledgeStore.links_view`` counts an unclear link as active when its
P(supported) reaches ``MODEL_CUTOFF``, the verifier's cut-off (``LOCAL_LLM_VERIFIER_MIN_CONFIDENCE``),
fixed before the first run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from apt_classifier.knowledge_store.extract import ActorNames, SoftwareMatcher

MODEL = "qwen3.5:4b-text"
PROMPT_VERSION = 1
MODEL_CUTOFF = 0.98
DEFAULT_URL = "http://127.0.0.1:11435"
# a report link shows the model at most this many sentences that name the software
REPORT_CONTEXT_SENTENCES = 6
KEEP_ALIVE = "30m"
# the server's host memory grows with every answer (9 GB after about 330 on 2026-11-03, enough
# to make the laptop swap); reloading the model this often frees it
RELOAD_EVERY = 50
# the same options for loading and asking, so a question never reloads the model differently
OPTIONS: dict[str, Any] = {
    "temperature": 0.0,
    "seed": 0,
    "num_ctx": 4096,
    "num_predict": 200,
    "num_gpu": 999,  # every layer on the GPU, or the model refuses to load
    "num_thread": 2,  # with every layer on the GPU the threads only sample and wait
}

SYSTEM = (
    "You check one claim taken from a threat-intelligence report against the report's own words. "
    "The claim says that a named threat actor uses a named piece of software (malware or a tool) "
    "in its own operations.\n"
    "Answer supported only when the text says, or plainly implies, that this actor itself used, "
    "deployed, delivered, developed or operated the software.\n"
    "Answer unsupported when the text says another actor or group used it; when it only compares "
    "them (similar to, the same as, overlaps with, linked to); when the software was leaked, "
    "published, sold or imitated; when the software name is a lure, a victim, a product being "
    "impersonated or another group's name; or when the text only mentions both.\n"
    "Answer uncertain when the text does not settle it.\n"
    "Write the reason first, in one short sentence, then the verdict."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
    },
    "required": ["reason", "verdict"],
}


@dataclass(frozen=True)
class Verdict:
    verdict: str
    p_supported: float | None
    reason: str
    tokens: int


def _surfaces(spans: Sequence[tuple[int, int, Any]], text: str) -> list[str]:
    return list(dict.fromkeys(text[start:end] for start, end, _ in spans))


def question(
    link: Mapping[str, Any],
    title: str,
    statements: Sequence[str],
    *,
    actors: ActorNames,
    software: SoftwareMatcher,
    group_names: Mapping[str, str],
) -> str:
    """The user message for one link: the actor and software as the text names them, and the
    sentences that decide it (a sentence link: its sentence and the one before; a report link:
    the post's sentences that name the software)."""
    actor = str(link["actor"])
    index = int(link["statement"])
    if link["rule"] == "sentence":
        lines = [statements[index - 1]] if index > 0 else []
        lines.append(statements[index])
    else:
        lines = [
            text
            for text in statements
            if any(item.id == link["software_id"] for _, _, item in software.find(text))
        ][:REPORT_CONTEXT_SENTENCES] or [statements[index]]
    actor_said: list[str] = []
    software_said: list[str] = []
    for text in [title, *lines]:
        actor_said += _surfaces([s for s in actors.spans(text) if s[2] == actor], text)
        software_said += _surfaces(
            [hit for hit in software.find(text) if hit[2].id == link["software_id"]], text
        )
    name = group_names.get(actor) or actor.split(":", 1)[-1]
    if actor.startswith("related:"):
        name = "a group the MISP list relates to " + ", ".join(
            group_names.get(item, item) for item in name.split(",")
        )
    actor_line = f"Actor: {name}"
    said = [surface for surface in dict.fromkeys(actor_said) if surface != name]
    if said:
        actor_line += f" (the text calls it: {', '.join(said)})"
    if link["rule"] == "report":
        actor_line += " - the report is about this actor"
    software_line = f"Software: {link['software_name']}"
    spelled = [s for s in dict.fromkeys(software_said) if s != link["software_name"]]
    if spelled:
        software_line += f" (the text calls it: {', '.join(spelled)})"
    numbered = "\n".join(f"[{n}] {text}" for n, text in enumerate(lines, 1))
    return (
        f'Report title: "{title}"\n{actor_line}\n{software_line}\nText:\n{numbered}\n\n'
        f"Claim: {name} itself used, deployed, delivered, developed or operated "
        f"{link['software_name']}."
    )


class LocalChecker:
    """One question per request to a local Ollama server, greedy decoding, token probabilities
    for the verdict. GPU only: ``load`` puts the model in GPU memory before the first question,
    and every answer is dropped (the run stops) unless all of the model is still there. The
    model is reloaded every ``reload_every`` answers to free the server's host memory."""

    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        model: str = MODEL,
        timeout: float = 300.0,
        reload_every: int = RELOAD_EVERY,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.reload_every = reload_every
        self._answered = 0
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def load(self) -> None:
        """Load the model without asking anything (an empty chat) and require all of it in GPU
        memory, so not even the first question runs on the CPU."""
        response = self._client.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [],
                "keep_alive": KEEP_ALIVE,
                "options": OPTIONS,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"local model: HTTP {response.status_code}: {response.text[:300]}")
        self.require_gpu()

    def unload(self) -> None:
        """Free the model's GPU memory and the server's host memory."""
        self._client.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0})

    def gpu_share(self) -> float:
        """The share of the loaded model in GPU memory (``/api/ps``); 0.0 when it is not loaded."""
        response = self._client.get(f"{self.url}/api/ps")
        rows = response.json().get("models", []) if response.status_code == 200 else []
        for row in rows:
            if self.model in (row.get("name"), row.get("model")):
                size = int(row.get("size") or 0)
                return int(row.get("size_vram") or 0) / size if size else 0.0
        return 0.0

    def require_gpu(self) -> None:
        """Stop unless all of the model sits in GPU memory: a model that did not fit, or a server
        that lost the GPU (the nightly driver update), would otherwise run on the CPU."""
        share = self.gpu_share()
        if share < 1.0:
            raise RuntimeError(
                f"local model {self.model}: {share:.0%} in GPU memory; stopped so that nothing "
                "runs on the CPU"
            )

    def check(self, message: str) -> Verdict:
        from apt_classifier.cyberre import _VERDICT_TOP_LOGPROBS, _verdict_confidences

        if self.reload_every and self._answered and self._answered % self.reload_every == 0:
            self.unload()
            self.load()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "think": False,
            "keep_alive": KEEP_ALIVE,
            "format": SCHEMA,
            "logprobs": True,
            "top_logprobs": _VERDICT_TOP_LOGPROBS,
            "options": OPTIONS,
        }
        response = self._client.post(f"{self.url}/api/chat", json=body)
        if response.status_code != 200:
            raise RuntimeError(f"local model: HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        self.require_gpu()  # an answer from a model partly on the CPU is never used
        content = str((payload.get("message") or {}).get("content") or "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}
        verdict = str(parsed.get("verdict") or "uncertain")
        confidences = _verdict_confidences(payload)
        self._answered += 1
        return Verdict(
            verdict if verdict in ("supported", "unsupported", "uncertain") else "uncertain",
            confidences[0] if confidences else None,
            str(parsed.get("reason") or "")[:500],
            int(payload.get("eval_count") or 0),
        )
