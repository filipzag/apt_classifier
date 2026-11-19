"""HTTP front of the technique head service: standard library only, so the request handling is
testable in the product's ``.venv`` without torch or transformers (``serve.py`` loads the model
and plugs a scorer in).

  GET  /health  -> {"status": "ready", "fingerprint": ..., "attack_version": ..., ...}
  POST /score   {"texts": ["...", ...]} -> {"fingerprint": ..., "rows": [row, ...]}

A row is ``{"status": "ok", "families": [[id, p], ...], "techniques": [[id, p], ...], "pick",
"p_pick", "technique_pick", "technique_p"}`` (most probable first, the kept top-K of each head)
or ``{"status": "error", "error": "..."}`` for a text the head could not read (over the token
limit). The fingerprint is the SHA-256 over the checkpoint's adapter and head files
(:func:`checkpoint_fingerprint`); the product pins it with ``TECHNIQUE_HEAD_EXPECTED_FINGERPRINT``
and an acceptance model fitted with the head records it.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

FINGERPRINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "family_head.pt",
    "technique_head.pt",
)
DEFAULT_PORT = 8102
MAX_TEXTS = 256
MAX_BODY = 4 << 20

Scorer = Callable[[list[str]], list[dict[str, Any]]]


def checkpoint_fingerprint(directory: Path) -> str:
    """SHA-256 over the names and bytes of the checkpoint files that define the head (a missing
    file hashes as empty, so a family-only checkpoint has its own fingerprint)."""
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        path = Path(directory) / name
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


class HeadService:
    """The service state: a scorer (texts -> rows), the manifest ``/health`` reports, and one
    lock so requests reach the GPU one at a time."""

    def __init__(self, scorer: Scorer, manifest: Mapping[str, Any], *, max_texts: int = MAX_TEXTS):
        self.scorer = scorer
        self.manifest = dict(manifest)
        self.max_texts = int(max_texts)
        self.lock = threading.Lock()
        self.scored = 0
        self.requests = 0

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            **self.manifest,
            "scored": self.scored,
            "requests": self.requests,
        }

    def score(self, payload: Any) -> tuple[int, dict[str, Any]]:
        """``(http status, body)`` for one ``/score`` request."""
        if not isinstance(payload, Mapping) or not isinstance(payload.get("texts"), list):
            return 400, {"error": 'expected {"texts": [...]}'}
        texts = payload["texts"]
        if not all(isinstance(text, str) for text in texts):
            return 400, {"error": "every text must be a string"}
        if len(texts) > self.max_texts:
            return 413, {"error": f"at most {self.max_texts} texts per request"}
        with self.lock:
            self.requests += 1
            try:
                rows = self.scorer(list(texts))
            except Exception as error:  # noqa: BLE001 - reported to the client, never raised
                return 500, {"error": f"scoring failed: {str(error)[:300]}"}
            self.scored += len(texts)
        if len(rows) != len(texts):
            return 500, {"error": "the scorer returned the wrong number of rows"}
        return 200, {"fingerprint": self.manifest.get("fingerprint"), "rows": rows}


def make_handler(service: HeadService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "technique-head/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
            return  # quiet: the caller prints its own progress

        def _send(self, status: int, body: Mapping[str, Any]) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            if self.path.split("?", 1)[0] == "/health":
                self._send(200, service.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            if self.path.split("?", 1)[0] != "/score":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self._send(413, {"error": f"body must be at most {MAX_BODY} bytes"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "body is not JSON"})
                return
            status, body = service.score(payload)
            self._send(status, body)

    return Handler


def serve(service: HeadService, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """Serve until interrupted (localhost only by default)."""
    server = ThreadingHTTPServer((host, port), make_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
