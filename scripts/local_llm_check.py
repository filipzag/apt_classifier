"""Check the local LLM (Ollama): reachable, model pulled, structured answers, speed, GPU memory.

    PYTHONPATH=. .venv/bin/python scripts/local_llm_check.py [--model local/qwen3.5:9b]

Runs two calls shaped like the real tasks (a verifier-style verdict and a short extraction)
through the same client the app uses, then prints tokens per second and how much of the model
sits in GPU memory (Ollama silently spills layers to the CPU when the GPU is full).
"""

import argparse
import subprocess
import sys
import time

import httpx

from apt_classifier.config import get_settings
from apt_classifier.llm import OpenRouterClient, local_llm_models

VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["technique_id", "verdict", "reason"],
    "properties": {
        "technique_id": {"type": "string"},
        "verdict": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
        "reason": {"type": "string"},
    },
}
ACTIONS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["actions"],
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sentence", "line"],
                "properties": {"sentence": {"type": "string"}, "line": {"type": "integer"}},
            },
        }
    },
}
REPORT = """1: The actor sent spearphishing emails with a malicious Word attachment.
2: Once opened, a macro launched PowerShell with an encoded command.
3: The payload added a Run key for persistence and dumped LSASS memory with procdump.
4: Data was compressed with 7-Zip and exfiltrated over HTTPS to a cloud storage bucket."""


def gpu_power_note() -> str:
    """One line on the GPU's power state; a laptop that caps the GPU makes the LLM crawl."""
    try:
        line = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pstate,clocks.sm,clocks.max.sm,power.draw,enforced.power.limit,"
                "power.default_limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
        pstate, clock, max_clock, draw, limit, default, temp = (x.strip() for x in line.split(","))
        note = (
            f"GPU {pstate}, {clock}/{max_clock} MHz, {draw} W of a {limit} W limit "
            f"(default {default} W), {temp} C"
        )
        if float(limit) < 0.5 * float(default):
            note += (
                "\n  The GPU power limit is far below its default (laptop power or thermal "
                "management): the local LLM will be very slow until it recovers."
            )
        return note
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return "GPU state unavailable (no nvidia-smi)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="local/<ollama model>, default: fallback")
    args = parser.parse_args()
    settings = get_settings().model_copy(update={"local_llm_enabled": True})
    model = args.model or settings.local_llm_fallback_model
    pulled = local_llm_models(settings)
    if pulled is None:
        print(f"Ollama is not reachable at {settings.local_llm_url} (start it: `ollama serve`).")
        return 1
    name = model.removeprefix("local/")
    if not any(tag in {name, f"{name}:latest"} for tag in pulled):
        print(f"{name} is not pulled (have: {', '.join(pulled) or 'none'}); run `make llm-pull`.")
        return 1
    print(gpu_power_note())
    client = OpenRouterClient(settings)
    checks = [
        (
            "verifier",
            VERDICT_SCHEMA,
            [
                {"role": "system", "content": "You verify MITRE ATT&CK technique candidates."},
                {
                    "role": "user",
                    "content": "Observation: 'a macro launched PowerShell with an encoded "
                    "command'. Candidate: T1059.001 PowerShell. Is it supported?",
                },
            ],
            600,
        ),
        (
            "extraction",
            ACTIONS_SCHEMA,
            [
                {
                    "role": "system",
                    "content": "Split the report into atomic attacker actions, one sentence "
                    "each, with the source line number.",
                },
                {"role": "user", "content": REPORT},
            ],
            1500,
        ),
    ]
    for label, schema, messages, max_tokens in checks:
        started = time.perf_counter()
        response = client.complete(
            messages, model=model, effort="low", schema=schema, max_tokens=max_tokens, seed=7
        )
        elapsed = time.perf_counter() - started
        raw = response.raw
        prompt_s = (raw.get("prompt_eval_duration") or 0) / 1e9
        eval_s = (raw.get("eval_duration") or 0) / 1e9
        prompt_rate = response.usage.prompt_tokens / prompt_s if prompt_s else 0.0
        eval_rate = response.usage.completion_tokens / eval_s if eval_s else 0.0
        print(
            f"{label:<10} {elapsed:6.1f}s  prompt {response.usage.prompt_tokens:>5} tok "
            f"@ {prompt_rate:7.0f} tok/s  output {response.usage.completion_tokens:>4} tok "
            f"@ {eval_rate:5.1f} tok/s"
        )
        print(f"           {response.parsed}")
    print(gpu_power_note())
    running = httpx.get(f"{client.local_url}/api/ps", timeout=5).json().get("models", [])
    for row in running:
        size, vram = int(row.get("size") or 0), int(row.get("size_vram") or 0)
        share = vram / size if size else 0.0
        print(
            f"loaded {row.get('name')}: {size / 2**30:.1f} GiB, {share:.0%} on the GPU"
            + ("" if share > 0.99 else "  (partly on the CPU: slower; free GPU memory)")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
