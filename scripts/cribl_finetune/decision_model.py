"""cribl-decision on Qwen3.5-4B-Base with a lettered-option readout (``.venv-cribl`` only).

The base loads in 4-bit NF4 (bf16 compute) on the GPU; the LoRA is Cribl's published adapter, a
fine-tuned checkpoint of it, or a fresh one with the same shape. The readout is the hidden state at
the last prompt position (the decision slot after ``Answer:``) through the native ``lm_head`` rows
of the option letters `` A`` … , softmaxed over the offered options only.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from common import combine_rotations, letter_tokens, render_prompt, rotate, rotation_count
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

BASE_ID = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
ADAPTER_ID = "cribl-ai/cribl-decision-1.0"
ADAPTER_REVISION = "75cb0dd9d551e774664c01ba8f7b25c0087b4abf"
CACHE_DIR = Path(os.environ.get("CRIBL_HF_CACHE", "/home/fzagar/apt_classifier/.huggingface"))
# Cribl's adapter shape (adapter_config.json): every language-model layer's linear-attention
# projections and full-attention q/k/v/o; nothing in the vision tower or the MTP head.
TARGET_MODULES = (
    r".*language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))"
)
LORA_SHAPE = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none"}


OOM_WAIT_S = 60


def is_oom(error: BaseException) -> bool:
    """A GPU allocation failure (the GPU is shared with other sessions' jobs)."""
    text = str(error)
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and ("out of memory" in text or "CUBLAS_STATUS_ALLOC_FAILED" in text)
    )


def wait_after_oom(note: str = "") -> None:
    print(
        f"GPU out of memory{f' ({note})' if note else ''}; retrying in {OOM_WAIT_S} s", flush=True
    )
    torch.cuda.empty_cache()
    time.sleep(OOM_WAIT_S)


def with_oom_retry(call: Callable[[], Any], attempts: int = 10, note: str = "") -> Any:
    """``call()``, waiting and calling again after a GPU allocation failure (for resumable work
    such as ``evaluate.run_split``)."""
    for attempt in range(attempts):
        try:
            return call()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if not is_oom(error) or attempt == attempts - 1:
                raise
        wait_after_oom(note)  # outside the handler, so the failed frames are freed first
    raise AssertionError("unreachable")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required (4-bit 4B model; the CPU is far too slow)")
    return torch.device("cuda")


def load_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_ID, revision=BASE_REVISION, cache_dir=CACHE_DIR, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    return tokenizer


def load_base(*, gradient_checkpointing: bool = False) -> Any:
    require_cuda()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head", "visual", "mtp"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_ID,
        revision=BASE_REVISION,
        cache_dir=CACHE_DIR,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def cribl_adapter_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ADAPTER_ID, revision=ADAPTER_REVISION, cache_dir=CACHE_DIR))


def verify_adapter(model: Any, adapter_dir: Path) -> dict[str, Any]:
    """Every tensor in ``adapter_model.safetensors`` is in the model, bit for bit (a key-path
    mismatch would otherwise leave the LoRA at its init silently)."""
    expected = load_file(str(adapter_dir / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    missing = sorted(set(expected) - set(loaded))
    extra = sorted(set(loaded) - set(expected))
    different = sorted(
        key
        for key in set(expected) & set(loaded)
        if not torch.equal(expected[key].to(loaded[key].device, loaded[key].dtype), loaded[key])
    )
    report = {
        "expected": len(expected),
        "loaded": len(loaded),
        "missing": missing[:5],
        "extra": extra[:5],
        "different": different[:5],
    }
    if missing or extra or different:
        raise SystemExit(f"adapter did not load cleanly: {report}")
    return report


def load_model(
    adapter: str | None = "cribl", *, trainable: bool = False
) -> tuple[Any, Any, dict[str, Any]]:
    """``adapter``: ``cribl`` (published), ``none`` (plain base), ``fresh`` (new LoRA with Cribl's
    shape, for training) or a checkpoint directory. Returns (model, tokenizer, info)."""
    tokenizer = load_tokenizer()
    model = load_base(gradient_checkpointing=trainable)
    info: dict[str, Any] = {"base": f"{BASE_ID}@{BASE_REVISION}", "adapter": adapter}
    if adapter in (None, "none"):
        model.eval()
        return model, tokenizer, info
    if adapter == "fresh":
        config = LoraConfig(target_modules=TARGET_MODULES, task_type="CAUSAL_LM", **LORA_SHAPE)
        model = get_peft_model(model, config)
    else:
        directory = cribl_adapter_dir() if adapter == "cribl" else Path(adapter)
        model = PeftModel.from_pretrained(model, str(directory), is_trainable=trainable)
        info["adapter_dir"] = str(directory)
        info["adapter_check"] = verify_adapter(model, directory)
        if adapter == "cribl":
            info["adapter"] = f"{ADAPTER_ID}@{ADAPTER_REVISION}"
    info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train(trainable)
    return model, tokenizer, info


class Readout:
    """Letter probabilities at the decision slot: the final hidden state at the last position
    times the native ``lm_head`` rows of the letters, in float32 (the model's own bf16 logits
    are rounded to 1/8 at typical magnitudes, too coarse for calibrated probabilities)."""

    def __init__(self, model: Any, tokenizer: Any, fmt: str, max_tokens: int = 2048) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.fmt = fmt
        self.max_tokens = max_tokens
        self.letter_ids = []
        for text in letter_tokens(26):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"option token {text!r} is not one token: {ids}")
            self.letter_ids.append(ids[0])
        self.device = next(model.parameters()).device
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.backbone = base.model
        self.lm_head = base.lm_head

    def encode(self, state: Mapping[str, Any], instructions: str, options: Sequence[Any]) -> Any:
        prompt = render_prompt(state, instructions, options, self.fmt)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens (> {self.max_tokens})")
        return torch.tensor([ids], device=self.device)

    def letter_logits(self, input_ids: Any, count: int) -> Any:
        """Float32 logits of the first ``count`` letters at the last position."""
        hidden = self.backbone(input_ids=input_ids, use_cache=False).last_hidden_state[0, -1]
        weights = self.lm_head.weight[self.letter_ids[:count]]
        return weights.float() @ hidden.float()

    @torch.inference_mode()
    def probabilities(
        self, state: Mapping[str, Any], instructions: str, options: Sequence[Mapping[str, str]]
    ) -> list[float]:
        logits = self.letter_logits(self.encode(state, instructions, options), len(options))
        return torch.softmax(logits, dim=-1).tolist()

    def choice(
        self,
        state: Mapping[str, Any],
        instructions: str,
        options: Sequence[Mapping[str, str]],
        rotations: int = 3,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """The rotation-combined distribution over option ids (C1 rules) and each rotation's."""
        order = [option["id"] for option in options]
        per_rotation = []
        for rotation in range(rotation_count(options, rotations)):
            shown = rotate(options, rotation)
            probabilities = self.probabilities(state, instructions, shown)
            per_rotation.append(
                {option["id"]: p for option, p in zip(shown, probabilities, strict=True)}
            )
        return combine_rotations(per_rotation, order), per_rotation
