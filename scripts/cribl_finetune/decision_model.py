"""cribl-decision on Qwen3.5-4B-Base with a lettered-option readout (``.venv-cribl`` only).

The base loads in 4-bit NF4 (bf16 compute) on the GPU; the LoRA is Cribl's published adapter, a
fine-tuned checkpoint of it, or a fresh one with the same shape. The readout is the hidden state at
the last prompt position (the decision slot after ``Answer:``) through the native ``lm_head`` rows
of the option letters `` A`` … , softmaxed over the offered options only.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from common import combine_rotations, letter_tokens, render_prompt, rotate, rotation_count
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ADAPTER_ID = "cribl-ai/cribl-decision-1.0"
ADAPTER_REVISION = "75cb0dd9d551e774664c01ba8f7b25c0087b4abf"
CACHE_DIR = Path(os.environ.get("CRIBL_HF_CACHE", "/home/fzagar/apt_classifier/.huggingface"))


@dataclass(frozen=True)
class BaseSpec:
    """A base model the heads can sit on: the pinned checkpoint, the LoRA target modules, the
    modules kept out of 4-bit quantisation and the chat format of the family prompt
    (``common.render_family_prompt``)."""

    repo: str
    revision: str
    target_modules: str
    skip_modules: tuple[str, ...]
    prompt_template: str
    # the auto class that loads the checkpoint: multimodal bases through image-text-to-text,
    # text-only ones (Llama) through causal LM; both expose ``.model`` as the backbone
    loader: str = "image_text_to_text"
    # modules kept in system RAM on a small GPU (under ``SMALL_GPU_GIB``): lookup tables that
    # 4-bit loading does not shrink (Gemma 4's per-layer embedding table, 5.2 GB in bf16)
    cpu_modules: tuple[str, ...] = ()


SMALL_GPU_GIB = 16


# Cribl's adapter shape (adapter_config.json): every language-model layer's linear-attention
# projections and full-attention q/k/v/o; nothing in the vision tower or the MTP head.
TARGET_MODULES = (
    r".*language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))"
)
BASES = {
    "qwen3.5-4b-base": BaseSpec(
        "Qwen/Qwen3.5-4B-Base",
        "1001bb4d826a52d1f399e183466143f4da7b741b",
        TARGET_MODULES,
        ("lm_head", "visual", "mtp"),
        "qwen",
    ),
    # Gemma 4 E4B (prereg cribl_recipe_base_2026-10-15.json): every text layer's attention
    # projections (k/v only where the layer has its own; the kv-shared layers have none).
    "gemma-4-e4b": BaseSpec(
        "google/gemma-4-E4B",
        "411aa17b749aa952df1359d2dcea73917a544d9a",
        r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        ("lm_head", "vision_tower", "audio_tower", "embed_vision", "embed_audio"),
        "gemma",
        cpu_modules=("model.language_model.embed_tokens_per_layer",),
    ),
    # Cisco's Foundation-Sec-8B (Llama 3.1 8B continued on 5.1B tokens of security text, Apache
    # 2.0; prereg cribl_recipe_base_2026-10-15.json amendment 1): every layer's attention projections
    "foundation-sec-8b": BaseSpec(
        "fdtn-ai/Foundation-Sec-8B",
        "9e6bb1f70b402e55bbf68cfd7abf88458daf09b3",
        r".*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        ("lm_head",),
        "llama3",
        loader="causal_lm",
        # on a small GPU the 1 GB token table is looked up on the CPU and the (unused) 1 GB
        # output head stays there too
        cpu_modules=("model.embed_tokens", "lm_head"),
    ),
}
DEFAULT_BASE = "qwen3.5-4b-base"
BASE_ID = BASES[DEFAULT_BASE].repo
BASE_REVISION = BASES[DEFAULT_BASE].revision
LORA_R = 16
LORA_SHAPE = {"r": LORA_R, "lora_alpha": 2 * LORA_R, "lora_dropout": 0.05, "bias": "none"}


def base_spec(base: str) -> BaseSpec:
    if base not in BASES:
        raise ValueError(f"unknown base {base!r} (one of {sorted(BASES)})")
    return BASES[base]


def lora_shape(r: int = LORA_R, alpha: int | None = None) -> dict[str, Any]:
    """Cribl's LoRA shape at rank ``r`` (alpha defaults to 2r, Cribl's ratio)."""
    if r < 1:
        raise ValueError("LoRA rank must be positive")
    return {**LORA_SHAPE, "r": int(r), "lora_alpha": int(2 * r if alpha is None else alpha)}


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


def load_tokenizer(base: str = DEFAULT_BASE) -> Any:
    spec = base_spec(base)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo, revision=spec.revision, cache_dir=CACHE_DIR, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    return tokenizer


def auto_class(spec: BaseSpec) -> Any:
    if spec.loader == "causal_lm":
        return AutoModelForCausalLM
    if spec.loader == "image_text_to_text":
        return AutoModelForImageTextToText
    raise ValueError(f"unknown loader {spec.loader!r}")


def small_gpu() -> bool:
    return torch.cuda.get_device_properties(0).total_memory < SMALL_GPU_GIB * 2**30


def device_map(spec: BaseSpec) -> dict[str, Any]:
    """Everything on the GPU; on a small GPU ``spec.cpu_modules`` go to the CPU. The map then
    names every top-level module (accelerate treats a root entry as "move everything")."""
    if not spec.cpu_modules or not small_gpu():
        return {"": 0}
    config = AutoConfig.from_pretrained(spec.repo, revision=spec.revision, cache_dir=CACHE_DIR)
    with torch.device("meta"):
        skeleton = auto_class(spec).from_config(config)
    placement: dict[str, Any] = {}

    def walk(module: Any, prefix: str) -> None:
        for child_name, child in module.named_children():
            name = f"{prefix}.{child_name}" if prefix else child_name
            if name in spec.cpu_modules:
                placement[name] = "cpu"
            elif any(item.startswith(name + ".") for item in spec.cpu_modules):
                walk(child, name)
            else:
                placement[name] = 0

    walk(skeleton, "")
    del skeleton
    return placement


def keep_on_cpu(model: Any, names: Sequence[str]) -> None:
    """Give each offloaded lookup table its weights back on the CPU and route its forward
    through the CPU: the rows of the batch come back on the GPU, the table never goes there
    (accelerate's own offload would copy the whole table per forward)."""
    from accelerate.hooks import remove_hook_from_module

    for name in names:
        module = model.get_submodule(name)
        hook = getattr(module, "_hf_hook", None)
        if hook is not None:
            weights = {
                key: hook.weights_map[key] for key, _ in module.named_parameters(recurse=False)
            }
            remove_hook_from_module(module)
            for key, value in weights.items():
                setattr(module, key, torch.nn.Parameter(value.to("cpu"), requires_grad=False))
        module.to("cpu")
        original = module.forward

        def forward(ids: torch.Tensor, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            return _original(ids.to("cpu"), *args, **kwargs).to(ids.device)

        module.forward = forward
    # PEFT re-dispatches a model whose placement map names the CPU (and then wants an offload
    # directory); the tables are wired up already, so the map it sees is GPU-only
    if names and hasattr(model, "hf_device_map"):
        model.hf_device_map = {
            key: value for key, value in model.hf_device_map.items() if value != "cpu"
        }


def load_base(*, gradient_checkpointing: bool = False, base: str = DEFAULT_BASE) -> Any:
    require_cuda()
    spec = base_spec(base)
    placement = device_map(spec)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=list(spec.skip_modules),
        llm_int8_enable_fp32_cpu_offload="cpu" in placement.values(),
    )
    model = auto_class(spec).from_pretrained(
        spec.repo,
        revision=spec.revision,
        cache_dir=CACHE_DIR,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map=placement,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    keep_on_cpu(model, [name for name, device in placement.items() if device == "cpu"])
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
    adapter: str | None = "cribl",
    *,
    trainable: bool = False,
    base: str = DEFAULT_BASE,
    lora_r: int = LORA_R,
    lora_alpha: int | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """``adapter``: ``cribl`` (published, Qwen only), ``none`` (plain base), ``fresh`` (new LoRA
    of Cribl's shape at ``lora_r``, for training) or a checkpoint directory. ``base`` names the
    base model (``BASES``). Returns (model, tokenizer, info)."""
    spec = base_spec(base)
    if adapter == "cribl" and base != DEFAULT_BASE:
        raise ValueError(f"Cribl's adapter fits {DEFAULT_BASE} only; use --init fresh on {base}")
    tokenizer = load_tokenizer(base)
    model = load_base(gradient_checkpointing=trainable, base=base)
    info: dict[str, Any] = {
        "base": f"{spec.repo}@{spec.revision}",
        "base_key": base,
        "prompt_template": spec.prompt_template,
        "adapter": adapter,
    }
    if adapter in (None, "none"):
        info["modules"] = module_summary(model)
        model.eval()
        return model, tokenizer, info
    if adapter == "fresh":
        shape = lora_shape(lora_r, lora_alpha)
        config = LoraConfig(target_modules=spec.target_modules, task_type="CAUSAL_LM", **shape)
        model = get_peft_model(model, config)
        info["lora"] = {"r": shape["r"], "alpha": shape["lora_alpha"]}
    else:
        directory = cribl_adapter_dir() if adapter == "cribl" else Path(adapter)
        saved_base = _adapter_base(directory)
        if saved_base and saved_base != spec.repo:
            raise ValueError(f"adapter {directory} was trained on {saved_base}, not {spec.repo}")
        model = PeftModel.from_pretrained(model, str(directory), is_trainable=trainable)
        info["adapter_dir"] = str(directory)
        info["adapter_check"] = verify_adapter(model, directory)
        peft_config = model.peft_config["default"]
        info["lora"] = {"r": int(peft_config.r), "alpha": int(peft_config.lora_alpha)}
        if adapter == "cribl":
            info["adapter"] = f"{ADAPTER_ID}@{ADAPTER_REVISION}"
    info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["modules"] = module_summary(model)
    model.train(trainable)
    return model, tokenizer, info


def module_summary(model: Any) -> dict[str, Any]:
    """How the base was placed: 4-bit linear layers, embeddings left in bf16 (an embedding that
    was quantised is a stop-rule failure), LoRA-wrapped modules and modules kept on the CPU."""
    quantised = lora = 0
    embeddings: list[str] = []
    cpu: list[str] = []
    for name, module in model.named_modules():
        kind = type(module).__name__
        if kind == "Linear4bit":
            quantised += 1
        if "lora_A" in dict(module.named_children()):
            lora += 1
        if isinstance(module, torch.nn.Embedding):
            weight = module.weight
            embeddings.append(f"{name}: {str(weight.dtype).removeprefix('torch.')} {weight.device}")
            if weight.device.type == "cpu":
                cpu.append(name)
    return {"linear_4bit": quantised, "lora_modules": lora, "embeddings": embeddings, "cpu": cpu}


def _adapter_base(directory: Path) -> str | None:
    """The base model an adapter directory records (``adapter_config.json``), if any."""
    path = directory / "adapter_config.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("base_model_name_or_path") or None


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
