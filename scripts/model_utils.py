#!/usr/bin/env python
"""Shared model/data utilities used by training, merging, and eval scripts.

Covers the two non-obvious quirks of the target stack:

1. Qwen/Qwen3.5-4B-Base is a *multimodal* model (Qwen3_5ForConditionalGeneration:
   hybrid Gated-DeltaNet linear attention + gated full attention + a vision
   encoder). Some transformers versions expose it only through
   AutoModelForImageTextToText, so we try AutoModelForCausalLM first and fall
   back. Everything here works text-only; the vision tower just sits unused.

2. The base model is *base* (not instruction-tuned): its tokenizer DOES ship a
   chat template, but if any model ships without one we install a minimal
   ChatML template so training/generation always have a well-defined format.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Optional

import torch

# Fallback chat template (identical to Qwen's ChatML format).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

# Canonical LoRA target projections for Qwen-style transformer blocks.
QWEN_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
_VISION_HINTS = ("visual", "vision_tower", "vision_tower.", "vision_model", "patches_embedding")


# --------------------------------------------------------------------------- #
# Attention implementation selection
# --------------------------------------------------------------------------- #
def resolve_attn(choice: str) -> str:
    """"auto" -> flash_attention_2 if installed, else PyTorch SDPA (always available)."""
    if choice != "auto":
        return choice
    try:
        importlib.import_module("flash_attn")
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = CHATML_TEMPLATE
        print("[info] tokenizer had no chat template — installed ChatML fallback")
    return tok


# --------------------------------------------------------------------------- #
# Model loading (handles the multimodal Qwen3.5 quirk)
# --------------------------------------------------------------------------- #
def _load_causal_class(model_id: str, kwargs: dict[str, Any]):
    """Try AutoModelForCausalLM, fall back to AutoModelForImageTextToText."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs), "causal_lm"
    except Exception as e:  # noqa: BLE001
        from transformers import AutoModelForImageTextToText

        print(f"[info] AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:120]}); "
              f"loading as multimodal AutoModelForImageTextToText")
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs), "image_text_to_text"


def autodetect_lora_targets(model) -> list[str] | str:
    """Pick LoRA target modules by inspecting the actual module tree.

    Rules:
      * Ignore the vision tower entirely (no adapters there for text SFT).
      * Never target lm_head / embeddings (often tied to inputs; LoRA there
        is wasteful and can break weight tying).
      * If the classic Qwen projections exist in the language stack, use them.
      * Otherwise use the unique leaf-linear names of the language stack.
      * If the model is plain-text and nothing matched, let PEFT use all-linear.
    """
    skip = {"lm_head"}
    text_leaves: set[str] = set()
    for name, mod in model.named_modules():
        lowered = name.lower()
        if any(h in lowered for h in _VISION_HINTS):
            continue
        if isinstance(mod, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in skip or "embed" in leaf.lower():
                continue
            text_leaves.add(leaf)

    classic = sorted(text_leaves & QWEN_PROJ_NAMES)
    if classic:
        return classic
    if text_leaves:
        return sorted(text_leaves)
    return "all-linear"


def load_model(
    model_id: str,
    attn: str = "auto",
    for_training: bool = False,
    load_in_4bit: bool = False,
    device_map: Optional[str] = None,
):
    """Load model (+tokenizer if with_tokenizer). Returns (model, info).

    info: {"attn": str, "class": str, "is_multimodal": bool,
           "lora_target_modules": list[str] | "all-linear"}
    """
    attn_impl = resolve_attn(attn)
    # transformers 5.x renamed torch_dtype -> dtype; support both
    import transformers as _tf
    major = int(_tf.__version__.split(".")[0])
    dtype_kw = "dtype" if major >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    kwargs[dtype_kw] = torch.bfloat16
    if device_map:
        kwargs["device_map"] = device_map
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": int(importlib.import_module("os").environ.get("LOCAL_RANK", 0))}

    model, cls = _load_causal_class(model_id, kwargs)
    model.config.pad_token_id = model.config.pad_token_id or None
    targets = autodetect_lora_targets(model)
    info = {
        "attn": attn_impl,
        "class": cls,
        "is_multimodal": cls == "image_text_to_text",
        "lora_target_modules": targets,
    }
    if not for_training:
        model.config.use_cache = True
    return model, info


def load_model_and_tokenizer(
    model_id: str,
    attn: str = "auto",
    for_training: bool = False,
    load_in_4bit: bool = False,
):
    model, info = load_model(model_id, attn=attn, for_training=for_training,
                             load_in_4bit=load_in_4bit)
    tok = load_tokenizer(model_id)
    return model, tok, info


# --------------------------------------------------------------------------- #
# Tokenization with exact assistant-only loss masking
# --------------------------------------------------------------------------- #
def make_encoder(tokenizer, max_seq_len: int):
    """Returns fn(messages) -> dict(input_ids, labels, attention_mask, length).

    Masking strategy (the core of SFT):
      - Render the full conversation, and the conversation without the final
        assistant turn but WITH the generation prompt (`<|im_start|>assistant\\n`).
      - The prompt text is a strict character-prefix of the full text, so every
        token ending before that character offset belongs to the prompt and is
        masked with -100. Only the assistant response (plus <|im_end|>) is
        trained on.

    Examples longer than max_seq_len return empty lists and are filtered out
    upstream (keeps `datasets.map` happy with a fixed schema).

    NOTE: no "length" column is emitted — transformers' Trainer drops unknown
    columns before the collator, so the collator derives length from the
    input_ids themselves.
    """
    tok = tokenizer

    def encode(messages: list[dict[str, str]]) -> dict[str, list[int]]:
        full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tok.apply_chat_template(messages[:-1], tokenize=False,
                                              add_generation_prompt=True)

        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        if len(input_ids) > max_seq_len:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        if full_text.startswith(prompt_text):
            boundary = len(prompt_text)
            labels = []
            for tok_id, (start, end) in zip(input_ids, offsets):
                if end <= boundary or start < boundary:
                    # fully inside the prompt, or straddling the boundary
                    # (e.g. a "\n<first-word>" BPE merge): don't train on it
                    labels.append(-100)
                else:
                    labels.append(tok_id)  # assistant response (incl. <|im_end|>)
        else:
            # Rare fallback: template isn't prefix-safe. Mask by token count of
            # the prompt render (approximate, off by at most a token or two).
            prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True,
                                                 add_generation_prompt=True)
            # clamp: keep alignment AND at least one trainable token
            k = min(len(prompt_ids), len(input_ids) - 1)
            labels = [-100] * k + input_ids[k:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    return encode


@dataclass
class SFTCollator:
    """Right-pads to the longest sequence in batch (multiple of 16 for tensor cores)."""
    pad_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maxlen = max(len(f["input_ids"]) for f in features)
        maxlen = (maxlen + 15) // 16 * 16
        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = maxlen - n
            batch_ids.append(f["input_ids"] + [self.pad_id] * pad)
            batch_mask.append(f["attention_mask"] + [0] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
