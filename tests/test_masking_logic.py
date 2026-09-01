#!/usr/bin/env python
"""Offline unit tests for the assistant-only loss masking logic.

Uses a deterministic stub tokenizer (no downloads, no torch) that mimics the
API surface used by model_utils.make_encoder:
  - apply_chat_template(messages, tokenize=..., add_generation_prompt=...)
  - __call__(text, add_special_tokens=False, return_offsets_mapping=True)

Run:  python tests/test_masking_logic.py     (exit 0 = all pass)
"""
from __future__ import annotations

import os
import sys

# model_utils imports torch at module level; make_encoder itself never uses it.
# Stub it when absent so this test runs on any machine with plain Python.
try:
    import torch  # noqa: F401
except ImportError:
    import types
    _fake = types.ModuleType("torch")
    _fake.nn = types.SimpleNamespace(Linear=object)
    sys.modules["torch"] = _fake

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from model_utils import make_encoder  # noqa: E402

SPECIALS = ["<|im_start|>", "<|im_end|>"]
CHUNK = 3  # stub "BPE": every 3 ordinary chars are one token


class StubTokenizer:
    def __init__(self):
        self.pad_token = type("T", (), {"id": 0})()
        self.vocab: dict[str, int] = {}

    # -- rendering (mirrors CHATML_TEMPLATE exactly) -------------------------
    def render(self, messages, add_generation_prompt=False):
        out = ""
        for m in messages:
            out += "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>" + "\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    # -- transformers-like API ----------------------------------------------
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = self.render(messages, add_generation_prompt)
        if not tokenize:
            return text
        return [tid for tid, _, _ in self._tokens(text)]

    def _tokens(self, text):
        """List of (piece, start, end). Specials are single tokens; other text
        is chopped into CHUNK-char tokens (offsets are character-based)."""
        toks: list[tuple[str, int, int]] = []
        i, buf, buf_start = 0, "", -1

        def flush():
            nonlocal buf, buf_start
            j = 0
            while j < len(buf):
                piece = buf[j: j + CHUNK]
                toks.append((piece, buf_start + j, buf_start + j + len(piece)))
                j += CHUNK
            buf, buf_start = "", -1

        while i < len(text):
            for sp in SPECIALS:
                if text.startswith(sp, i):
                    flush()
                    toks.append((sp, i, i + len(sp)))
                    i += len(sp)
                    break
            else:
                if buf_start < 0:
                    buf_start = i
                buf += text[i]
                i += 1
        flush()
        return toks

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        toks = self._tokens(text)
        ids = []
        for piece, _, _ in toks:
            if piece not in self.vocab:
                self.vocab[piece] = 100 + (sum(map(ord, piece)) % 900)
            ids.append(self.vocab[piece])
        return {"input_ids": ids, "offset_mapping": [(s, e) for _, s, e in toks]}


def make_stub_with_broken_prefix():
    """Stub whose prompt render is NOT a char-prefix of the full render
    (simulates a template that isn't prefix-safe) to exercise the fallback."""
    class Broken(StubTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt and len(messages) and messages[-1]["role"] != "assistant":
                # deliberately different from the full render -> breaks prefix property
                text = "<|im_start|>user\nXX<|im_end|>\n<|im_start|>assistant\n"
                return text if not tokenize else \
                    self(text, add_special_tokens=False)["input_ids"]
            return super().apply_chat_template(messages, tokenize, add_generation_prompt)
    return Broken()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_basic_masking():
    print("test_basic_masking")
    tok = StubTokenizer()
    enc = make_encoder(tok, max_seq_len=512)
    messages = [
        {"role": "system", "content": "You are a finance tutor."},
        {"role": "user", "content": "What is duration?"},
        {"role": "assistant", "content": "Duration measures interest rate risk."},
    ]
    out = enc(messages)
    full = tok.render(messages)
    ids, labels, offs = out["input_ids"], out["labels"], None
    check(len(ids) == len(labels), "labels align with input_ids")
    check(len(ids) > 0, "non-empty encoding")

    # recompute offsets for verification
    offs = tok(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    boundary = len(tok.render(messages[:-1], add_generation_prompt=True))
    check(full.startswith(tok.render(messages[:-1], add_generation_prompt=True)),
          "prompt render is a char-prefix of full render")

    first_unmasked = next(i for i, l in enumerate(labels) if l != -100)
    check(all(l == -100 for l in labels[:first_unmasked]),
          "everything before first response token is masked")
    check(all(l == -100 for i, l in enumerate(labels)
              if offs[i][1] <= boundary), "tokens fully inside prompt are -100")
    check(all(labels[i] == ids[i] for i in range(first_unmasked, len(ids))),
          "unmasked region trains on the true token ids")
    check(first_unmasked <= boundary, "response training starts at/after the boundary")
    # the last tokens (<|im_end|> + newline) must be trained
    check(labels[-2] != -100 and labels[-1] != -100, "closing <|im_end|> is trained")


def test_boundary_straddle_masked():
    print("test_boundary_straddle_masked")
    tok = StubTokenizer()
    enc = make_encoder(tok, max_seq_len=512)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there, investor."},
    ]
    out = enc(messages)
    offs = tok(tok.render(messages), add_special_tokens=False,
               return_offsets_mapping=True)["offset_mapping"]
    boundary = len(tok.render(messages[:-1], add_generation_prompt=True))
    straddling = [i for i, (s, e) in enumerate(offs) if s < boundary < e]
    for i in straddling:
        check(out["labels"][i] == -100, f"straddling token {i} is masked (conservative)")


def test_max_len_drop():
    print("test_max_len_drop")
    tok = StubTokenizer()
    enc = make_encoder(tok, max_seq_len=5)
    out = enc([{"role": "user", "content": "x" * 100},
               {"role": "assistant", "content": "y" * 100}])
    check(out["input_ids"] == [], "over-length example returns empty lists (filtered upstream)")


def test_no_system_message():
    print("test_no_system_message")
    tok = StubTokenizer()
    enc = make_encoder(tok, max_seq_len=512)
    out = enc([{"role": "user", "content": "Define VaR."},
               {"role": "assistant", "content": "Value at Risk estimates potential loss."}])
    n_unmasked = sum(1 for l in out["labels"] if l != -100)
    check(n_unmasked > 0, "user+assistant-only conversation produces trained tokens")


def test_fallback_broken_prefix():
    print("test_fallback_broken_prefix")
    tok = make_stub_with_broken_prefix()
    enc = make_encoder(tok, max_seq_len=512)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Response text here."},
    ]
    out = enc(messages)
    prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    check(len(out["labels"]) == len(out["input_ids"]), "fallback keeps alignment")
    check(all(l == -100 for l in out["labels"][: len(prompt_ids)]),
          "fallback masks ~prompt-length prefix")
    check(any(l != -100 for l in out["labels"][len(prompt_ids):]),
          "fallback still trains on the tail")


if __name__ == "__main__":
    test_basic_masking()
    test_boundary_straddle_masked()
    test_max_len_drop()
    test_no_system_message()
    test_fallback_broken_prefix()
    print("\nALL MASKING TESTS PASSED")
