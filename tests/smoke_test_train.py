#!/usr/bin/env python
"""Offline end-to-end smoke test: train a TINY random model with the REAL
pipeline components (make_encoder masking, SFTCollator, Trainer, PEFT LoRA,
save + adapter reload). No downloads: the tokenizer is trained in-script and
the model is randomly initialized GPT2.

Run this on a login node (or laptop) before burning GPU-hours:

  python tests/smoke_test_train.py            # CPU, ~1-2 minutes
  python tests/smoke_test_train.py --gpu      # tiny model on GPU instead

Requires: torch, transformers, peft, datasets (see setup/requirements.txt).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from model_utils import (  # noqa: E402
    SFTCollator,
    autodetect_lora_targets,
    make_encoder,
)

SPECIALS = ["[PAD]", "[UNK]", "<|im_start|>", "<|im_end|>"]

SAMPLE_CONVERSATIONS = [
    ("What is compound interest?",
     "Compound interest is interest earned on both the principal and previously accumulated interest."),
    ("Explain the time value of money.",
     "A dollar today is worth more than a dollar later because it can be invested to earn returns."),
    ("What does a balance sheet show?",
     "It reports assets, liabilities, and shareholders' equity at a point in time."),
    ("Define liquidity risk.",
     "Liquidity risk is the danger of being unable to buy or sell assets quickly without large price moves."),
    ("What is an IPO?",
     "An initial public offering is when a private company first sells shares to the public."),
    ("Explain diversification.",
     "Spreading investments across assets reduces exposure to any single source of risk."),
    ("What is a bond yield?",
     "The return an investor realizes on a bond, based on price and coupon payments."),
    ("Define drawdown.",
     "A drawdown is the decline from an investment peak to its subsequent trough."),
]


def build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from model_utils import CHATML_TEMPLATE

    # render actual chat text for the BPE trainer
    def render(q, a):
        return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"

    corpus = [render(q, a) for q, a in SAMPLE_CONVERSATIONS * 6]

    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=900,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    backend.train_from_iterator(corpus, trainer)

    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="<|im_end|>",
    )
    tok.chat_template = CHATML_TEMPLATE
    return tok


def build_dataset(tok, encoder):
    from datasets import Dataset

    rows = []
    for q, a in SAMPLE_CONVERSATIONS * 4:
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    ds = Dataset.from_list(rows)
    return ds.map(
        lambda ex: encoder(ex["messages"]),
        batched=False,
        remove_columns=ds.column_names,
        desc="smoke: tokenizing",
    ).filter(lambda ex: len(ex["input_ids"]) > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        Trainer,
        TrainingArguments,
    )

    print("== smoke_test_train ==")
    tok = build_tokenizer()
    print(f"[1/6] tokenizer built: vocab_size={len(tok)} (trained in-script, no downloads)")

    # --- masking sanity on the real tokenizer -------------------------------
    encoder = make_encoder(tok, max_seq_len=160)
    probe = encoder([
        {"role": "user", "content": "What is an IPO?"},
        {"role": "assistant", "content": "An initial public offering."},
    ])
    n = len(probe["input_ids"])
    n_train = sum(1 for l in probe["labels"] if l != -100)
    assert 0 < n_train < n, "masking must leave some but not all tokens trainable"
    assert len(probe["labels"]) == n
    print(f"[2/6] masking ok: {n} tokens, {n_train} trained (assistant-only)")

    # --- tiny random model ---------------------------------------------------
    cfg = GPT2Config(
        vocab_size=len(tok), n_positions=160, n_embd=64, n_layer=2, n_head=2,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    # also verify the loader path used for hub models works on this local model
    targets = autodetect_lora_targets(model)
    assert targets == "all-linear" or all(t not in ("lm_head",) for t in targets), \
        f"lm_head must never be a LoRA target, got {targets}"
    print(f"[3/6] tiny GPT2 (2 layers, 64 hidden) — auto LoRA targets: {targets}")

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                      lora_dropout=0.0, bias="none", target_modules=targets)
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()

    train_ds = build_dataset(tok, encoder)
    print(f"[4/6] dataset tokenized: {len(train_ds)} examples")

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    ta_kwargs = dict(
        output_dir=tempfile.mkdtemp(prefix="smoke_sft_"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=(device == "cpu"),
        remove_unused_columns=True,     # exercise the REAL Trainer column-dropping path
        dataloader_num_workers=0,
    )
    # transformers 5.x removed some legacy kwargs — keep only what this version accepts
    import inspect
    import transformers as _tf

    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in ta_kwargs if k not in valid)
    for k in dropped:
        ta_kwargs.pop(k)
    if dropped:
        print(f"[warn] transformers {_tf.__version__}: dropped unsupported "
              f"TrainingArguments kwargs: {dropped}")
    targs = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=SFTCollator(pad_id=tok.pad_token_id),
        processing_class=tok,
    )
    out = trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    assert all(math.isfinite(x) for x in losses), f"non-finite loss: {losses}"
    print(f"[5/6] training ok on {device}: {args.max_steps} steps, "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    # --- save + reload adapter ----------------------------------------------
    final_dir = os.path.join(targs.output_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    assert os.path.exists(os.path.join(final_dir, "adapter_model.safetensors")), \
        "LoRA adapter file missing"
    from peft import PeftModel

    base = GPT2LMHeadModel(cfg)
    PeftModel.from_pretrained(base, final_dir)
    print(f"[6/6] adapter saved + reloadable -> {final_dir}")

    print("\nSMOKE TEST PASSED — pipeline mechanics are sound. "
          "Next: real data prep, then a 1-GPU debug run before the full node.")


if __name__ == "__main__":
    main()
