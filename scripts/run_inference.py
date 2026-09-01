#!/usr/bin/env python
"""Batch generation for the pre/post comparison: run a model over eval prompts.

Works with:
  * a hub model id (e.g. Qwen/Qwen3.5-4B-Base)              -> baseline answers
  * a merged model directory (outputs/merged_v1)             -> fine-tuned answers
  * a hub/local base + --adapter (LoRA dir)                  -> fine-tuned answers
    (no merge needed — the adapter is applied on the fly)

  # baseline
  python scripts/run_inference.py --model Qwen/Qwen3.5-4B-Base --out outputs/eval/base.jsonl

  # fine-tuned (adapter applied on the fly)
  python scripts/run_inference.py --model Qwen/Qwen3.5-4B-Base \
      --adapter outputs/lora_v1/final --out outputs/eval/tuned.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup"))
from envcfg import load_repo_dotenv  # noqa: E402

load_repo_dotenv(__file__)  # optional repo .env: HF_HOME/HF_TOKEN etc.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, load_tokenizer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="hub id or local model dir")
    p.add_argument("--adapter", default=None, help="optional LoRA adapter dir to apply")
    p.add_argument("--prompts", default="outputs/eval/eval_prompts.jsonl")
    p.add_argument("--out", required=True, help="output JSONL of predictions")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy decoding (deterministic — required for fair A/B)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None, help="only first N prompts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.model)
    tok.padding_side = "left"  # required for correct batched generation
    model, info = load_model(args.model, for_training=False)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[model] applied adapter from {args.adapter}")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[model] {type(model).__name__} on {device} (attn={info['attn']})")

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"[data] {len(prompts)} prompts from {args.prompts}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    greedy = args.temperature <= 0.0
    # Stop generation at ANY end-of-turn marker. The base model's generation
    # config eos is <endoftext>, but the chat template ends turns with
    # <|im_end|> — without both, the model answers correctly and then keeps
    # inventing follow-up turns until the token cap (poisoning every metric).
    eos_ids = []
    vocab = tok.get_vocab()
    for cand in (tok.eos_token, "<|im_end|>", "<|endoftext|>"):
        if cand and cand in vocab and vocab[cand] not in eos_ids:
            eos_ids.append(vocab[cand])
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not greedy,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids or None,
    )
    if not greedy:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    if len(eos_ids) > 1:
        print(f"[info] stopping at any of {eos_ids} (eos + chat <|im_end|> markers)")

    t_start = time.time()
    with open(args.out, "w", encoding="utf-8") as fout, torch.inference_mode():
        for b0 in range(0, len(prompts), args.batch_size):
            batch = prompts[b0: b0 + args.batch_size]
            texts = [
                tok.apply_chat_template(ex["messages"], tokenize=False,
                                        add_generation_prompt=True)
                for ex in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            input_len = enc["input_ids"].shape[1]
            out_ids = model.generate(**enc, **gen_kwargs)
            new_ids = out_ids[:, input_len:]

            for ex, ids in zip(batch, new_ids):
                n_new = int((ids != tok.pad_token_id).sum())
                pred = tok.decode(ids, skip_special_tokens=True).strip()
                fout.write(json.dumps({
                    "id": ex["id"],
                    "messages": ex["messages"],
                    "reference": ex.get("reference", ""),
                    "prediction": pred,
                    "new_tokens": n_new,
                    "truncated": n_new >= args.max_new_tokens,
                    "model": args.model + (f"+{args.adapter}" if args.adapter else ""),
                    "temperature": args.temperature,
                }, ensure_ascii=False) + "\n")
            fout.flush()

            done = min(b0 + args.batch_size, len(prompts))
            if done % (args.batch_size * 5) == 0 or done == len(prompts):
                rate = done / max(time.time() - t_start, 1e-6)
                print(f"  {done:>5}/{len(prompts)}  ({rate:.2f} prompts/s)")

    print(f"[done] predictions -> {args.out}  "
          f"({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    sys.exit(main())
