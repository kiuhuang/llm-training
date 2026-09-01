#!/usr/bin/env python
"""Held-out perplexity / NLL of a model on test.jsonl (training objective).

Answers the question "did fine-tuning move the model toward the target
distribution?" on the SAME masked-prompt objective used in training:
only assistant tokens contribute to the loss.

  python scripts/perplexity.py --model Qwen/Qwen3.5-4B-Base --tag base
  python scripts/perplexity.py --model Qwen/Qwen3.5-4B-Base \
      --adapter outputs/lora_v1/final --tag tuned

Results are appended to outputs/eval/perplexity.json for side-by-side reading.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup"))
from envcfg import load_repo_dotenv  # noqa: E402

load_repo_dotenv(__file__)  # optional repo .env: HF_HOME/HF_TOKEN etc.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import SFTCollator, load_model, load_tokenizer, make_encoder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--tag", default=None, help="label stored in the results file")
    p.add_argument("--results-file", default="outputs/eval/perplexity.json")
    args = p.parse_args()

    tok = load_tokenizer(args.model)
    model, info = load_model(args.model, for_training=False)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    device = next(model.parameters()).device

    encoder = make_encoder(tok, args.max_seq_len)
    rows = [json.loads(line)["messages"] for line in open(args.data, encoding="utf-8")]
    rows = rows[: args.max_samples]
    feats = [encoder(msgs) for msgs in rows]
    feats = [f for f in feats if len(f["input_ids"]) > 0]
    print(f"[data] {len(feats)} usable examples from {args.data} (max_seq_len={args.max_seq_len})")

    collate = SFTCollator(pad_id=tok.pad_token_id)
    nll_sum, n_tok = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(feats), args.batch_size):
            batch = collate(feats[i: i + args.batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n_labels = int((batch["labels"] != -100).sum())
            nll_sum += float(out.loss) * n_labels
            n_tok += n_labels
            if (i // args.batch_size) % 50 == 0:
                print(f"  {i + args.batch_size}/{len(feats)}  running ppl="
                      f"{math.exp(min(nll_sum / max(n_tok, 1), 20)):.3f}")

    avg_nll = nll_sum / max(n_tok, 1)
    result = {
        "tag": args.tag or os.path.basename(args.model.rstrip("/")),
        "model": args.model + (f"+{args.adapter}" if args.adapter else ""),
        "data": args.data,
        "n_examples": len(feats),
        "n_tokens": n_tok,
        "nll_per_token": avg_nll,
        "perplexity": math.exp(avg_nll),
        "bits_per_token": avg_nll / math.log(2),
    }
    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    results = []
    if os.path.exists(args.results_file):
        with open(args.results_file) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    results = [r for r in results if r.get("tag") != result["tag"]] + [result]
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[write] {args.results_file}")


if __name__ == "__main__":
    main()
