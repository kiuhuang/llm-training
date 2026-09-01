#!/usr/bin/env python
"""Inspect the processed dataset (read-only, no downloads by default).

  python data_prep/inspect_data.py
  python data_prep/inspect_data.py --tokenizer Qwen/Qwen3.5-4B-Base   # adds token stats
                                                      (downloads ~15 MB of tokenizer files)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics


def percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--tokenizer", default=None,
                   help="optional HF tokenizer id for token-level stats (downloads tokenizer files)")
    p.add_argument("--examples", type=int, default=2)
    args = p.parse_args()

    enc = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for split in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        rows = [json.loads(line)["messages"] for line in open(path, encoding="utf-8")]

        user_lens, asst_lens, tot_lens = [], [], []
        n_system = 0
        for msgs in rows:
            ul = next((len(m["content"]) for m in msgs if m["role"] == "user"), 0)
            al = next((len(m["content"]) for m in msgs if m["role"] == "assistant"), 0)
            n_system += any(m["role"] == "system" for m in msgs)
            user_lens.append(ul)
            asst_lens.append(al)
            tot_lens.append(ul + al)

        user_lens_s, asst_lens_s, tot_lens_s = (sorted(v) for v in (user_lens, asst_lens, tot_lens))

        print(f"\n===== {split} ({len(rows)} rows) =====")
        print(f"  has system message : {n_system}/{len(rows)}")
        for label, vals_s, vals in [("user chars", user_lens_s, user_lens),
                                    ("assistant chars", asst_lens_s, asst_lens),
                                    ("total chars", tot_lens_s, tot_lens)]:
            if vals:
                print(f"  {label:<16s}: p50={percentile(vals_s, .5):>6,}  "
                      f"p90={percentile(vals_s, .9):>6,}  p99={percentile(vals_s, .99):>6,}  "
                      f"max={vals_s[-1]:>7,}  mean={statistics.mean(vals):>8,.0f}")

        if enc is not None:
            sample = rows[: min(1000, len(rows))]
            tok_lens = sorted(
                len(enc.apply_chat_template(m, tokenize=True)) for m in sample
            )
            over_4096 = sum(t > 4096 for t in tok_lens)
            print(f"  tokens (n={len(sample)}): p50={percentile(tok_lens, .5):,}  "
                  f"p90={percentile(tok_lens, .9):,}  p99={percentile(tok_lens, .99):,}  "
                  f"max={tok_lens[-1]:,}  |  >4096 tokens: {over_4096}")

        for i, msgs in enumerate(rows[: args.examples]):
            print(f"  --- example {i} ---")
            for m in msgs:
                text = m["content"][:220].replace("\n", " ")
                print(f"    [{m['role']:>9s}] {text}{'…' if len(m['content']) > 220 else ''}")

    stats_path = os.path.join(args.data_dir, "stats.json")
    if os.path.exists(stats_path):
        print("\n===== stats.json =====")
        print(json.dumps(json.load(open(stats_path)), indent=2)[:1200])


if __name__ == "__main__":
    main()
