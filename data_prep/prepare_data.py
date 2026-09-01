#!/usr/bin/env python
"""Prepare Finance-Instruct-500k for SFT: stream -> clean -> sample -> split.

Dataset facts (verified on the HF hub):
  - id: Josephgflowers/Finance-Instruct-500k
  - 518,432 rows, single `train` split, ~1.83 GB, license apache-2.0
  - exactly 3 string columns: `system`, `user`, `assistant`
    (system is often just "\\n"; RAG entries prepend context to `user`;
     no category column, no predefined test split)

What this script does (no model downloads — dataset only):
  1. STREAM the dataset (tries the auto-converted Parquet revision first, which
     streams properly without pulling the whole 1.8 GB JSON; falls back to
     streaming the raw JSON, then to a full download).
  2. CLEAN: strip whitespace; drop empty system prompts ("\\n"); drop rows with
     empty user/assistant; drop rows over length caps; exact-dedup.
  3. SAMPLE: reservoir-style shuffled take of train+val+test rows.
  4. SPLIT: seeded shuffle -> train / val / test (test is NEVER trained on).
  5. WRITE: messages-format JSONL for training + eval_prompts.jsonl for the
     pre/post comparison (prompt = everything before the answer, reference =
     the held-out answer).

Usage:
  python data_prep/prepare_data.py                          # 30k train default
  python data_prep/prepare_data.py --train-rows 5000        # quicker first run
  python data_prep/prepare_data.py --no-streaming           # full 1.8 GB download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup"))
from envcfg import load_repo_dotenv  # noqa: E402

DEFAULT_DATASET = "Josephgflowers/Finance-Instruct-500k"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare finance SFT dataset")
    p.add_argument("--dataset", default=None,
                   help=f"defaults to DATASET_ID from .env or {DEFAULT_DATASET}")
    p.add_argument("--train-rows", type=int, default=30000)
    p.add_argument("--val-rows", type=int, default=500)
    p.add_argument("--test-rows", type=int, default=500)
    p.add_argument("--eval-prompts", type=int, default=300,
                   help="how many test rows to also export as eval prompts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--max-total-chars", type=int, default=100_000,
                   help="drop rows whose user+assistant text exceeds this (RAG rows can be huge)")
    p.add_argument("--min-assistant-chars", type=int, default=1)
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--no-streaming", action="store_true", help="download the full dataset instead")
    return p.parse_args()


def open_stream(args):
    """Return an iterable of raw rows, printing which load path worked."""
    from datasets import load_dataset

    if args.no_streaming:
        print(f"[load] full download of {args.dataset} (streaming disabled)")
        return load_dataset(args.dataset, split="train")

    # 1) auto-converted parquet — real streaming, small memory footprint
    try:
        ds = load_dataset(args.dataset, revision="refs/convert/parquet",
                          split="train", streaming=True)
        next(iter(ds))  # force one item to validate
        print("[load] streaming via refs/convert/parquet (recommended path)")
        return ds
    except Exception as e:  # noqa: BLE001
        print(f"[load] parquet revision failed ({type(e).__name__}: {str(e)[:100]})")

    # 2) native streaming of the hub JSON
    try:
        ds = load_dataset(args.dataset, split="train", streaming=True)
        next(iter(ds))
        print("[load] streaming the raw hub file (may download most of it anyway)")
        return ds
    except Exception as e:  # noqa: BLE001
        print(f"[load] native streaming failed too ({type(e).__name__}); full download fallback")
        return load_dataset(args.dataset, split="train")


def normalize(row: dict) -> tuple[list[dict] | None, str]:
    """Return (messages, drop_reason). messages=None means drop the row."""
    system = (row.get("system") or "").strip()
    user = (row.get("user") or "").strip()
    assistant = (row.get("assistant") or "").strip()

    if not user:
        return None, "empty_user"
    if not assistant:
        return None, "empty_assistant"

    messages = []
    if system:  # system is often just "\n" -> stripped to "" -> dropped
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})

    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > 10_000_000:  # corrupt guard
        return None, "absurd_length"
    return messages, ""


def main() -> None:
    load_repo_dotenv(__file__)  # optional repo .env: HF_TOKEN/HF_HOME etc.
    args = parse_args()
    args.dataset = args.dataset or os.environ.get("DATASET_ID", DEFAULT_DATASET)
    os.makedirs(args.out_dir, exist_ok=True)
    eval_dir = os.path.join("outputs", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    n_want = args.train_rows + args.val_rows + args.test_rows
    stream = open_stream(args)
    # IterableDataset (streaming): buffer_size gives an approximate shuffle.
    # A full Dataset (--no-streaming / fallback): TRUE global shuffle, and
    # Dataset.shuffle() does NOT accept buffer_size.
    import datasets as hf_datasets

    if isinstance(stream, hf_datasets.IterableDataset):
        stream = stream.shuffle(buffer_size=min(50_000, max(10_000, n_want)), seed=args.seed)
    else:
        stream = stream.shuffle(seed=args.seed)

    kept: list[list[dict]] = []
    drops: Counter[str] = Counter()
    seen: set[str] = set()
    scanned = 0

    print(f"[scan] target rows: {n_want} (train {args.train_rows} / val {args.val_rows} / test {args.test_rows})")
    for row in stream:
        scanned += 1
        messages, reason = normalize(row)
        if messages is None:
            drops[reason or "unknown"] += 1
        elif not args.no_dedup:
            key = hashlib.sha1(
                (messages[-2]["content"] + "\x00" + messages[-1]["content"]).encode()
            ).hexdigest()
            if key in seen:
                drops["duplicate"] += 1
            else:
                seen.add(key)
                kept.append(messages)
        else:
            kept.append(messages)

        if len(kept) >= n_want:
            break
        if scanned % 50_000 == 0:
            print(f"[scan] {scanned:>7,} rows scanned, {len(kept):>7,} kept, "
                  f"drops: {dict(drops)}")

    if len(kept) < n_want:
        print(f"[warn] stream ended early: only {len(kept)}/{n_want} rows kept")

    rng = random.Random(args.seed)
    rng.shuffle(kept)

    n_val, n_test = args.val_rows, args.test_rows
    n_train = max(0, len(kept) - n_val - n_test)
    splits = {
        "train": kept[:n_train],
        "val": kept[n_train:n_train + n_val],
        "test": kept[n_train + n_val:],
    }

    for name, rows in splits.items():
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for msgs in rows:
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        print(f"[write] {path}: {len(rows)} rows")

    # ---- eval prompts from the held-out test split --------------------------
    eval_rows = splits["test"][:args.eval_prompts]
    with open(os.path.join(eval_dir, "eval_prompts.jsonl"), "w", encoding="utf-8") as f:
        for i, msgs in enumerate(eval_rows):
            f.write(json.dumps({
                "id": i,
                "messages": msgs[:-1],                 # prompt only (no answer)
                "reference": msgs[-1]["content"],      # held-out reference answer
            }, ensure_ascii=False) + "\n")
    print(f"[write] outputs/eval/eval_prompts.jsonl: {len(eval_rows)} prompts "
          f"(from the held-out test split)")

    all_chars = [sum(len(m["content"]) for m in msgs) for msgs in kept]
    srt = sorted(all_chars)
    stats = {
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "rows_scanned": scanned,
        "kept": len(kept),
        "dropped": dict(drops),
        "splits": {k: len(v) for k, v in splits.items()},
        "char_length_p50_p90_p99_max": (
            [srt[int(0.5 * len(srt))], srt[int(0.9 * len(srt))],
             srt[int(0.99 * len(srt))], srt[-1]] if srt else []
        ),
        "notes": "system='\\n' rows dropped the system message; test split never used for training",
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[write] data/processed/stats.json")
    print(json.dumps({k: v for k, v in stats.items() if k != "notes"}, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
