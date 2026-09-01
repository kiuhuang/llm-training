#!/usr/bin/env python
"""Merge a LoRA adapter into its base model -> standalone model directory.

Do this AFTER training if you want a single self-contained model for
inference/serving (instead of carrying base + adapter around). Skip it if you
apply adapters on the fly (run_inference.py --adapter).

  python scripts/merge_lora.py --base_model Qwen/Qwen3.5-4B-Base \
      --adapter outputs/lora_v1/final --out_dir outputs/merged_v1
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, load_tokenizer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="hub id or local dir the adapter was trained on")
    p.add_argument("--adapter", required=True, help="LoRA adapter dir (contains adapter_model.safetensors)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    from peft import PeftModel

    device_map = {"auto": "auto", "cpu": "cpu", "cuda": "cuda:0"}[args.device]
    print(f"[load] base: {args.base_model}  adapter: {args.adapter}  device_map={device_map}")
    model, info = load_model(args.base_model, attn="sdpa", device_map=device_map)
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[merge] merging adapters into base weights ...")
    model = model.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)

    tok = load_tokenizer(args.adapter)  # adapter dir contains the tokenizer too
    tok.save_pretrained(args.out_dir)

    n_bytes = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir) if f.endswith(".safetensors")
    )
    print(f"[done] merged model -> {args.out_dir}  ({n_bytes / 1e9:.2f} GB of weights) "
          f"(class={info['class']})")


if __name__ == "__main__":
    main()
