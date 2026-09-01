#!/usr/bin/env python
"""Supervised fine-tuning (SFT) for causal LLMs — LoRA / QLoRA / full-parameter.

Design goals
------------
* Transparent: loss masking (prompt tokens -> -100) is done explicitly in
  model_utils.make_encoder so you can *see* what the model learns to predict.
* Version-stable: only `transformers.Trainer` + `peft` — no TRL API churn.
* Single-node multi-GPU (DGX H800 = 8x H800 80GB) via torchrun + DDP,
  optional DeepSpeed ZeRO-2/3 for full-parameter runs.
* Handles the Qwen3.5 quirk: the model is multimodal; LoRA targets are
  auto-detected on the language stack only (vision tower untouched).

Usage
-----
  # LoRA on 8 GPUs (from repo root)
  torchrun --standalone --nproc_per_node=8 scripts/train_sft.py \
      --config configs/train_lora.yaml

  # Full-parameter SFT with ZeRO-2
  bash scripts/train_full.sh

  # Tiny debug run, 1 GPU, 200 samples, 1 epoch
  python scripts/train_sft.py --config configs/train_lora.yaml \
      --max_samples 200 --epochs 1

Dataset contract: JSONL, one object per line, each with a "messages" key:
  {"messages": [{"role": "system", "content": "..."},      # optional
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}
Produced by data_prep/prepare_data.py. Any CLI flag overrides the YAML value.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "setup"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envcfg import load_repo_dotenv  # noqa: E402

load_repo_dotenv(__file__)  # optional repo .env: HF_HOME/HF_TOKEN etc.
from model_utils import (  # noqa: E402
    SFTCollator,
    load_model,
    load_tokenizer,
    make_encoder,
)


# --------------------------------------------------------------------------- #
# Config handling: YAML defaults <- CLI overrides
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training (LoRA/QLoRA/full)")
    p.add_argument("--config", type=str, default=None, help="YAML file with defaults")

    p.add_argument("--mode", type=str, default=None, choices=["lora", "full"])
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, help="train JSONL (messages format)")
    p.add_argument("--eval_dataset", type=str, default=None, help="validation JSONL (recommended)")
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--per_device_batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument("--lora_target_modules", type=str, default=None,
                   help="comma-separated module names; default = auto-detected")
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA: nf4 4-bit base weights")

    p.add_argument("--attn", type=str, default=None,
                   choices=["auto", "flash_attention_2", "sdpa", "eager"])
    p.add_argument("--deepspeed", type=str, default=None, help="path to DeepSpeed JSON config")
    p.add_argument("--max_samples", type=int, default=None, help="cap train set size (debug)")
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--save_only_model", action="store_true",
                   help="skip optimizer states in checkpoints (much smaller, cannot resume)")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg.update(cli)

    # defaults for anything still unset
    cfg.setdefault("mode", "lora")
    cfg.setdefault("max_seq_len", 4096)
    cfg.setdefault("per_device_batch_size", 2)
    cfg.setdefault("grad_accum", 8)
    cfg.setdefault("epochs", 2.0)
    cfg.setdefault("warmup_ratio", 0.03)
    cfg.setdefault("seed", 42)
    cfg.setdefault("lora_r", 16)
    cfg.setdefault("lora_alpha", 32)
    cfg.setdefault("lora_dropout", 0.05)
    cfg.setdefault("attn", "auto")
    cfg.setdefault("logging_steps", 10)
    cfg.setdefault("save_steps", 500)
    cfg.setdefault("eval_steps", 250)
    cfg.setdefault("weight_decay", 0.0)
    if "learning_rate" not in cfg:
        cfg["learning_rate"] = 1e-4 if cfg["mode"] == "lora" else 1e-5
    return cfg


class PerplexityLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics and metrics["eval_loss"] < 20:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    set_seed(int(cfg["seed"]))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print("=" * 70)
        print("Effective config:")
        for k in sorted(cfg):
            print(f"  {k:26s} = {cfg[k]}")
        print("=" * 70)

    # ---- model -------------------------------------------------------------
    model, info = load_model(
        cfg["model_name_or_path"],
        attn=cfg["attn"],
        for_training=True,
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
    )
    tokenizer = load_tokenizer(cfg["model_name_or_path"])
    if local_rank == 0:
        print(f"[model] class={info['class']} attn={info['attn']} "
              f"multimodal={info['is_multimodal']}")

    if cfg["mode"] == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        targets = cfg.get("lora_target_modules")
        if isinstance(targets, str) and targets != "all-linear":
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        if not targets:  # not set anywhere -> use auto-detected
            targets = info["lora_target_modules"]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias="none",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)
        # required so gradient checkpointing works through frozen base weights
        model.enable_input_require_grads()
        if local_rank == 0:
            model.print_trainable_parameters()
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    model.config.use_cache = False  # incompatible with gradient checkpointing

    # ---- datasets ----------------------------------------------------------
    encoder = make_encoder(tokenizer, int(cfg["max_seq_len"]))
    files = {"train": cfg["dataset"]}
    if cfg.get("eval_dataset"):
        files["validation"] = cfg["eval_dataset"]
    raw = load_dataset("json", data_files=files)

    tokenized = {}
    for split, ds in raw.items():
        if split == "train" and cfg.get("max_samples"):
            ds = ds.select(range(min(int(cfg["max_samples"]), len(ds))))
        if split == "validation" and cfg.get("max_eval_samples"):
            ds = ds.select(range(min(int(cfg["max_eval_samples"]), len(ds))))
        tokenized[split] = ds.map(
            lambda ex: encoder(ex["messages"]),
            batched=False,
            num_proc=8,
            remove_columns=ds.column_names,
            desc=f"Tokenizing + masking ({split})",
        )

    train_ds = tokenized["train"].filter(lambda ex: len(ex["input_ids"]) > 0, num_proc=8)
    eval_ds = tokenized.get("validation")
    if eval_ds is not None:
        eval_ds = eval_ds.filter(lambda ex: len(ex["input_ids"]) > 0, num_proc=8)

    if local_rank == 0:
        lens = [len(x) for x in train_ds["input_ids"]]
        print(f"train examples kept: {len(train_ds)}  "
              f"tokens: mean={sum(lens) / len(lens):.0f} max={max(lens)}")
        if eval_ds:
            elens = [len(x) for x in eval_ds["input_ids"]]
            print(f"eval examples kept: {len(eval_ds)}  "
                  f"tokens: mean={sum(elens) / len(elens):.0f}")

    # ---- trainer -----------------------------------------------------------
    # keep the checkpoint with the lowest eval loss (same idea as the reference
    # Databricks tutorial's metric_for_best_model="eval_loss")
    best_kwargs: dict[str, Any] = {}
    if eval_ds is not None:
        best_kwargs = dict(
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        # load_best_model_at_end requires save cadence to be a multiple of eval cadence
        if int(cfg["save_steps"]) % int(cfg["eval_steps"]) != 0:
            cfg["save_steps"] = cfg["eval_steps"]

    ta_kwargs: dict[str, Any] = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=int(cfg["per_device_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        num_train_epochs=float(cfg["epochs"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(cfg["warmup_ratio"]),
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        logging_steps=int(cfg["logging_steps"]),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=3,
        save_only_model=bool(cfg.get("save_only_model", False)),
        save_safetensors=True,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=int(cfg["eval_steps"]) if eval_ds else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        seed=int(cfg["seed"]),
        deepspeed=cfg.get("deepspeed") or None,
        # throughput: batches of similar length kill padding waste (mean 247 vs
        # max 4096 tokens); DDP: LoRA adapters on used modules only, so the
        # find-unused-parameters graph walk is pure overhead (5.x warns about it)
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )
    ta_kwargs.update(best_kwargs)

    # transformers 5.x removed/renamed legacy TrainingArguments kwargs (first
    # casualty: overwrite_output_dir). Keep only what THIS version accepts, and
    # say so loudly if something was dropped.
    import inspect
    import transformers as _tf

    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in ta_kwargs if k not in valid)
    for k in dropped:
        ta_kwargs.pop(k)
    if dropped and local_rank == 0:
        print(f"[warn] transformers {_tf.__version__}: dropped unsupported "
              f"TrainingArguments kwargs: {dropped}")
        if "warmup_ratio" in dropped:
            print("[warn] 5.x removed warmup args — training runs WITHOUT warmup. "
                  "Negligible for LoRA at 1e-4 over ~500 steps; revisit if you "
                  "train full-parameter or with a larger LR.")

    targs = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SFTCollator(pad_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[PerplexityLogger()],
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    if local_rank == 0:
        print(f"training wall time: {(time.time() - t0) / 60:.1f} min")

    # ---- save final --------------------------------------------------------
    if local_rank == 0:
        final_dir = os.path.join(cfg["output_dir"], "final")
        trainer.save_model(final_dir)          # LoRA: adapter only; full: full model
        tokenizer.save_pretrained(final_dir)
        summary = {
            "model": cfg["model_name_or_path"],
            "mode": cfg["mode"],
            "model_class": info["class"],
            "lora_target_modules": info["lora_target_modules"] if cfg["mode"] == "lora" else None,
            "train_examples": len(train_ds),
            "eval_examples": len(eval_ds) if eval_ds else 0,
            "final_train_loss": trainer.state.log_history[-1].get("train_loss"),
            "best_eval_loss": min(
                (m.get("eval_loss", float("inf")) for m in trainer.state.log_history),
                default=None,
            ),
            "runtime_minutes": (time.time() - t0) / 60,
            "config": dict(cfg),
        }
        with open(os.path.join(cfg["output_dir"], "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[done] artifacts in {cfg['output_dir']}")


if __name__ == "__main__":
    main()
