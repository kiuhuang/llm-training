# 06 — Worked-example model: Qwen/Qwen3.5-4B-Base (verified 2026-04)

All facts below were read from the HF repo (model card, `config.json`,
file tree) for the model this repo was first run with. Swapping models is
expected and mostly automatic (see README "Swapping the model or dataset");
the version lesson at the bottom is the generalizable part.

| property | value |
|---|---|
| repo | `Qwen/Qwen3.5-4B-Base` (Apache-2.0) |
| architecture | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` |
| composition | **hybrid**: 32 layers = 8 × (3 × Gated DeltaNet linear-attention → FFN, 1 × gated full-attention → FFN) — 24 linear + 8 full |
| LM params | 4B (model card); ~5B total incl. vision encoder (depth 24, hidden 1024) |
| extra | 1 MTP (multi-token prediction) layer |
| hidden | 2560, FFN 9216, SiLU, RMSNorm (eps 1e-6) |
| full-attn | GQA: 16 Q heads, 4 KV heads, head_dim 256, attn output gate |
| DeltaNet | 32 V / 16 QK heads, head_dim 128, conv kernel 4, SSM dtype fp32 |
| context | 262,144 native (card: extensible to ~1M) |
| RoPE | theta 1e7, partial rotary 0.25 (64 of 256 dims), interleaved mrope |
| tokenizer | BPE, vocab 248,320, **chat template included** |
| weights | 2 shards, ~9.32 GB, bf16 (+ fp32 SSM tensors) |
| transformers | **≥ 5.16.1 required** — the `qwen3_5` architecture ships in the 5.x line; 4.57.x does NOT recognize it (the "4.57.0.dev0" string in config.json is the exporter's version, not a support guarantee) |

## Consequences for this repo

1. **Multimodal loading.** Some transformers versions expose Qwen3.5 only via
   `AutoModelForImageTextToText`. `scripts/model_utils.py` tries
   `AutoModelForCausalLM` first and falls back — either way text-only SFT
   works and the vision tower simply stays unused.
2. **LoRA targets.** `autodetect_lora_targets()` inspects the module tree,
   excludes the vision tower and `lm_head`, and targets the classic
   q/k/v/o/gate/up/down projections of the language stack. Auto-detection
   (not a hard-coded list) keeps this correct even if naming differs.
3. **Memory.** The checkpoint is ~9.3 GB bf16 — plan ~12-15 GB/GPU just for
   weights in DDP (replicated), and note ZeRO-3 shards them if needed.
4. **Base ≠ instruct.** The model ships a chat template but was *not*
   instruction-tuned — exactly why SFT on finance instructions should produce
   a dramatic, visible behavior change (see docs/04_evaluation.md).
5. **DeltaNet layers** don't use flash-attention; flash-attn only affects the
   8 full-attention layers. Missing flash-attn is a minor speed loss, not a
   blocker (`attn: auto` falls back to SDPA).

## Reference implementation comparison

The Microsoft/Databricks tutorial (`docs/reference/`) fine-tunes the older
text-only `Qwen/Qwen3-4B` with TRL `SFTTrainer`, full parameters, effective
batch 8, lr 2e-5, 50 demo steps, MLflow logging, eval_loss-only validation on
1×H100. This repo keeps the same *family* of techniques (HF stack, bf16,
gradient checkpointing, best-checkpoint-by-eval_loss) but targets the newer
Qwen3.5-4B-Base, adds LoRA/QLoRA paths, multi-GPU H800 launchers, the finance
dataset pipeline, and generation-based pre/post evaluation.
