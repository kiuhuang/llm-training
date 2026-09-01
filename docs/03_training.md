# 03 — Training: what actually happens, and why these hyperparameters

## 1. The SFT objective (and the loss mask)

Supervised fine-tuning = next-token prediction on the assistant's response,
with the prompt tokens *masked out* (`labels = -100`). Concretely, for

```
<|im_start|>user\nWhat is duration?<|im_end|>\n<|im_start|>assistant\nDuration measures ...<|im_end|>\n
```

only the tokens after `<|im_start|>assistant\n` (including the closing
`<|im_end|>`) contribute loss. Masking the prompt matters: without it the model
wastes capacity learning to *parrot questions* instead of *answering them*.

Implementation: `scripts/model_utils.py::make_encoder` renders the full
conversation and the prompt-only conversation, uses the character offset of
the boundary, and masks every token that ends before it (tokens straddling the
boundary — BPE merges like `"\nDuration"` — are conservatively masked too).
`tests/test_masking_logic.py` proves the invariant with a stub tokenizer.

## 2. LoRA in one paragraph

Instead of updating all W ∈ R^{d×k}, LoRA learns W + BA where B ∈ R^{d×r},
A ∈ R^{r×k}, r ≪ min(d,k). For a 4B model with r=16 on all attention+MLP
projections, trainable params are ~0.1-0.5% of the model:
- adapter weights: ~tens of MB (vs 9.3 GB for the full model)
- optimizer states: only for adapters
- base weights: frozen → no weight gradients, no fp32 master copies

`alpha/r` scales the update (α=32, r=16 → effective scale 2.0). Higher r =
more capacity for style/domain shift; for a 4B model and 30k examples, r=16-64
is the sensible range. Target modules default to the 7 linear projections of
the *language* stack (q/k/v/o/gate/up/down) — the vision tower is excluded
automatically because Qwen3.5-4B-Base is multimodal.

## 3. Full-parameter SFT (advanced)

The reference Databricks tutorial does full FT of Qwen3-4B on 1×H100 with
effective batch 8, lr 2e-5, 50 demo steps. Full FT of a ~5B model needs ~16-18
bytes/param (bf16 weights + grads + fp32 AdamW states) ≈ 80-90 GB → does not
fit one 80 GB H800. Options: `scripts/train_full.sh` runs DeepSpeed ZeRO-2
(optimizer+grads sharded over 8 GPUs; weights replicated ~10 GB/GPU). Use
`ZERO=3` if still OOM. Full FT risks catastrophic forgetting — watch eval_loss
and prefer more LoRA experiments first.

## 4. Hyperparameters (LoRA config as shipped)

| param | value | why |
|---|---|---|
| lr | 1e-4 (cosine, 3% warmup) | standard for r=16 LoRA; full FT would use 1e-5 |
| global batch | 128 (4 per-GPU × grad_accum 4 × 8 GPUs) | stable gradients, 234 steps/epoch at 30k rows |
| epochs | 2 | SFT typically 1-3; more → memorization |
| max_seq_len | 4096 | check `inspect_data.py` p99; over-long rows are dropped |
| precision | bf16 + tf32 | H800 native; fp16 risks overflow |
| grad checkpointing | on (use_reentrant=False) | ~30% slower, huge activation memory win |
| attention | flash-attn 2 if installed, else SDPA | SDPA is fine on H800 |

Throughput sanity (do the arithmetic before launching):
`6 × params × tokens` FLOPs per token ≈ 6 × 5e9. At a conservative 30% MFU on
one H800 (~300 TFLOPS effective) that's ~10k tokens/s/GPU. Measured on the
real data: mean ≈ 171 tokens/example (see `make inspect`), so 30k examples ×
2 epochs ≈ 10.3M tokens → ~5-10 min of pure compute on 8 GPUs;
expect 1-2 h wall-clock including data loading and eval. **Measure**
`s/it` from the logs, don't trust any estimate blindly.
For the FULL ~518k dataset × 2 epochs (~177M tokens): ~40-60 min of pure
compute → roughly 1.5-3 h wall-clock — very tractable, see `make data-full`.

## 5. What the logs mean

- `loss` (train) should fall from ~2-4 toward ~0.8-1.5 for this data scale.
- `eval_loss` every 250 steps — the number that *selects the final checkpoint*
  (`load_best_model_at_end=True`). Rising eval_loss with falling train loss =
  overfitting → fewer epochs / lower lr / smaller r.
- `eval_perplexity` = exp(eval_loss) on masked-prompt tokens.
- TensorBoard: `tensorboard --logdir outputs/lora_v1` (port-forward through
  SSH or the cluster's web portal).

## 6. Debug-first discipline (cheap → expensive)

1. `make smoke` — mechanics, CPU, 1 min.
2. `make local-debug` — 1 GPU, 200 samples, 1 epoch: catches data bugs,
   OOM, and config typos in minutes (runs directly on the node, no SLURM).
3. Full node LoRA run (`make local-lora` or `make submit-lora`).
4. Full-parameter run (optional, after LoRA results make sense).
