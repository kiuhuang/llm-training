# Reference: Databricks "Full fine-tuning of Qwen3-4B" tutorial

This repo was originally built while following the Microsoft/Databricks
tutorial below. To respect Microsoft's copyright, the original notebook text
is **not** reproduced here — this file is our own short summary, plus links
to the official sources.

## Official sources

- Tutorial page:
  https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-qwen3-4b
- Notebook source (Databricks docs mirror):
  https://docs.databricks.com/aws/en/notebooks/source/sgc-examples/sgc-finetune-qwen3-4b.html

## What the tutorial does (our summary)

- **Model**: `Qwen/Qwen3-4B` (text-only 4B), loaded bf16 via
  `AutoModelForCausalLM`.
- **Method**: full-parameter fine-tuning with Hugging Face **TRL**
  (`SFTTrainer`/`SFTConfig`) in a Databricks "AI v5" Serverless GPU
  environment (1× H100 80 GB).
- **Data**: `trl-lib/Capybara` (conversational "messages" format),
  split 90/10 train/validation with `train_test_split(test_size=0.1, seed=42)`.
- **Key hyperparameters**: `per_device_train_batch_size=1`,
  `gradient_accumulation_steps=8` (effective batch 8), `learning_rate=2e-5`,
  `max_steps=50` (demo scale), `warmup_steps=10`, `weight_decay=0.01`,
  `bf16=True`, `gradient_checkpointing=True` (non-reentrant),
  `eval_steps=25`, `save_steps=25`, `metric_for_best_model="eval_loss"`,
  `load_best_model_at_end=True`.
- **Tokenizer**: fast tokenizer; `pad_token = eos_token`; ChatML setup only
  if the tokenizer ships without one.
- **Logging/registry**: MLflow run with system metrics; model registered to
  Unity Catalog via `mlflow.transformers.log_model` (task `llm/v1/chat`).
- **Evaluation**: validation loss on the 10% split only — no generation-based
  evaluation.

## How this repo differs

| | reference tutorial | this repo |
|---|---|---|
| model | Qwen3-4B (text-only) | Qwen3.5-4B-Base (worked example; any causal LM works) |
| data | trl-lib/Capybara | your custom dataset (worked example: Finance-Instruct-500k) |
| method | full FT, 1×H100 | LoRA (default) or full FT w/ DeepSpeed on 8×H800 |
| loss masking | inside SFTTrainer | explicit + unit-tested (`make_encoder`) |
| checkpoint choice | eval_loss | eval_loss **+** held-out perplexity |
| output quality | not evaluated | greedy A/B generation, ROUGE-L + paired stats, side-by-side report |

See `docs/03_training.md` for the training concepts and `docs/04_evaluation.md`
for the evaluation methodology.
