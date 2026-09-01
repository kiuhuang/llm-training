# llm-training — a general LLM/LoRA fine-tuning pipeline

A complete, learn-by-doing pipeline for supervised fine-tuning (SFT) of
causal LLMs with **LoRA/QLoRA or full-parameter training** on a single
multi-GPU node (8× H800 80 GB in our runs — any DGX-class node works), with a
rigorous **before/after evaluation**.

The *method* is model- and dataset-agnostic; the repo ships with a fully
executed **worked example** — Qwen3.5-4B-Base fine-tuned on finance
instructions — including real before/after numbers (see Results).

| | |
|---|---|
| worked-example model | [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) — Apache-2.0, 4B LM params (+ vision tower, ~9.3 GB bf16), 262k context, hybrid DeltaNet+attention |
| worked-example dataset | [Josephgflowers/Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k) — Apache-2.0, 518k rows (`system`/`user`/`assistant`); we sample 20-30k |
| target hardware | single multi-GPU node: 8× H800 80 GB, 112 cores, 2 TB RAM, 30+ TB local NVMe |
| reference tutorial | [Databricks: fine-tune Qwen3-4B](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-qwen3-4b) (captured in `docs/reference/`) |

**Status: two complete train+eval cycles executed on one 8× H800 node**
(30k × 2 epochs, then the full ~509k dataset × 1 epoch) — numbers below are
from `outputs/eval/` on the cluster.

## 🎓 Interactive tutorial — start here

Open **`tutorial/index.html`** in any browser (no install, no internet needed):
a 25-minute animated walkthrough of the whole method — why fine-tune, how
custom data becomes flashcards, what the loss mask really does (toggle it and
watch what the model would learn), LoRA explained with a rank slider, the
training loop with an overfitting lamp, and an honest-evaluation dashboard —
using the real numbers from the runs in this repo. No formulas, six
interactive widgets.

**Available in 5 languages**: English · 廣東話（香港） · 繁體中文 · 简体中文 · 日本語
(auto-detected from your browser, switchable in the sidebar, preference
remembered).

To serve it (e.g. for a classroom): `python3 -m http.server -d tutorial 8000`
→ `http://<host>:8000`.

## How it works (the 30-second version)

```
Finance-Instruct-500k ──stream/clean/sample/split──▶ data/processed/{train,val,test}.jsonl
                                                             │
Qwen3.5-4B-Base ──LoRA r=16 (language stack only)────────────┤ SFT, bf16, 8×H800 DDP
                                                             ▼
                                      outputs/lora_v1/final  (adapter)
                                                             │
held-out test prompts ──greedy generation──▶ base vs tuned ──┤ ROUGE-L, win/tie/loss,
perplexity on masked objective ──────────────────────────────┘ side-by-side report
                                                      ▼
                                  outputs/eval/comparison_report.md
```

## Quickstart

```bash
# 0) one-time (login node)                              details: docs/01_environment.md
make init                          # creates .env from .env.example — edit SCRATCH/SLURM_*
bash setup/setup_env.sh && python setup/verify_env.py
python tests/test_masking_logic.py && python tests/test_prepare_data.py && python tests/smoke_test_train.py
make env-show                      # check the effective configuration

# 1) data: stream → clean → sample 30k → split → eval prompts   docs/02_data.md
python data_prep/prepare_data.py
python data_prep/inspect_data.py

# 2) cheap debug: 1 GPU, 200 samples                            docs/03_training.md
make local-debug

# 3) real run — two modes, .env RUN_MODE=local|slurm            docs/05_hpc_runbook.md
make train-lora          # RUN_MODE=local → runs directly on this node (works without
                         #                 SLURM when a node is allocated/drained to you)
                         # RUN_MODE=slurm → sbatch submission (make submit-lora)
# explicit variants: local-full / submit-full, container: local-lora-ctn / submit-lora-ctn
# container artifacts: persistent sandbox on $SCRATCH (fast) + optional SIF (BUILD_SIF=1)

# 4) evaluate: base vs tuned                                    docs/04_evaluation.md
make eval                # dispatches on RUN_MODE; or local-eval / local-eval-ctn / submit-eval
# → outputs/eval/comparison_report.md
```

`make` lists all entry points (`make data`, `make train-lora`, `make compare`, …).

## Repository layout

```
setup/          environment: requirements.txt, setup_env.sh, verify_env.py, .env loaders
.env.example    all cluster-specific knobs (paths, SLURM, container) — copy to .env: make init
container/      qwen35.def — Apptainer image (NGC PyTorch + HF stack)
data_prep/      prepare_data.py (stream→clean→sample→split), inspect_data.py
configs/        train_lora.yaml, train_full.yaml, deepspeed_zero{2,3}.json
scripts/        train_sft.py (core: Trainer+PEFT, explicit loss masking)
                train_lora.sh / train_full.sh (torchrun launchers)
                container_exec.sh / train_container.sh (Apptainer wrapper, same code path inside)
                model_utils.py (multimodal-safe loader, masking, collator)
                merge_lora.py, run_inference.py, compare_results.py, perplexity.py,
                run_eval_all.sh
slurm/          train_lora.sbatch, train_full.sbatch, train_lora_container.sbatch, eval.sbatch
tests/          offline unit tests + end-to-end smoke test (tiny random model)
docs/           01_environment … 08_containers + reference/ (captured tutorial)
```

## Learning path (read in this order)

1. **docs/03_training.md** — what SFT optimizes, why prompts are masked with
   -100, LoRA math, memory arithmetic for a 4B model on 80 GB GPUs.
2. **docs/02_data.md** — dataset facts, why we sample + hold out a test split.
3. **scripts/model_utils.py** — read `make_encoder`: the 30 lines that *are* SFT.
4. **docs/05_hpc_runbook.md** — command-by-command for both run modes.
5. **docs/04_evaluation.md** — how to judge whether fine-tuning worked.
6. **docs/06_model_notes.md** — verified Qwen3.5-4B-Base specifics.
7. **docs/07_troubleshooting.md** — when (not if) something OOMs.

## How this extends the reference tutorial

The Databricks notebook (full FT of text-only Qwen3-4B, TRL `SFTTrainer`,
effective batch 8, lr 2e-5, 50 demo steps, eval_loss-only validation) teaches
the mechanics. This repo keeps the same HF-stack foundations and adds what a
real run needs:

| | reference tutorial | this repo |
|---|---|---|
| model | Qwen3-4B (text) | Qwen3.5-4B-Base (multimodal-capable hybrid) |
| data | trl-lib/Capybara | Finance-Instruct-500k (cleaned, sampled, split) |
| method | full FT, 1×H100 | LoRA (default) or full FT w/ ZeRO on 8×H800 |
| loss masking | inside SFTTrainer | explicit + unit-tested (`make_encoder`) |
| checkpoint choice | eval_loss | eval_loss **+** held-out perplexity |
| output quality | not evaluated | greedy A/B generation, ROUGE-L + paired stats, side-by-side report |

## Swapping the model or dataset

The pipeline is architecture-agnostic — the worked example is just the default:

- **Other dataset**: set `DATASET_ID` in `.env`. `prepare_data.py` expects
  `{system, user, assistant}` string columns (the common instruct format);
  other schemas need a small edit in `normalize()` (docs/02). Eval prompts are
  rebuilt automatically from the new test split.
- **Other model**: change `model_name_or_path` (configs) or `MODEL` (Makefile).
  What adapts automatically: LoRA targets (inspected from the module tree —
  q/k/v/o/gate/up/down or `all-linear` fallback), missing chat templates
  (ChatML fallback installed), multimodal checkpoints (vision tower excluded
  from LoRA), and EOS stopping at inference (eos + chat end-of-turn markers).
  What you must check per model: the **transformers version** its architecture
  needs (qwen3_5 required ≥ 5.16.1 — see docs/06), and sensible LR/batch for
  its size (docs/03 has the memory arithmetic).
- **Other container base**: `CONTAINER_BASE` in `.env` (any CUDA 12.x/13.x
  NGC-style PyTorch image with a matching driver) — no code changes; the same
  sandbox serves any model.

## Results

Two full runs, same evaluation: 300 held-out prompts (greedy) + masked
perplexity on 498 held-out rows. Both = Qwen3.5-4B-Base, LoRA r=16 / α=32
(21.2M trainable, 0.50%), bf16, seq 4096, global batch 128, one 8× H800 node.

| | run 1: 30k examples × 2 epochs | run 2: ~509k examples × 1 epoch |
|---|---|---|
| optimizer steps | 470 | 3,978 |
| wall-clock | ~1 h | 9.5 h (un-tuned config; 2–4 h with length grouping) |
| held-out perplexity (masked) | 2.40 | **2.07** (−14%) |
| ROUGE-L F1 vs reference | 0.309 | 0.313 (+0.004) |
| exact match | 0.0% | **5.0%** |
| paired win/tie/loss vs base | 219 / 64 / 17 | 216 / 65 / 19 |
| degenerate outputs | 0% | 0.7% |
| truncated generations (512 cap) | 0.3% | 2.3% |
| mean answer length (words) | 59.2 (ref: 59.4) | 66.1 (ref: 59.4) |

Reading:
- **Both runs transform the base model** (ROUGE 0.129 → ~0.31, degenerate
  ~0, reference-scale answers with proper stopping; base perplexity on this
  test set is 3.02).
- **17× more data → perplexity −14%, generation metrics ~flat.** Classic SFT
  scaling: format, style, and stopping saturate within a few thousand
  examples; more data keeps improving distribution fit (ppl) and exact
  recall (5% of answers now match a reference nearly verbatim) but does not
  move n-gram overlap.
- **Practical sweet spot: 30k × 2 epochs at ~1 h.** The 518k run buys
  coverage/robustness at ~10× cost — worth it if the use case rewards
  long-tail knowledge, not for style alone.
- The exact-match jump deserves a look in `outputs/eval/comparison_report.md`:
  near-verbatim reference recall can be genuine (canonical regulatory Q&A)
  or mild memorization of frequent patterns.

First-run lessons now encoded in the scripts: generation must stop at the
chat turn end (`<|im_end|>`), transformers 5.x removed several
`TrainingArguments` kwargs (the script adapts automatically), and
`--cleanenv` needs an explicit pass-through list for config variables.

## License

MIT (see LICENSE). Model and dataset are Apache-2.0 — respect both upstream
licenses when redistributing weights or derived data.
