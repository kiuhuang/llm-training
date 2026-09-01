# 02 — Data preparation (worked example: Finance-Instruct-500k)

The pipeline is dataset-agnostic: it turns chat-style rows into the
messages JSONL the trainer consumes, samples what you need, and holds out a
test split. The finance dataset below is the worked example.

## The raw dataset (verified on the HF hub)

- **id**: `Josephgflowers/Finance-Instruct-500k` — Apache-2.0
- **518,432 rows**, single `train` split, ~1.83 GB (one big `train.json`)
- **exactly 3 string columns**: `system`, `user`, `assistant`
- quirks discovered by inspection:
  - `system` is often just `"\n"` (i.e. empty) — most rows are 2-turn
  - RAG entries prepend source context to the `user` text (some `user` fields
    are very long)
  - no category/task column; merged from many sources (Sujet-177k, Phinance,
    OpenMathInstruct-1, WebInstructSub, Financial-NER, ...)
  - author claims dedup, but we re-dedup anyway (cheap, safer)

## Why we don't train on all 518k rows

For *learning* fine-tuning, 20-50k high-quality examples already teach a 4B
base model the domain style. 518k rows × ~2 epochs would be many GPU-hours for
your first run. Sample first, scale later — the pipeline is identical.

## The pipeline: `data_prep/prepare_data.py`

```
stream (parquet revision → no 1.8 GB download)
  → clean: strip whitespace, drop empty/ "\n"-only system, drop empty
           user/assistant, drop >100k-char rows, exact-dedup (sha1)
  → seeded shuffle-sample
  → split: train / val / test            ← test is NEVER trained on
  → write:
     data/processed/train.jsonl   {"messages":[...]}   ← training
     data/processed/val.jsonl     {"messages":[...]}   ← eval_loss during training
     data/processed/test.jsonl    {"messages":[...]}   ← perplexity + prompt mining
     outputs/eval/eval_prompts.jsonl                   ← prompts for A/B generation
     data/processed/stats.json                         ← what happened
```

Key design point: **eval prompts come from the held-out test split**, with the
original assistant answer stored as `reference`. That's what makes the
pre/post comparison honest — the models are graded on questions whose answers
neither ever saw.

## Usage

All data commands run **inside the container** (the host python is too old
for the HF stack; the container's `datasets` does the streaming, and outputs
land in the repo via the `/workspace` bind):

```bash
make data                                    # 30k train / 500 val / 500 test + eval prompts
make data ROWS=5000                          # quick first pass
make inspect                                 # length stats + examples (char-level)
# token-level stats (downloads only the ~15 MB tokenizer):
bash scripts/container_exec.sh python data_prep/inspect_data.py --tokenizer Qwen/Qwen3.5-4B-Base
```

## How to read `inspect_data.py` output

- **total chars / tokens p50-p99** → choose `max_seq_len`. If p99 ≈ 3k tokens,
  `max_seq_len: 4096` drops almost nothing; examples over the limit are
  *dropped*, not truncated (truncated answers teach the model to stop mid-sentence).
- **assistant length distribution** → pick `max_new_tokens` for eval (cover
  ~p90 of references; longer wastes GPU time, shorter inflates "truncation").

Measured on the real sample (Aug 2026 run): mean ≈ 171 tokens/example,
total-chars p99 ≈ 5.2k → `max_seq_len: 4096` keeps ~everything, and
`max_new_tokens: 512` comfortably covers the reference-length p90.

## Training on the FULL ~518k dataset

The 30k default is an iteration choice, not a limit. With the measured
token length, the full dataset is cheap:

```bash
make data-full            # downloads the 1.8 GB JSON, full dedup + full shuffle
make inspect              # expect ~517k train rows after cleaning
make local-lora           # 2 epochs ≈ 8k steps → ~1.5-3 h on 8x H800
```

Notes for the big run:
- `--no-streaming` downloads the whole file once and gives a TRUE global
  shuffle (streaming shuffle is buffer-approximate — fine for sampling, less
  ideal when every row is used).
- Consider **1 epoch first** — evaluate, then decide whether epoch 2 is worth it:
  `bash scripts/container_exec.sh bash scripts/train_lora.sh --epochs 1`
- Everything else (eval cadence, checkpoints) scales automatically; adapter
  checkpoints are small (~85 MB each).

## Offline tests

`tests/test_prepare_data.py` covers the cleaning rules (run it anywhere, no
dependencies needed).
