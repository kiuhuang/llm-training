# 04 — Evaluation: comparing pre-training vs post-training, honestly

The reference Databricks tutorial validates with `eval_loss` only. That's
necessary but not sufficient — eval_loss tells you the model fits the
*training distribution* better, not that its *answers* are better. This repo
therefore runs four complementary checks on the held-out test split.

## 0. What's held out

`data_prep/prepare_data.py` carves 500 test rows (never trained on) and
exports 300 as `outputs/eval/eval_prompts.jsonl`:
`{"id", "messages" (prompt only), "reference" (original answer)}`.

## 1. Generation A/B (`run_inference.py`)

Same prompts, same decoding for both models: **greedy** (temperature 0) so the
comparison is deterministic. Base model = `Qwen/Qwen3.5-4B-Base`; tuned model =
adapter applied on the fly (`--adapter outputs/lora_v1/final`) or a merged
model dir. Outputs land in `outputs/eval/{base,tuned}.jsonl`.

## 2. Metrics (`compare_results.py` → `comparison_report.md`)

- **ROUGE-L F1 vs reference** — lexical overlap. Coarse (a perfect answer in
  different words scores low), but *paired* (same prompts, both models) it's a
  useful signal. The report shows mean delta with a bootstrap 95% CI and
  win/tie/loss counts.
- **Exact match** — after normalization; for definitional Q&A sometimes real.
- **Degenerate-output rate** — empty answers or repetition loops (a classic
  base-model failure mode that SFT should reduce).
- **Truncation & length stats** — did the tuned model learn to answer at
  reference-like length? Drifting to 3× length suggests verbosity bias.

Read the **side-by-side samples** at the bottom of the report — biggest wins
and biggest losses first. Metrics rank models; samples tell you *why*.

## 3. Perplexity on the test split (`perplexity.py`)

NLL of the *masked* (assistant-only) objective on held-out rows — the same
objective as training. Expected: tuned < base (often 0.2-0.6 nats/token on a
domain shift). If tuned ≈ base, the model didn't move; if tuned is *much*
lower while generation quality didn't improve, suspect overfitting/memorization
and re-check with fresh prompts.

## 4. LLM-as-judge (optional extension)

ROUGE-L can't reward better-but-differently-worded answers. The standard fix:
ask a strong model to score both answers (1-10, rubric: correctness, finance
terminology, completeness, concision). Not shipped here to keep the first
pipeline dependency-light — a good exercise: add `scripts/judge.py` that feeds
`(question, reference, base_answer, tuned_answer)` to a larger Qwen instruct
model and tallies pairwise verdicts. Same JSONL joins as
`compare_results.py` apply.

## 5. The one-command flow

```bash
bash scripts/run_eval_all.sh
# 1) base generation  2) tuned generation  3) base ppl  4) tuned ppl
# 5) comparison_report.md + summary.json + per_item.csv
```

## 6. Sanity checklist for "did it work?"

- [ ] eval_loss during training decreased and plateaued (not still falling at stop)
- [ ] tuned perplexity < base perplexity on test
- [ ] ROUGE-L delta positive (or neutral CI) with fewer degenerate outputs
- [ ] tuned answers *follow the chat format* — a base model often rambles,
      continues with new questions, or ignores the role play entirely
- [ ] samples read like a finance assistant, not a paraphrase machine
