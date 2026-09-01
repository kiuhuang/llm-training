#!/usr/bin/env bash
# =============================================================================
# run_eval_all.sh — one command: generate base + tuned answers, compute
# perplexities, and build the comparison report.
#
# Configuration precedence: shell env > .env (make init) > defaults below.
#
#   bash scripts/run_eval_all.sh                          # after LoRA training
#
# Tuned-model resolution order:
#   1. $EVAL_ADAPTER (LoRA applied on the fly) if set and exists
#   2. $EVAL_MERGED (merged model) if that dir exists
#   3. otherwise base-only eval
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source setup/load_env.sh

MODEL="${MODEL:-${EVAL_MODEL:-Qwen/Qwen3.5-4B-Base}}"
ADAPTER="${ADAPTER:-${EVAL_ADAPTER:-outputs/lora_v1/final}}"
MERGED="${MERGED:-${EVAL_MERGED:-outputs/merged_v1}}"
PROMPTS="${PROMPTS:-${EVAL_PROMPTS:-outputs/eval/eval_prompts.jsonl}}"
EVALDIR="${EVALDIR:-${EVAL_DIR:-outputs/eval}}"
MAXNEW="${MAXNEW:-${EVAL_MAX_NEW_TOKENS:-512}}"
BATCH="${BATCH:-${EVAL_BATCH_SIZE:-8}}"

mkdir -p "$EVALDIR"
[[ -f "$PROMPTS" ]] || { echo "missing $PROMPTS — run data_prep/prepare_data.py first"; exit 1; }

echo "== configuration =="
echo "   model  : $MODEL"
echo "   adapter: ${ADAPTER:-<none — will fall back to $MERGED if present>}"
echo "   prompts: $PROMPTS  ($(wc -l < "$PROMPTS") rows — stale? re-run data prep after data-full)"
echo

echo "== [1/5] baseline generation =="
python scripts/run_inference.py --model "$MODEL" \
  --prompts "$PROMPTS" --out "$EVALDIR/base.jsonl" \
  --max-new-tokens "$MAXNEW" --batch-size "$BATCH"

echo "== [2/5] fine-tuned generation =="
if [[ -n "$ADAPTER" && -d "$ADAPTER" ]]; then
  python scripts/run_inference.py --model "$MODEL" --adapter "$ADAPTER" \
    --prompts "$PROMPTS" --out "$EVALDIR/tuned.jsonl" \
    --max-new-tokens "$MAXNEW" --batch-size "$BATCH"
elif [[ -d "$MERGED" ]]; then
  python scripts/run_inference.py --model "$MERGED" \
    --prompts "$PROMPTS" --out "$EVALDIR/tuned.jsonl" \
    --max-new-tokens "$MAXNEW" --batch-size "$BATCH"
else
  echo "!! no adapter ($ADAPTER) and no merged model ($MERGED) — copying base as placeholder"
  cp "$EVALDIR/base.jsonl" "$EVALDIR/tuned.jsonl"
fi

echo "== [3/5] perplexity: base =="
python scripts/perplexity.py --model "$MODEL" --tag base

echo "== [4/5] perplexity: tuned =="
if [[ -n "$ADAPTER" && -d "$ADAPTER" ]]; then
  python scripts/perplexity.py --model "$MODEL" --adapter "$ADAPTER" --tag tuned
elif [[ -d "$MERGED" ]]; then
  python scripts/perplexity.py --model "$MERGED" --tag tuned
else
  echo "(skipped — no tuned model)"
fi

echo "== [5/5] comparison report =="
python scripts/compare_results.py --base "$EVALDIR/base.jsonl" --tuned "$EVALDIR/tuned.jsonl"

echo
echo "Report: $EVALDIR/comparison_report.md"
