#!/usr/bin/env bash
# =============================================================================
# train_full.sh — FULL-parameter SFT on a single node (8 GPUs + DeepSpeed ZeRO)
#
# Full fine-tuning of a 4B model needs optimizer sharding: AdamW states alone
# are ~32 GB (fp32 m+v) on top of 8 GB weights + 8 GB grads. ZeRO-2 shards the
# optimizer across the 8 GPUs; ZeRO-3 also shards the weights (use if OOM).
#
# Configuration precedence: shell env > .env (make init) > defaults below.
#
# Usage (from repo root):
#   bash scripts/train_full.sh
#   DEEPSPEED_ZERO=3 bash scripts/train_full.sh   # switch to ZeRO-3
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/setup/load_env.sh"

GPUS="${GPUS:-8}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/train_full.yaml}"
ZERO="${DEEPSPEED_ZERO:-2}"
DS_CONFIG="$REPO_ROOT/configs/deepspeed_zero${ZERO}.json"
[[ -f "$DS_CONFIG" ]] || { echo "missing $DS_CONFIG"; exit 1; }

export HF_HOME="${HF_HOME:-$HOME/llm-scratch/hf_cache}"
mkdir -p "$HF_HOME" 2>/dev/null || true

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "==> GPUS=$GPUS  CONFIG=$CONFIG  DEEPSPEED=zero$ZERO"
exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  "$REPO_ROOT/scripts/train_sft.py" \
  --config "$CONFIG" \
  --deepspeed "$DS_CONFIG" \
  "$@"
