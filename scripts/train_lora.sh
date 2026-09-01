#!/usr/bin/env bash
# =============================================================================
# train_lora.sh — LoRA SFT on a single node (default: all 8 GPUs of a DGX node)
#
# Configuration precedence: shell env > .env (make init) > defaults below.
#
# Usage (from repo root):
#   bash scripts/train_lora.sh                          # uses configs/train_lora.yaml
#   GPUS=1 bash scripts/train_lora.sh --max_samples 200 # quick debug
#   bash scripts/train_lora.sh --learning_rate 5e-5     # any flag overrides YAML
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/setup/load_env.sh"

GPUS="${GPUS:-8}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/train_lora.yaml}"
export HF_HOME="${HF_HOME:-$HOME/llm-scratch/hf_cache}"   # .env SCRATCH/HF_HOME preferred
mkdir -p "$HF_HOME" 2>/dev/null || true

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "==> GPUS=$GPUS  CONFIG=$CONFIG  HF_HOME=$HF_HOME"
exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  "$REPO_ROOT/scripts/train_sft.py" \
  --config "$CONFIG" \
  "$@"
