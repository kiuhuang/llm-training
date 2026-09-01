#!/usr/bin/env bash
# =============================================================================
# train_container.sh — LoRA/full training inside the Apptainer container.
#
# Works in BOTH modes without SLURM:
#   * a local SSH session on an allocated/drained node: just run it
#   * inside a SLURM allocation (or use slurm/train_lora_container.sbatch)
#
# Configuration precedence: shell env > .env (make init) > defaults.
#
#   bash scripts/train_container.sh                                 # all GPUs
#   GPUS=1 bash scripts/train_container.sh --max_samples 200        # debug
#
# Build the image first:  make container-build
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/setup/load_env.sh"

export GPUS="${GPUS:-8}"
echo "==> containerized training  GPUS=$GPUS"
exec "$REPO_ROOT/scripts/container_exec.sh" bash scripts/train_lora.sh "$@"
