#!/bin/bash
#SBATCH --job-name=llm-eval
#SBATCH --partition=gpu            # static default; `make submit-eval` overrides with .env SLURM_*
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # evaluation needs a single GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

# =============================================================================
# Post-training evaluation (generation A/B + perplexities + comparison report).
# Auto-detects the environment, in this order:
#   1. Apptainer container, if .env APPTAINER_IMAGE points at an existing SIF
#   2. bare-metal venv (VENV_DIR from .env), otherwise
#
# Submit:                        make submit-eval
# Run locally (no SLURM):        make local-eval   or   make local-eval-ctn
# =============================================================================
set -euo pipefail

echo "node: $(hostname)  start: $(date)"
nvidia-smi

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source setup/load_env.sh

IMG=""
if bash scripts/resolve_container.sh >/dev/null 2>&1; then
  IMG="$(bash scripts/resolve_container.sh)"
fi

if [[ -n "$IMG" ]]; then
  echo "==> eval inside container: $IMG"
  export APPTAINER_IMAGE="$IMG"
  export APPTAINER_SANDBOX=""   # path already resolved above
  # container_exec.sh applies the CUDA libcuda fix, PYTHONNOUSERSITE,
  # tmpdir-on-scratch and the standard binds (single code path).
  bash scripts/container_exec.sh bash scripts/run_eval_all.sh "$@"
else
  echo "==> eval bare metal (venv)"
  VENV="${VENV_DIR:-$HOME/.venvs/llm-train}/bin/activate"
  [[ -f "$VENV" ]] || { echo "no container image and no venv — run 'make container-build' or 'bash setup/setup_env.sh'"; exit 1; }
  # shellcheck disable=SC1090
  source "$VENV"
  export HF_HOME="${HF_HOME:-$HOME/llm-scratch/hf_cache}"
  mkdir -p "$HF_HOME"
  bash scripts/run_eval_all.sh "$@"
fi

echo "done: $(date)"
