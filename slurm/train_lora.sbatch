#!/bin/bash
#SBATCH --job-name=llm-sft-lora
#SBATCH --partition=gpu            # static default; `make submit-lora` overrides with .env SLURM_*
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # one torchrun process launches all 8 GPU workers
#SBATCH --gres=gpu:8               # ADAPT if the cluster names the type: --gres=gpu:h800:8
#SBATCH --cpus-per-task=32         # node has 112 cores total; 32 is plenty for data loading
#SBATCH --mem=400G                 # node has 2 TB RAM; take what you need
#SBATCH --time=24:00:00            # ADAPT to the queue's max walltime
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

# =============================================================================
# LoRA SFT of a 4B-class model on one DGX H800 node (8x H800 80 GB) — bare metal.
# For the containerized version (recommended when the host python/CUDA are
# dated), use
# slurm/train_lora_container.sbatch.
#
# Configuration comes from the repo .env (make init): HF_HOME, VENV_DIR, ...
#
# Submit with .env SLURM_* overrides:   make submit-lora
# =============================================================================
set -euo pipefail

echo "node: $(hostname)  start: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
nvidia-smi

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source setup/load_env.sh
if [[ "${SLURM_PARTITION:-}" && "${SLURM_PARTITION}" != "gpu" ]]; then
  echo "NOTE: .env sets SLURM_PARTITION=$SLURM_PARTITION but this file requests 'gpu'."
  echo "      Submit with 'make submit-lora' to apply .env values."
fi

# --- Environment --------------------------------------------------------------
# module purge
# module load cuda/12.4          # only needed to compile flash-attn; wheels run on the driver alone
source "${VENV_DIR:-$HOME/.venvs/llm-train}/bin/activate"   # created by setup/setup_env.sh

export HF_XET_HIGH_PERFORMANCE=1   # hf_transfer retired in hub 1.x; Xet is the fast path
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# HF_HOME comes from .env (SCRATCH/hf_cache); fall back safely:
export HF_HOME="${HF_HOME:-$HOME/llm-scratch/hf_cache}"
mkdir -p "$HF_HOME"

# --- Launch ------------------------------------------------------------------
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
bash scripts/train_lora.sh "$@"

echo "done: $(date)"
