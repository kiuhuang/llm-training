#!/bin/bash
#SBATCH --job-name=llm-sft-lora-ctn
#SBATCH --partition=gpu            # static default; `make submit-lora-ctn` overrides with .env SLURM_*
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8               # ADAPT: --gres=gpu:h800:8 if typed GRES exist
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

# =============================================================================
# LoRA SFT of Qwen3.5-4B-Base on one DGX H800 node — INSIDE an Apptainer
# container (NGC PyTorch base). Container layout, CUDA fixes and env are all
# handled by scripts/container_exec.sh (single code path, also used locally).
#
# Build once on the login node (internet needed):  make container-build
# Submit:                                          make submit-lora-ctn
# =============================================================================
set -euo pipefail

echo "node: $(hostname)  start: $(date)"
nvidia-smi

module load apptainer              # ADAPT: matches `module av` on the cluster

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source setup/load_env.sh

export GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-8}}"

# container_exec.sh resolves the container (persistent sandbox preferred, SIF
# fallback), applies the CUDA libcuda fix, PYTHONNOUSERSITE, and the standard
# binds.
exec bash scripts/container_exec.sh bash scripts/train_lora.sh "$@"
