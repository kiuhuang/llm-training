#!/bin/bash
#SBATCH --job-name=llm-sft-full
#SBATCH --partition=gpu            # static default; `make submit-full` overrides with .env SLURM_*
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8               # ADAPT: --gres=gpu:h800:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

# =============================================================================
# FULL-parameter SFT of a 4B-class model on one DGX H800 node.
# Uses DeepSpeed ZeRO-2 (DEEPSPEED_ZERO=3 in .env for ZeRO-3) to shard states.
# Configuration comes from the repo .env (make init). Submit: make submit-full
# =============================================================================
set -euo pipefail

echo "node: $(hostname)  start: $(date)"
nvidia-smi

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source setup/load_env.sh

# module purge
# module load cuda/12.4
source "${VENV_DIR:-$HOME/.venvs/llm-train}/bin/activate"

export HF_XET_HIGH_PERFORMANCE=1   # hf_transfer retired in hub 1.x; Xet is the fast path
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HOME="${HF_HOME:-$HOME/llm-scratch/hf_cache}"
mkdir -p "$HF_HOME"

GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
ZERO="${DEEPSPEED_ZERO:-2}"
bash scripts/train_full.sh "$@"

echo "done: $(date)"
