#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — create the Python environment for LLM fine-tuning
#
# Usage:
#   bash setup/setup_env.sh                 # venv + torch cu124 + requirements
#   FLASH_ATTN=1 bash setup/setup_env.sh    # also compile flash-attn (slow)
#   VENV_DIR=~/.venvs/llm-train bash setup/setup_env.sh
#
# On an HPC login node with Lmod modules (common on university/enterprise
# clusters), load the CUDA
# and compiler modules FIRST, or export the paths manually, e.g.:
#   module load cuda/12.4
#   module load gcc/12.3.0        # needed to compile flash-attn / deepspeed ops
#   module load anaconda3/2024.06 # if you prefer conda over venv
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Optional repo .env (make init): VENV_DIR, TORCH_INDEX, FLASH_ATTN, HF_HOME, SCRATCH
# shellcheck disable=SC1091
source "$REPO_ROOT/setup/load_env.sh"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/llm-train}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
FLASH_ATTN="${FLASH_ATTN:-0}"
HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"

echo "==> Repo root : $REPO_ROOT"
echo "==> Venv      : $VENV_DIR"
echo "==> Torch idx : $TORCH_INDEX"
echo "==> HF cache  : $HF_HOME (models/datasets will land here — point it at a scratch disk on HPC!)"

# ---------------------------------------------------------------------------
# 1. Create/activate virtual environment
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
  echo "==> Created venv at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 2. PyTorch first (pin the CUDA build so flash-attn/deepspeed match)
# ---------------------------------------------------------------------------
if python -c "import torch" 2>/dev/null; then
  echo "==> torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
  pip install torch --index-url "$TORCH_INDEX"
fi

# ---------------------------------------------------------------------------
# 3. Everything else
# ---------------------------------------------------------------------------
pip install -r "$REPO_ROOT/setup/requirements.txt"

# ---------------------------------------------------------------------------
# 4. Optional: flash-attention 2 (big speed/memory win on H800, compiles slow)
# ---------------------------------------------------------------------------
if [[ "$FLASH_ATTN" == "1" ]]; then
  echo "==> Installing flash-attn (this can take 30-60 min; MAX_JOBS limits parallelism)"
  export MAX_JOBS="${MAX_JOBS:-16}"
  pip install flash-attn --no-build-isolation || {
    echo "!! flash-attn build failed. You can run without it (attn_implementation='sdpa')."
  }
fi

# ---------------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------------
echo "==> Running environment verification..."
HF_HOME="$HF_HOME" python "$REPO_ROOT/setup/verify_env.py"

cat <<'EOF'

Setup done. Next steps:
  1. export HF_HOME=<scratch path>   # keep the multi-GB model/dataset off $HOME
  2. python data_prep/prepare_data.py --help     (prepare the finance dataset)
  3. bash scripts/train_lora.sh --help           (launch LoRA SFT)
  Full walkthrough: docs/01_environment.md then README.md
EOF
