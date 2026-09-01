#!/usr/bin/env bash
# =============================================================================
# container_exec.sh — run ANY command inside the training container with the
# standard mounts. Single place that knows about the image + bind layout.
#
#   bash scripts/container_exec.sh <command...>
#     bash scripts/container_exec.sh bash scripts/run_eval_all.sh
#     bash scripts/container_exec.sh python -c "import torch; print(torch.cuda.device_count())"
#
# Image comes from .env APPTAINER_IMAGE (or env var). Also works on a plain
# SSH session: if `apptainer` is not on PATH it tries `module load apptainer`.
# Used by BOTH run modes (local + slurm/train_lora_container.sbatch).
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/setup/load_env.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/cuda_fix.sh"

# --- locate the container (persistent sandbox preferred over SIF) ------------
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/resolve_container.sh"
CONTAINER_PATH="$(_resolve_container)" || exit 1
if [[ -d "$CONTAINER_PATH" ]]; then
  CONTAINER_MODE="sandbox (persistent — no per-run SIF conversion)"
else
  CONTAINER_MODE="SIF FALLBACK — sandbox NOT FOUND at ${APPTAINER_SANDBOX:-<default>}!"\
"(moved/deleted? the SIF may also be STALE — rebuild with: make container-build)"
fi

# --- make sure apptainer exists (non-interactive shells lack the module fn) ---
_ensure_apptainer() {
  command -v apptainer >/dev/null 2>&1 && return 0
  local init
  for init in /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash \
              /etc/profile.d/modules.sh /etc/profile.d/z00_modules.sh; do
    if [[ -f "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init" 2>/dev/null || true
      break
    fi
  done
  if command -v module >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || true
  fi
  command -v apptainer >/dev/null 2>&1
}
_ensure_apptainer || { echo "ERROR: apptainer not found — run: module load apptainer"; exit 1; }

# --- mounts + env passthrough -------------------------------------------------
HF_CACHE="${HF_HOME:-$HOME/llm-scratch/hf_cache}"   # .env SCRATCH/HF_HOME preferred
mkdir -p "$HF_CACHE" 2>/dev/null || true
mkdir -p "$HF_CACHE/triton_cache" 2>/dev/null || true

EXTRA_ENV=(
  --env "HF_HOME=/hf_cache"
  --env "PYTHONNOUSERSITE=1"          # host ~/.local must not shadow container pkgs
  --env "ENV=/dev/null"               # neutralize NGC's /etc/shinit_v2: it is
  --env "BASH_ENV=/dev/null"          # dash-incompatible and breaks the CUDA
                                      # compat lib under apptainer (see cuda_fix.sh)
  --env "TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib"
  # DGX nodes expose many NICs (8x NDR IB + storage IB + ethernet); NCCL 2.25
  # fails merging them ("TOPO/NET: Tried merging multiple devices...").
  # LOC = merge only within the same host — correct for single-node runs.
  --env "NCCL_NET_MERGE_LEVEL=LOC"
  # keep triton's autotune cache off NFS ($HOME) — put it on the NVMe bind
  --env "TRITON_CACHE_DIR=/hf_cache/triton_cache"
)
# --cleanenv strips the host environment, so explicitly forward every repo
# configuration variable (from .env or the invoking shell) that scripts and
# commands inside the container may rely on. Without this, `ADAPTER=... make
# local-eval` was silently ignored (three identical 30k-adapter eval runs).
_PASSTHROUGH=(
  GPUS ADAPTER MODEL OUT ROWS RUN_MODE ZERO DEEPSPEED_ZERO
  DATASET_ID DATA_TRAIN_ROWS DATA_VAL_ROWS DATA_TEST_ROWS
  EVAL_MODEL EVAL_ADAPTER EVAL_MERGED EVAL_PROMPTS EVAL_DIR
  EVAL_MAX_NEW_TOKENS EVAL_BATCH_SIZE PROMPTS EVALDIR MAXNEW BATCH
  HF_TOKEN
)
for _v in "${_PASSTHROUGH[@]}"; do
  if [[ -n "${!_v:-}" ]]; then
    EXTRA_ENV+=(--env "$_v=${!_v}")
  fi
done

BINDS=(
  --bind "${REPO_ROOT}:/workspace"
  --bind "${HF_CACHE}:/hf_cache"
)

# --- CUDA driver lib fix (bind staged lib over the compat dir) ---------------
# take the LAST line defensively: only the dir may live on stdout
if CUDA_FIX_DIR="$(_ensure_cuda_fix | tail -n 1)"; then
  BINDS+=(--bind "${CUDA_FIX_DIR}:/usr/local/cuda/compat/lib")
else
  echo "==> continuing without the libcuda fix (triton will fall back or fail)"
fi

# --- SIF sandbox temp dir (only relevant when running from a SIF) ------------
# SIF without squashfuse gets extracted to a temp sandbox (~15 GB) — keep it
# off /tmp (often tmpfs/RAM) and on NVMe scratch instead. Sandboxes skip this
# entirely.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH:-$HOME/llm-scratch}/apptainer_tmp}"
mkdir -p "$APPTAINER_TMPDIR" 2>/dev/null || true

echo "==> container : $CONTAINER_PATH  ($CONTAINER_MODE)"
echo "==> repo bind : $REPO_ROOT -> /workspace"
echo "==> cache bind: $HF_CACHE -> /hf_cache"

exec apptainer exec \
  --nv \
  --cleanenv \
  "${BINDS[@]}" \
  "${EXTRA_ENV[@]}" \
  "$CONTAINER_PATH" \
  bash -c 'cd /workspace && exec "$@"' container_exec "$@"
