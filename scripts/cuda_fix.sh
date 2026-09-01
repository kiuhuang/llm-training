#!/usr/bin/env bash
# =============================================================================
# cuda_fix.sh — stage a driver-matched libcuda.so.1 into scratch and expose
# _ensure_cuda_fix, used by container entry points.
#
# Why: NGC images ship /usr/local/cuda/compat/lib/libcuda.so.1 (a compat
# driver lib) and their /etc/shinit_v2 script rewires it at startup based on
# the RUNNING driver version. That script is bash-flavored but gets sourced
# by /bin/sh (dash) under Apptainer, where it fails mid-way ("not a valid
# test operator") and leaves the compat lib broken -> triton dies with
# "libcuda.so cannot found!" when compiling its CUDA utils.
#
# Fix strategy:
#   1. copy a REAL libcuda.so.1 (host driver first — exact match with the
#      running kernel module; NGC compat lib as fallback) to $SCRATCH/cuda_fix
#   2. entry points bind that dir OVER /usr/local/cuda/compat/lib and set
#      TRITON_LIBCUDA_PATH, plus ENV/BASH_ENV=/dev/null so the broken
#      shinit_v2 never runs inside the container
# =============================================================================
_ensure_cuda_fix() {
  local fix="${CUDA_FIX_DIR:-${SCRATCH:-$HOME/llm-scratch}/cuda_fix}"
  if [[ ! -f "$fix/libcuda.so.1" ]]; then
    mkdir -p "$fix"
    local cand found=""
    # candidates can be overridden (space-separated) for clusters that keep
    # driver libs in non-standard module paths
    local candidates="${LIBCUDA_CANDIDATES:-/usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so.1}"
    for cand in $candidates; do
      if [[ -e "$cand" ]]; then
        if cp -L "$cand" "$fix/libcuda.so.1" 2>/dev/null; then
          found="$cand"
          break
        fi
      fi
    done
    if [[ -z "$found" ]]; then
      echo "WARN: no libcuda.so.1 found on this host (triton kernels will likely fail;" >&2
      echo "       training still works with attn=sdpa). Set LIBCUDA_CANDIDATES if the" >&2
      echo "       driver lib lives in a non-standard path." >&2
      return 1
    fi
    # NOTE: info goes to STDERR — callers capture this function's stdout as the dir
    echo "==> staged driver lib: $found -> $fix/libcuda.so.1" >&2
  fi
  ln -sf libcuda.so.1 "$fix/libcuda.so"   # linker needs the dev symlink too
  echo "$fix"                             # ONLY the dir on stdout
}
