#!/usr/bin/env bash
# =============================================================================
# resolve_container.sh — pick the container artifact to run:
#   1. persistent sandbox dir (.env APPTAINER_SANDBOX)   <- preferred: no
#      per-run SIF conversion (the host has no squashfuse, so SIF runs extract
#      ~15 GB to a temp dir every single time)
#   2. SIF (.env APPTAINER_IMAGE)                        <- fallback
# Prints the resolved path on stdout; exits 1 with the reason on stderr if
# neither exists.
#
# Usable as a sourced helper (_resolve_container) or standalone:
#   bash scripts/resolve_container.sh
# =============================================================================
_resolve_container() {
  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  local sandbox="${APPTAINER_SANDBOX:-$root/images/qwen35_sandbox}"
  [[ "$sandbox" != /* ]] && sandbox="$root/$sandbox"
  local image="${APPTAINER_IMAGE:-}"
  [[ -n "$image" && "$image" != /* ]] && image="$root/$image"

  if [[ -n "$sandbox" && -d "$sandbox" ]]; then
    echo "$sandbox"
    return 0
  fi
  if [[ -n "$image" && -f "$image" ]]; then
    echo "$image"
    return 0
  fi
  {
    echo "no container found."
    echo "  expected sandbox dir : $sandbox  (make container-build)"
    echo "  or SIF               : ${image:-<unset>}"
  } >&2
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # standalone mode: load .env for the paths, then resolve
  # shellcheck disable=SC1091
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/setup/load_env.sh"
  _resolve_container
fi
