#!/usr/bin/env bash
# =============================================================================
# load_env.sh — source this at the top of any repo shell script to pick up
# the optional repo-root .env file.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../setup/load_env.sh"   # from scripts/
#   source setup/load_env.sh                                       # from repo root
#
# Semantics:
#   * missing .env  -> no-op (scripts fall back to their built-in defaults)
#   * KEY=VALUE lines; `export ` prefix, quotes and trailing "# comment" handled
#   * $VAR / ${VAR} are expanded from already-exported variables (so SCRATCH
#     can be referenced by later lines); unset references are left literal
#   * does NOT override variables already present in the environment
#     (precedence: shell env > .env > script defaults)
#   * override the file location with ENV_FILE=/path/to/.env
# =============================================================================
_load_repo_env() {
  local self="${BASH_SOURCE[0]}"
  local root
  root="$(cd "$(dirname "$self")/.." && pwd)"
  local envf="${ENV_FILE:-$root/.env}"
  [[ -f "$envf" ]] || return 0

  local line raw key val name match prev
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="${raw%$'\r'}"
    # skip blanks / comments
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    # optional "export " prefix
    line="${line#"${line%%[![:space:]]*}"}"            # ltrim
    [[ "$line" == export\ * ]] && line="${line#export }"
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    val="${val#"${val%%[![:space:]]*}"}"               # rtrim/ltrim
    val="${val%"${val##*[![:space:]]}"}"
    # trailing comment for unquoted values
    if [[ "$val" != \"* && "$val" != \'* && "$val" == *" #"* ]]; then
      val="${val%% \#*}"
      val="${val%"${val##*[![:space:]]}"}"
    fi
    # strip one matching pair of quotes
    if [[ "$val" == \"*\" && ${#val} -ge 2 ]]; then val="${val:1:${#val}-2}"; fi
    if [[ "$val" == \'*\' && ${#val} -ge 2 ]]; then val="${val:1:${#val}-2}"; fi
    # expand $VAR / ${VAR} from the current environment (bounded, no eval)
    prev=""
    while [[ "$val" == *'$'* && "$val" != "$prev" ]]; do
      prev="$val"
      if [[ "$val" =~ \$\{([A-Za-z_][A-Za-z0-9_]*)\} || "$val" =~ \$([A-Za-z_][A-Za-z0-9_]*) ]]; then
        name="${BASH_REMATCH[1]}"
        match="${BASH_REMATCH[0]}"
        if [[ -n "${!name+x}" ]]; then val="${val//"$match"/${!name}}"; fi
      else
        break
      fi
    done
    # shell env wins over .env
    if [[ -z "${!key+x}" ]]; then
      export "$key=$val"
    fi
  done < "$envf"

  _ENV_FILE_LOADED="$envf"
  return 0
}
_load_repo_env
unset -f _load_repo_env
