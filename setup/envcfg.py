#!/usr/bin/env python
"""Tiny dependency-free .env loader, matching setup/load_env.sh semantics.

Usage in repo scripts:

    sys.path.insert(0, <repo_root>/setup)
    from envcfg import load_repo_dotenv
    load_repo_dotenv(__file__)     # loads <repo_root>/.env if present

Precedence: os.environ > .env > caller defaults. $VAR/${VAR} references are
expanded from os.environ (unset references are left literal).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: str | os.PathLike, override: bool = False) -> list[str]:
    """Parse a .env file into os.environ. Returns the keys that were set."""
    p = Path(path)
    if not p.exists():
        return []
    loaded: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:  # trailing comment on unquoted values
            val = val.split(" #", 1)[0].strip()
        val = os.path.expanduser(os.path.expandvars(val))
        if override or key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def load_repo_dotenv(script_file: str, override: bool = False) -> list[str]:
    """Load <repo_root>/.env, where repo_root is two levels above script_file
    (works for scripts/ and data_prep/ and tests/ entry points)."""
    root = Path(script_file).resolve().parent.parent
    return load_dotenv(root / ".env", override=override)


def repo_root(script_file: str) -> Path:
    return Path(script_file).resolve().parent.parent


if __name__ == "__main__":
    # Diagnostic: audit every assignment-looking line in <repo>/.env —
    # loaded / shadowed-by-shell / SKIPPED (with raw repr for invisible chars).
    # Stdlib only, runs on any host python (3.9+).
    #   python3 setup/envcfg.py            # full audit
    #   python3 setup/envcfg.py KEY ...    # only these keys
    import re
    import sys

    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if not env_file.exists():
        print(f"no .env at {env_file} — run 'make init' and edit it")
        sys.exit(1)
    loaded = load_dotenv(env_file)
    loaded_set = set(loaded)
    keys = sys.argv[1:]

    seen: set[str] = set()
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            print(f"  UNPARSABLE LINE: {raw!r}")
            continue
        if keys and key not in keys:
            continue
        if key in seen:
            continue
        seen.add(key)
        if key in loaded_set:
            print(f"  {key} = {os.environ.get(key)}")
        elif key in os.environ:
            print(f"  {key} = {os.environ[key]}   (SHADOWED by shell env)")
        else:
            print(f"  {key} = <UNSET>  SKIPPED LINE: {raw!r}"
                  "  <- invisible character or bad syntax")

    shell_only = [k for k in keys if k not in seen and k in os.environ]
    for k in shell_only:
        print(f"  {k} = {os.environ[k]}   (from shell env; no line in .env)")
    print(f"({len(loaded)} keys loaded from {env_file})")
