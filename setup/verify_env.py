#!/usr/bin/env python
"""Verify the training environment: versions, CUDA, GPUs, optional extras.

Run:  python setup/verify_env.py
Exit code is 0 if the *core* stack is usable, 1 otherwise.
Downloads nothing.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envcfg import load_repo_dotenv  # noqa: E402

load_repo_dotenv(__file__)  # optional repo .env: HF_HOME etc.

OK, WARN, FAIL = "  OK ", "WARN ", "FAIL "
rows: list[tuple[str, str, str]] = []  # (status, item, detail)


def check(item: str, fn, required: bool = True):
    try:
        detail = fn()
        rows.append((OK if required else WARN, item, str(detail)))
        return detail
    except Exception as e:  # noqa: BLE001
        rows.append((FAIL if required else WARN, item, f"{type(e).__name__}: {e}"))
        return None


def mod_version(name: str):
    def _f():
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    return _f


print(f"Python  : {platform.python_version()}  ({sys.executable})")

check("torch", mod_version("torch"))
torch = None
try:
    import torch  # noqa: F401
except Exception:
    pass

if torch is not None:
    check("torch.cuda.is_available", lambda: torch.cuda.is_available())
    check("torch CUDA runtime", lambda: torch.version.cuda)
    check("GPU count", lambda: torch.cuda.device_count())
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        rows.append((OK, "GPUs", ", ".join(names)))
        cap = torch.cuda.get_device_capability(0)
        rows.append((OK, "Compute capability", f"sm_{cap[0]}{cap[1]} (H800 = sm_90)"))
        check("bf16 supported", lambda: torch.cuda.is_bf16_supported())

check("transformers", mod_version("transformers"))
check("trl", mod_version("trl"))
check("peft", mod_version("peft"))
check("datasets", mod_version("datasets"))
check("accelerate", mod_version("accelerate"))
check("deepspeed", mod_version("deepspeed"), required=False)
check("bitsandbytes", mod_version("bitsandbytes"), required=False)
check("rouge_score", mod_version("rouge_score"), required=False)
check("tensorboard", mod_version("tensorboard"), required=False)

try:
    import flash_attn  # type: ignore
    rows.append((WARN, "flash-attn", f"{flash_attn.__version__} (optional, faster attn)"))
except Exception:
    rows.append((WARN, "flash-attn", "not installed — training will use PyTorch SDPA (fine)"))

# --- Filesystem -----------------------------------------------------------
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
rows.append(("INFO ", "HF_HOME", hf_home))
usage = shutil.disk_usage(hf_home)
free_gb = usage.free / 1e9
rows.append((OK if free_gb > 100 else WARN, "free disk at HF_HOME", f"{free_gb:.0f} GB (need ~30+ GB for 4B model + dataset + checkpoints)"))
if torch is not None and torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    rows.append((OK if mem >= 70 else WARN, "GPU memory (device 0)", f"{mem:.0f} GB"))

width = max(len(r[1]) for r in rows) + 2
print()
for status, item, detail in rows:
    print(f"{status}{item.ljust(width)}{detail}")

core_ok = not any(s == FAIL and i in {"torch", "transformers", "trl", "peft", "datasets", "accelerate"} for s, i, _ in rows)
print("\nCORE STACK:", "READY" if core_ok else "INCOMPLETE — fix FAIL rows above")
sys.exit(0 if core_ok else 1)
