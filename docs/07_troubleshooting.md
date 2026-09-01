# 07 — Troubleshooting

## CUDA / memory

| symptom | cause | fix |
|---|---|---|
| NCCL `TOPO/NET: Tried merging multiple devices` | DGX nodes expose 8+ NICs (NDR IB, storage IB, ethernet); NCCL 2.25 fails merging them | `NCCL_NET_MERGE_LEVEL=LOC` — set automatically by `container_exec.sh` |
| triton cache warning (NFS) | `$HOME` is bind-mounted; triton defaults to `~/.triton` | `TRITON_CACHE_DIR=/hf_cache/triton_cache` — set automatically |
| `find_unused_parameters=True ... did not find any unused` (DDP warning) | Trainer auto-enables the flag; for LoRA on a fixed arch it's pure overhead | `ddp_find_unused_parameters=False` (set by default in train_sft.py) |
| 7+ s/it on 8 GPUs with short examples | tiny micro-batches + padding waste (mean 247 vs max 4096 tokens) | `group_by_length=True` (default in train_sft.py); optionally raise `per_device_batch_size` to 8 + `--grad_accum 2` (watch memory on the rare all-long batches) |
| `CUDA out of memory` in first steps | per-device batch too big for 80 GB | lower `per_device_batch_size` to 2 or 1, raise `grad_accum` to keep global batch 128 |
| OOM *after* many steps | fragmentation / long batches | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launcher env; or reduce `max_seq_len` |
| OOM only with DeepSpeed ZeRO-3 | bucket sizes | keep ZeRO-2 for 4B+LoRA; for full FT ZeRO-3 add `"offload_optimizer": {"device": "cpu"}` |
| 8 GPUs but ~1 used | launched with `python` instead of torchrun | use `scripts/train_lora.sh` (torchrun) |
| NCCL timeout/hang | IB/PCIe topology | `NCCL_P2P_DISABLE=1` as a first test; check with admins for the cluster's recommended NCCL env |

## Model / library

| symptom | cause | fix |
|---|---|---|
| `KeyError: 'qwen3_5'` / unknown architecture | transformers too old — the architecture ships in the 5.x line only | `pip install -U "transformers>=5.16.1,<6"` (container: rebuild with `make container-build`) |
| flash-attn import error | wheel/arch mismatch | run with `--attn sdpa` (default fallback) or reinstall `FLASH_ATTN=1 bash setup/setup_env.sh` |
| `AssertionError: torch was compiled with...` | wrong CUDA wheel for module | install torch matching the cluster's CUDA (see setup_env.sh TORCH_INDEX) |
| loss = `nan` early | fp16 or lr too high | bf16 only (default); drop lr to 5e-5; check data for garbage rows (inspect_data.py) |
| eval_loss explodes | overfitting | fewer epochs, lower lr, smaller r, bigger dataset sample |

## Data

| symptom | cause | fix |
|---|---|---|
| `prepare_data.py` very slow / downloads 1.8 GB | parquet revision unreachable → JSON fallback | normal; use `--no-streaming` if you prefer one clean download |
| far fewer rows than requested | strict cleaning | read `data/processed/stats.json` drop reasons; relax `--max-total-chars` |
| all examples dropped at tokenize step | `max_seq_len` below typical prompt+answer | check token stats via `inspect_data.py --tokenizer ...`; raise `max_seq_len` |

## .env

| symptom | cause | fix |
|---|---|---|
| a variable "doesn't work" though the line looks right in `cat .env` | invisible character in the line (stray `\r`, non-breaking space — often from pasting) — the loader silently skips unparsable lines | audit: `python3 setup/envcfg.py KEY` — it prints every assignment line as LOADED / SHADOWED (shell wins) / SKIPPED (with raw repr). Fix: delete the line and re-add it by typing (not pasting), then re-audit |
| `.env` value ignored entirely | shell environment beats `.env` by design | `python3 setup/envcfg.py KEY` shows `SHADOWED by shell env` — `unset` the variable or edit the shell profile |
| unsure which variables the scripts actually see | — | `python3 setup/envcfg.py` lists every key with its resolved value and source |

## Databricks-vs-here differences (if you followed the reference tutorial first)

- Their `SFTTrainer` handles masking internally; here masking is explicit in
  `model_utils.make_encoder` (see `tests/test_masking_logic.py`).
- Their 1×H100 full-FT fits because Qwen3-4B text-only is 4B params and
  effective batch 8; our full-FT config assumes 8×H800 + ZeRO.
- `setup_chat_format(...)` (their template fallback) is replaced by the
  CHATML fallback inside `load_tokenizer` — Qwen3.5 already ships a template,
  so it rarely triggers.

## Getting help

Run with `--max_samples 200 --epochs 1` on 1 GPU; if that works, the bug is
scale/config, not code. Capture: full traceback, `nvidia-smi`, the config
block printed at training start.
