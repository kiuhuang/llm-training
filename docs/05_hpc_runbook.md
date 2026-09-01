# 05 — Runbook: two ways to run, end to end

Every command from the repo root. First configure once:

```bash
make init          # creates .env — set SCRATCH (NVMe) and SLURM_* for your cluster
make env-show      # verify the resolved configuration
```

The repo supports **two run modes** (`.env` → `RUN_MODE`):

| | Mode A: `local` (default) | Mode B: `slurm` |
|---|---|---|
| how | direct execution on the node you SSH-ed into | `sbatch` submission |
| use when | a node is allocated/drained to you / interactive dev | cluster is busy / long unattended runs |
| entry points | `make local-*` | `make submit-*` |
| dispatcher | `make train-lora` (reads `RUN_MODE`) | same |

---

## 0. First time only

```bash
cd ~/github/llm-training
make init && $EDITOR .env          # SCRATCH + SLURM_PARTITION at minimum
```

**Container route (recommended when the host python/CUDA are dated):**

```bash
make container-build               # login node, internet, ~12 GB SIF on NVMe
# sanity: 8 GPUs visible inside (on an allocated node, run directly; in slurm, srun):
apptainer exec --nv images/qwen35.sif python -c "import torch; print(torch.cuda.device_count())"
```

**Bare-metal route (works, but no flash-attn — SDPA fallback):**

```bash
module load python39               # host python 3.9 is the max available
bash setup/setup_env.sh && python setup/verify_env.py
```

Either way, run the offline tests (CPU, no downloads — inside the container):

```bash
python tests/test_masking_logic.py && python tests/test_prepare_data.py   # host-ok, stdlib only
make smoke                        # full pipeline mechanics on a tiny random model
```

## 1. Data (CPU + network — runs inside the container)

```bash
make data                         # stream + clean + sample 30k + split + eval prompts
make inspect                      # sanity: counts, token/char length percentiles
```

## 2. Debug run (1 GPU, minutes) — do this before any full run

```bash
# local mode (container, 1 GPU, 200 samples):
make local-debug
# slurm:
make submit-lora-ctn               # (edit the sbatch size first) — or srun interactively, see below
```

## 3a. Mode A — local full run on an allocated node (no SLURM needed)

```bash
make local-lora                    # LoRA on all 8 GPUs (in the container), ~30-60 min
make local-full                    # full-parameter variant (DeepSpeed ZeRO)
```

Then evaluate on the same node:

```bash
make local-eval                    # generation A/B + perplexities + report
# → outputs/eval/comparison_report.md
```

Interactive shell tips: run inside `tmux`/`screen` so SSH drops don't kill
the run; watch `nvidia-smi` in a second pane; TensorBoard:
`tensorboard --logdir outputs/lora_v1 --port 6006` and forward
`ssh -L 6006:localhost:6006 <node-hostname>`.

## 3b. Mode B — SLURM submission

```bash
make submit-lora                   # LoRA, full node
make submit-lora-ctn               # containerized LoRA
make submit-full                   # full-parameter SFT (ZeRO)
squeue -u $USER
tail -f slurm-qwen4b-sft-lora-<jobid>.out
```

Interactive allocation instead of batch (debugging inside the queue):

```bash
srun --partition=$SLURM_PARTITION --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=00:40:00 --pty bash
# inside: GPUS=1 bash scripts/train_lora.sh --max_samples 200 --epochs 1 --output_dir outputs/debug
```

## 4. Evaluation (after training, either mode)

```bash
make local-eval          # container, on the node you're on
make submit-eval         # SLURM job (1 GPU; auto-detects container vs venv)
```

Outputs: `outputs/eval/comparison_report.md` (start here), `summary.json`,
`per_item.csv`, `perplexity.json`.

## 5. Iterate

Change one thing at a time; keep every run's artifacts:

```bash
OUT=outputs/lora_r64 make local-lora --            # or: make local-lora OUT=outputs/lora_r64
make merge OUT=outputs/lora_r64 MODEL=Qwen/Qwen3.5-4B-Base
make local-eval
```

Compare `outputs/*/training_summary.json` + each run's `comparison_report.md`.

## 6. Operational notes

- Pre-download the model before batch jobs so 8 workers don't race:
  `make container-build` first, then
  `bash scripts/container_exec.sh python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B-Base')"`
- Bare-metal checkpoints/venv on the node's NVMe (`SCRATCH` in `.env`), never `$HOME`.
- If a job dies at startup, check: partition name (`sinfo`), GRES naming
  (`gpu` vs `gpu:h800`), and that the SIF path in `.env` is absolute-or-repo-relative.
