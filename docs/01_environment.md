# 01 — Environment setup (multi-GPU training node)

Target hardware per node — example numbers from the DGX-class node these docs
were executed on:
- 8× NVIDIA H800 (80 GB HBM3 each, SM90), 2× 56-core Xeon (112 cores), 2 TB RAM
- 30.72 TB local NVMe, 8× 400 Gb/s NDR InfiniBand
- (a ~55-node pod of identical nodes; the pipeline itself needs exactly one)

## 1. Where things live (matters a lot on shared clusters)

Configure once with `make init` (creates `.env` from `.env.example`), then edit
`SCRATCH`, `SLURM_*`, `CONTAINER_BASE`, `APPTAINER_IMAGE` etc. Precedence:
shell environment > `.env` > script defaults. Inspect the resolved values
with `make env-show`.

| what | where | why |
|---|---|---|
| code (this repo) | your home or project dir | small, versioned |
| HF cache (`HF_HOME`) | **local NVMe** (`SCRATCH/hf_cache` from `.env`) | model 9.3 GB + dataset ~2 GB + checkpoints 10-100 GB; home quotas are small and network FS is slow |
| venv (`VENV_DIR`) | home (small) or NVMe | a few GB |

The model (~9.3 GB bf16), the dataset (~1.8 GB), and full-FT checkpoints (~50 GB
each with optimizer states) will crush a small home quota if you leave
`HF_HOME` at its default.

## 2. Modules (adapt names to `module avail`)

```bash
module purge
module load cuda/12.4        # or whatever CUDA the cluster provides (12.1+ works)
module load gcc/12.3.0       # only needed to compile flash-attn / deepspeed ops
module load python/3.11      # or use anaconda3 module, or system python3
```

> **Dated-host reality check (common on shared clusters; verified on our
> node):** host modules are older (python 3.9, Anaconda 2023.09, CUDA
> 11.8/12.2 toolkits) while the GPU driver is modern (580.x → CUDA 13.0). The
> **recommended route is an Apptainer container** — NGC PyTorch 25.03 base
> (torch 2.7, CUDA 12.8.1) with prebuilt flash-attn, full GPU/NVLink speed.
> See **docs/08_containers.md**. The bare-metal venv path below still works
> for a quick smoke test on hosts with a modern python (cu124 torch wheels
> run fine on a 580-series driver; you just won't have flash-attn without a
> matching nvcc — training falls back to SDPA).

If the cluster uses containers (Apptainer/Podman), the same steps apply inside
a CUDA 12.x container with a bind-mounted scratch dir.

## 3. Create the environment

```bash
cd llm-training
bash setup/setup_env.sh                    # venv at ~/.venvs/llm-train
# variants:
FLASH_ATTN=1 bash setup/setup_env.sh       # + compile flash-attention-2 (30-60 min, one-time)
VENV_DIR=/raid/$USER/venvs/llm-train bash setup/setup_env.sh
```

`setup_env.sh` installs torch (cu124 wheels) first, then
`setup/requirements.txt` (transformers>=5.16.1 — required by Qwen3.5's
`qwen3_5` model type), then runs `setup/verify_env.py`.

`transformers>=5.16.1` matters: Qwen3.5 is a new architecture
(`Qwen3_5ForConditionalGeneration` — hybrid Gated-DeltaNet linear attention +
gated full attention) that ships in the transformers **5.x** line only —
4.57.x does not recognize it. transformers 5.x also needs **python >= 3.10**,
so on this cluster (host python 3.9) the container route is effectively
mandatory for Qwen3.5 training; the bare-metal venv path is only usable on
hosts with a modern python.

## 4. Verify (no downloads)

```bash
python setup/verify_env.py
```

Expect: torch CUDA available, 8 GPUs, sm_90, bf16 supported. On a login node
GPU count may be 0 — that's fine; check inside a `srun`/job allocation.

## 5. HF token (optional)

Not needed for Qwen3.5-4B-Base or Finance-Instruct-500k (both Apache-2.0, not
gated). Set `HF_TOKEN` only if you later add gated models.

## 6. Before the first real run

```bash
make smoke        # offline end-to-end mechanics test (tiny random model, CPU)
```

This exercises the exact masking/collator/Trainer/LoRA/save-reload path in
~1-2 minutes without downloading anything.
