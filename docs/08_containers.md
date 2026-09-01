# 08 — Apptainer containers (recommended route)

The host environment is dated (python 3.9, Anaconda 2023.09, CUDA toolkits
11.8/12.2 only) but ships a modern **driver (580.159.03 → CUDA 13.0)** and
Apptainer + Slurm. That's exactly the situation containers solve: bring your
own CUDA 12.8 user space, keep the host kernel and GPU driver.

## Short answer: does a container cost performance?

**No — GPU throughput is identical to bare metal.** Apptainer is not a VM:
- same Linux kernel, just namespaces; syscalls pass straight through
- GPUs appear as real `/dev/nvidia*` device nodes (`--nv` injects the host
  driver's user-space libraries); CUDA kernels launch directly on the tensor
  cores — there is no virtualization layer to bypass
- NVLink topology is visible inside (`nvidia-smi topo -m`), so NCCL P2P runs
  at full NVLink speed; NCCL ships inside the NGC image, built for H100/H800
- bind mounts are the same filesystem paths the host would use — put
  `HF_HOME` on the 30.7 TB local NVMe and I/O is bare-metal too

Real costs: one-time image build (download ~10-15 GB SIF), ~5-20 s container
startup per job, and nothing else. The things that actually determine
throughput (MFU, flash-attention, sequence length — docs/03_training.md) are
unchanged. The NGC base even *saves* time: flash-attn is prebuilt, while a
bare-metal compile would need an nvcc matching the torch CUDA build (host
only has 11.8/12.2 — a mismatch that would quietly cost you the fast path).

## One-time setup (login node, needs internet)

```bash
cd ~/llm-training
make init                            # creates .env — set SCRATCH (NVMe) + SLURM_* + APPTAINER_SANDBOX
make container-build                 # builds a PERSISTENT SANDBOX at APPTAINER_SANDBOX
# if --fakeroot is not permitted on your cluster:
#   apptainer build --sandbox "$APPTAINER_SANDBOX" container/.build-qwen35.def   # userns build
```

Why a sandbox instead of a SIF: this host has no `squashfuse`, so **every SIF
run extracts the ~15 GB image to a temp dir and deletes it afterwards**
(`Converting SIF file to temporary sandbox...` → `Cleaning up image...`).
A persistent sandbox directory on NVMe pays that cost once at build time;
runs start in seconds. Set `BUILD_SIF=1` in `.env` (or `make
container-build-sif`) if you also want a packed SIF for backup/portability —
the loader prefers the sandbox whenever it exists and falls back to the SIF.

The def file starts from `nvcr.io/nvidia/pytorch:25.03-py3` (torch 2.7.0a0 +
NCCL 2.25.1, CUDA 12.8.1, Python 3.12) and only pip-upgrades the HF stack on
top — the `transformers>=5.16.1,<6` pin exists because the `qwen3_5`
architecture ships in the 5.x line only (and NGC torch builds carry
pre-release version strings ("2.7.0a0"), and transformers gates on a minimum
torch via PEP440 comparison (the older 24.09 base, "2.5.0a0", failed that
gate), and the `%post` check also asserts `AutoConfig.for_model("qwen3_5")`
so missing-architecture surprises fail the build instead of first training.

## Verify before burning GPU time

```bash
apptainer exec --nv images/qwen35.sif python -c "import torch, transformers; \
  print(torch.__version__, transformers.__version__, torch.cuda.device_count())"
# expect: 2.7.0a0 5.16.x 8   (8 inside a GPU allocation, 0 on the login node)
apptainer exec --nv images/qwen35.sif nvidia-smi      # driver/topo visible
apptainer exec --nv images/qwen35.sif python -c "import flash_attn; print('flash-attn OK')"
```

## Training

```bash
# interactive debug (1 GPU):
srun --partition=gpu --gres=gpu:1 --mem=100G --time=00:40:00 --pty bash
IMAGE=/raid/$USER/images/qwen35.sif GPUS=1 \
  bash scripts/train_container.sh --max_samples 200 --epochs 1 --output_dir outputs/debug

# full node:
IMAGE=/raid/$USER/images/qwen35.sif sbatch slurm/train_lora_container.sbatch

# evaluation and everything else — same wrapper:
IMAGE=... bash scripts/train_container.sh --config configs/train_lora.yaml ...
```

`scripts/train_container.sh` bind-mounts the repo at `/workspace` and the
scratch at `/hf_cache`, then runs the *same* `scripts/train_lora.sh` inside —
one code path, containerized interpreter.

## Notes & gotchas

- **Home quota:** the SIF (~12 GB), `APPTAINER_CACHEDIR`, and `HF_HOME`
  (model 9.3 GB + dataset + checkpoints) all belong on NVMe scratch, not
  `$HOME`. The sbatch sets this up; keep it that way.
- **NGC's `shinit_v2` breaks under Apptainer (CUDA compat lib):** NGC images
  source `/etc/shinit_v2` in every shell to rewire `/usr/local/cuda/compat/lib`
  for the running driver — but the script is bash-flavored and crashes when
  `/bin/sh` (dash) sources it inside Apptainer ("not a valid test operator"),
  leaving a broken compat lib → triton dies with
  `libcuda.so cannot found!` when compiling its CUDA utils. Fix (automated in
  `scripts/cuda_fix.sh` + `container_exec.sh`): copy the host's real
  `libcuda.so.1` to `$SCRATCH/cuda_fix`, bind it over the compat dir, set
  `TRITON_LIBCUDA_PATH`, and neutralize the script with
  `ENV=/dev/null BASH_ENV=/dev/null`.
- **`$HOME` leak (the classic Apptainer footgun):** `$HOME` is bind-mounted by
  default, and a host `~/.local/lib/python3.X/site-packages` SHADOWS the
  container's packages (user-site beats system dist-packages) → mysterious
  version-mixing ImportErrors at runtime while the build passed. All our
  container entry points set `PYTHONNOUSERSITE=1` (see `%environment` in the
  def file). If you ever did `pip install --user` inside a container shell,
  clean the debris: `rm -rf ~/.local/lib/python3.12` (host python is 3.9, so
  a `python3.12` dir under `~/.local` is container debris).
- **`squashfuse` missing → sandbox extraction:** without it, Apptainer
  converts the SIF to a temporary sandbox (~15 GB) per run. We redirect that
  to NVMe via `APPTAINER_TMPDIR` (`.env`) — don't let it land in `/tmp`
  (often tmpfs/RAM).
- **`/dev/shm`:** NCCL and the DataLoader use shared memory. Check
  `df -h /dev/shm` inside the container; if it's tiny on your cluster, add
  `--bind /dev/shm` or ask admins about tmpfs sizing (DGX nodes usually have
  ~1 TB — fine by default).
- **`--cleanenv`:** the wrapper uses it to avoid host `PYTHON*`/`LD_*` leaks.
  If you need a proxy env var inside, pass it explicitly with `--env`.
- **Downloads inside jobs:** pre-pull the model on the login node (inside the
  container, same `HF_HOME`) so 8 workers don't race the download:
  ```bash
  apptainer exec --nv --bind $HOME/llm-training:/workspace \
    --bind /raid/$USER/hf_cache:/hf_cache --env HF_HOME=/hf_cache \
    images/qwen35.sif \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B-Base')"
  ```
- **Multi-node (later):** single-node needs nothing special. For multi-node
  over NDR InfiniBand you'll run `srun` + torchrun; NCCL talks to the host's
  kernel IB drivers, and NGC images are built for exactly this. Treat it as a
  follow-up exercise — every result in this repo is achievable on one node.
- **If builds are blocked** on your cluster (no network, no fakeroot), fall
  back to: `apptainer pull docker://nvcr.io/nvidia/pytorch:25.03-py3` +
  a writable overlay where you pip-install the HF stack:
  ```bash
  apptainer overlay create --size 4096 overlay.img          # 4 GB ext3
  apptainer exec --overlay overlay.img --nv image.sif bash  # pip install ... (persists)
  ```
