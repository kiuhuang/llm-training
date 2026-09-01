# Handy entry points. Run `make` with no args to list targets.
# Cluster-specific paths/versions live in .env (create with `make init`).
#
# TWO RUN MODES (.env RUN_MODE, default local):
#   local = execute directly on the node you SSH-ed into (no SLURM needed —
#           a node allocated/drained to you, so full-node runs work)
#   slurm = submit through sbatch (uses SLURM_* values from .env)
# The `train-*` / `eval` targets dispatch on RUN_MODE; use the explicit
# local-* / submit-* targets to pin a mode.

PYTHON  ?= python
GPUS    ?=                # empty -> .env GPUS (default 8)
ROWS    ?=                # empty -> .env DATA_TRAIN_ROWS (or script default 30000)
MODEL   ?= Qwen/Qwen3.5-4B-Base   # worked example — any causal LM on the Hub works
OUT     ?= outputs/lora_v1

.PHONY: help init env-show setup verify verify-ctn smoke data data-full inspect \
        local-debug local-lora local-full local-eval \
        local-lora-venv local-full-venv local-eval-venv local-lora-ctn local-eval-ctn \
        train-lora train-lora-ctn train-full eval \
        submit-lora submit-lora-ctn submit-full submit-eval \
        merge infer-base infer-tuned compare report container-build container-build-sif clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- config ----
init:             ## Create .env from .env.example (never overwrites an existing .env)
	@if [[ -f .env ]]; then \
	  echo ".env already exists — nothing to do (edit it directly)"; \
	else \
	  cp .env.example .env; \
	  echo "created .env from .env.example"; \
	  echo "next: edit .env — at minimum SCRATCH and SLURM_* on a new cluster"; \
	fi

env-show:         ## Print the effective configuration (shell env > .env > defaults)
	@bash -c 'source setup/load_env.sh; \
	for v in RUN_MODE SCRATCH HF_HOME APPTAINER_CACHEDIR SLURM_PARTITION SLURM_GRES \
	         SLURM_TIME SLURM_CPUS SLURM_MEM CONTAINER_BASE APPTAINER_SANDBOX \
	         APPTAINER_IMAGE BUILD_SIF GPUS DEEPSPEED_ZERO \
	         EVAL_MODEL EVAL_ADAPTER EVAL_MERGED DATASET_ID \
	         DATA_TRAIN_ROWS VENV_DIR TORCH_INDEX; do \
	  printf "  %-20s %s\n" "$$v" "$${!v-<unset>}"; done'

# ------------------------------------------------------------ environment ----
setup:            ## Create venv + install all dependencies (bare-metal path)
	bash setup/setup_env.sh

verify:           ## Check torch/CUDA/GPUs/library versions (bare-metal venv path)
	$(PYTHON) setup/verify_env.py

verify-ctn:       ## Same checks inside the container (GPUs visible on an allocated node)
	bash scripts/container_exec.sh python setup/verify_env.py

smoke:            ## [container] Offline end-to-end smoke test (tiny random model, CPU, ~1 min)
	bash scripts/container_exec.sh python tests/smoke_test_train.py

data:             ## [container] Stream + clean + sample + split the finance dataset
	bash scripts/container_exec.sh python data_prep/prepare_data.py $(if $(ROWS),--train-rows $(ROWS),)

data-full:        ## [container] Use the FULL ~518k dataset (1.8 GB download, full shuffle)
	bash scripts/container_exec.sh python data_prep/prepare_data.py --no-streaming --train-rows 518000

inspect:          ## [container] Stats + examples for the processed dataset
	bash scripts/container_exec.sh python data_prep/inspect_data.py

# ------------------------------------------------------------ LOCAL MODE ----
# "local" = execute directly on the node you SSH-ed into, INSIDE the container
# (the host python has no torch stack — see docs/01). Bare-metal venv variants
# are the explicit *-venv targets.

local-debug:      ## [container] 1-GPU, 200-sample debug run before burning GPU-hours
	GPUS=1 bash scripts/container_exec.sh bash scripts/train_lora.sh --max_samples 200 --epochs 1 --output_dir $(OUT)-debug

local-lora:       ## [container] LoRA SFT directly on this node (all .env GPUS, no SLURM)
	bash scripts/container_exec.sh bash scripts/train_lora.sh --output_dir $(OUT)

local-full:       ## [container] Full-parameter SFT with DeepSpeed ZeRO on this node
	bash scripts/container_exec.sh bash scripts/train_full.sh --output_dir outputs/full_v1

local-eval:       ## [container] Full evaluation on this node: generation + ppl + report
	bash scripts/container_exec.sh bash scripts/run_eval_all.sh

local-lora-venv:  ## [venv] LoRA SFT with the host venv (only if the venv has the stack)
	bash scripts/train_lora.sh --output_dir $(OUT)

local-full-venv:  ## [venv] Full-parameter SFT with the host venv
	bash scripts/train_full.sh --output_dir outputs/full_v1

local-eval-venv:  ## [venv] Full evaluation with the host venv
	bash scripts/run_eval_all.sh

# aliases kept for continuity with earlier docs
local-lora-ctn: local-lora
local-eval-ctn: local-eval

# ------------------------------------------------------- DISPATCH (RUN_MODE) --
train-lora:       ## LoRA SFT — RUN_MODE from .env decides local vs slurm
	@bash -c 'source setup/load_env.sh; mode="$${RUN_MODE:-local}"; \
	  case "$$mode" in \
	    local) exec $(MAKE) --no-print-directory local-lora ;; \
	    slurm) exec $(MAKE) --no-print-directory submit-lora ;; \
	    *) echo "RUN_MODE must be 'local' or 'slurm' (got: $$mode) — fix .env"; exit 1 ;; esac'

train-lora-ctn:   ## Containerized LoRA SFT — RUN_MODE decides local vs slurm
	@bash -c 'source setup/load_env.sh; mode="$${RUN_MODE:-local}"; \
	  case "$$mode" in \
	    local) exec $(MAKE) --no-print-directory local-lora-ctn ;; \
	    slurm) exec $(MAKE) --no-print-directory submit-lora-ctn ;; \
	    *) echo "RUN_MODE must be 'local' or 'slurm' (got: $$mode) — fix .env"; exit 1 ;; esac'

train-full:       ## Full-parameter SFT — RUN_MODE decides local vs slurm
	@bash -c 'source setup/load_env.sh; mode="$${RUN_MODE:-local}"; \
	  case "$$mode" in \
	    local) exec $(MAKE) --no-print-directory local-full ;; \
	    slurm) exec $(MAKE) --no-print-directory submit-full ;; \
	    *) echo "RUN_MODE must be 'local' or 'slurm' (got: $$mode) — fix .env"; exit 1 ;; esac'

eval:             ## Evaluation — RUN_MODE decides local (venv) vs slurm submit
	@bash -c 'source setup/load_env.sh; mode="$${RUN_MODE:-local}"; \
	  case "$$mode" in \
	    local) exec $(MAKE) --no-print-directory local-eval ;; \
	    slurm) exec $(MAKE) --no-print-directory submit-eval ;; \
	    *) echo "RUN_MODE must be 'local' or 'slurm' (got: $$mode) — fix .env"; exit 1 ;; esac'

# ------------------------------------------------------------ SLURM MODE ----
submit-lora:      ## [slurm] sbatch LoRA run on the full node (SLURM_* from .env)
	@bash -c 'source setup/load_env.sh; \
	sbatch --partition="$${SLURM_PARTITION}" --gres="$${SLURM_GRES}" --time="$${SLURM_TIME}" \
	       --cpus-per-task="$${SLURM_CPUS}" --mem="$${SLURM_MEM}" slurm/train_lora.sbatch'

submit-lora-ctn:  ## [slurm] sbatch the containerized LoRA run (sandbox or SIF)
	@bash -c 'bash scripts/resolve_container.sh >/dev/null || exit 1; \
	sbatch --partition="$${SLURM_PARTITION}" --gres="$${SLURM_GRES}" --time="$${SLURM_TIME}" \
	       --cpus-per-task="$${SLURM_CPUS}" --mem="$${SLURM_MEM}" slurm/train_lora_container.sbatch'

submit-full:      ## [slurm] sbatch full-parameter SFT (DeepSpeed ZeRO)
	@bash -c 'source setup/load_env.sh; \
	sbatch --partition="$${SLURM_PARTITION}" --gres="$${SLURM_GRES}" --time="$${SLURM_TIME}" \
	       --cpus-per-task="$${SLURM_CPUS}" --mem="$${SLURM_MEM}" slurm/train_full.sbatch'

submit-eval:      ## [slurm] sbatch the evaluation job (1 GPU; container auto-detected)
	@bash -c 'source setup/load_env.sh; \
	sbatch --partition="$${SLURM_PARTITION}" --gres=gpu:1 --time=02:00:00 \
	       --cpus-per-task=16 --mem=100G slurm/eval.sbatch'

# ----------------------------------------------------------------- merge/eval -
merge:            ## [container] Merge LoRA adapter into the base model
	bash scripts/container_exec.sh python scripts/merge_lora.py --base_model $(MODEL) --adapter $(OUT)/final --out_dir outputs/merged_v1

infer-base:       ## [container] Generate eval answers with the UN-finetuned base model
	bash scripts/container_exec.sh python scripts/run_inference.py --model $(MODEL) --out outputs/eval/base.jsonl

infer-tuned:      ## [container] Generate eval answers with the fine-tuned model
	bash scripts/container_exec.sh python scripts/run_inference.py --model outputs/merged_v1 --out outputs/eval/tuned.jsonl

compare:          ## [container] Metrics + side-by-side report: base vs fine-tuned
	bash scripts/container_exec.sh python scripts/compare_results.py --base outputs/eval/base.jsonl --tuned outputs/eval/tuned.jsonl

report:           ## [container] All of eval in one go (same as local-eval)
	bash scripts/container_exec.sh bash scripts/run_eval_all.sh

# --------------------------------------------------------------- container ---
container-build:  ## Build container as PERSISTENT SANDBOX on $SCRATCH (no per-run SIF conversion)
	@bash -c 'source setup/load_env.sh; \
	cdir="$${APPTAINER_CACHEDIR:-$$SCRATCH/apptainer_cache}"; \
	sbox="$${APPTAINER_SANDBOX:-$$SCRATCH/apptainer_sandbox}"; \
	mkdir -p images "$$cdir"; \
	echo "==> base image : $${CONTAINER_BASE}"; \
	echo "==> sandbox    : $$sbox"; \
	sed "s|^From:.*|From: $${CONTAINER_BASE}|" container/qwen35.def > container/.build-qwen35.def; \
	rm -rf "$$sbox"; \
	apptainer build --fakeroot --sandbox "$$sbox" container/.build-qwen35.def \
	  && rm -f container/.build-qwen35.def; \
	if [[ "$${BUILD_SIF:-0}" == 1 ]]; then \
	  echo "==> packing SIF (backup/portability copy)"; \
	  mkdir -p images; \
	  apptainer build images/qwen35.sif "$$sbox"; \
	  echo "==> SIF: images/qwen35.sif"; \
	fi; \
	echo "done — runs use the sandbox (APPTAINER_SANDBOX), no per-run conversion"'

container-build-sif: ## Pack the existing sandbox into a SIF (no %post re-run)
	@bash -c 'source setup/load_env.sh; \
	sbox="$${APPTAINER_SANDBOX:-$$SCRATCH/apptainer_sandbox}"; \
	[[ -d "$$sbox" ]] || { echo "no sandbox at $$sbox — run make container-build"; exit 1; }; \
	mkdir -p images; \
	apptainer build images/qwen35.sif "$$sbox" && echo "==> SIF: images/qwen35.sif"'

clean:            ## Remove outputs (keeps data/)
	rm -rf outputs/* slurm-*.out

# compatibility alias
train-lora-debug: local-debug
