#!/usr/bin/env bash
# scripts/colocate/run_smoke_host.sh
#
# Cheap-host smoke runner for the colocate (MPS+NCCL) MPS-required tests.
#
# Why this exists:
#   Modal sandbox H100 nodes don't pass --ipc=host to the container, so
#   NVIDIA MPS server reports "operation not supported" and the colocate
#   path can't actually run (see docs/colocate/implementation_log.md
#   §"Modal sandbox MPS limitation"). The Phase-4 / 6 / 7 tests
#   correctly skip on Modal but still need to run *somewhere* to
#   validate end-to-end correctness.
#
#   This script lets you do that on the cheapest GPU rental you can
#   find (Vast.ai 3090/4090/L40S, Lambda Labs spot, Hyperstack L40S,
#   etc.) — anything with one CUDA-8.0+ GPU and a container runtime
#   that doesn't sandbox IPC. Total cost on Vast.ai L40S is ~$0.20–$0.40
#   for one full pass once the cache is warm.
#
# Prerequisites on the host:
#   * Linux + NVIDIA driver >= 535 + CUDA Driver API 12.4+
#   * `nvidia-smi` shows at least 1 GPU
#   * Either:
#     - `--ipc=host` Docker container (Vast.ai default; Hyperstack default)
#     - OR bare-VM SSH (no Docker isolation at all)
#   * Python 3.10 or 3.11 + `pip` available
#   * `git` available, and outbound HTTPS to github.com + huggingface.co
#   * (optional) HF_TOKEN exported for gated models — Qwen3-0.6B-Base is
#     not gated, so this is only needed if you change the config.
#
# Usage (from a fresh checkout of this repo):
#   bash scripts/colocate/run_smoke_host.sh                 # tiny smoke (1 GPU)
#   bash scripts/colocate/run_smoke_host.sh --skip-setup    # tests only
#   bash scripts/colocate/run_smoke_host.sh --setup-only    # bootstrap, no tests
#   bash scripts/colocate/run_smoke_host.sh --full          # tiny + 4xGPU Phase 4/6/7
#   bash scripts/colocate/run_smoke_host.sh --tests=A,B,C   # run specific test files
#
# Environment overrides:
#   COLOCATE_TINY_CONVERGE_STEPS=50    # default 20; raise for stability
#   PHASE6_STABILITY_STEPS=200         # default 200; bump to 1000 on 4xH100
#   PHASE7_CONVERGE_STEPS=50           # default 50; bump to 1000 for full
#   SGLANG_DIR=/abs/path/to/sglang     # default <repo>/_sglang
#   PYTHON=python3.11                  # default whatever python3 is on PATH
#   PIP_INDEX_URL=...                  # default PyPI
#   COLOCATE_PIN_TORCH=1               # pin torch==2.5.* if you hit a wheel mismatch
#
# Exit codes:
#   0 — every selected test either PASSED or SKIPPED (clean)
#   1 — host pre-flight failed (no GPU / no MPS binary / no driver)
#   2 — invalid CLI flag
#   non-0 from pytest — at least one test FAILED; see captured log
#
# What it does:
#   1. (setup) Clone sglang at the pinned commit and apply both patches
#      (the existing disagg sglang.patch and our new colocate.patch).
#   2. (setup) `pip install -e .` torchspec + sglang in --user mode so
#      the host python sees them.
#   3. (run)   Pre-flight: report nvidia-smi, MPS daemon, GPU count.
#   4. (run)   `pytest tests/colocate/test_colocate_tiny.py -xvs`
#              — this is the 1-GPU + Qwen3-0.6B variant of Phase-4
#              one-step + Phase-7 mini convergence. The MPS skip gate
#              (tests/colocate/_mps_probe.py::mps_works) auto-skips with
#              a clear reason on hosts where MPS doesn't actually work,
#              so a SKIP outcome here means *the host* doesn't support
#              MPS, not that the colocate code is broken.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

SGLANG_DIR="${SGLANG_DIR:-$REPO_ROOT/_sglang}"
SGLANG_COMMIT="0f2df9370a1de1b4fb11b071d39ab3ce2287a350"
SGLANG_PATCH_VERSION="v0.5.8.post1"
PATCHES_DIR="$REPO_ROOT/patches/sglang/$SGLANG_PATCH_VERSION"

PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

DO_SETUP=1
DO_RUN=1
RUN_FULL=0
TESTS_OVERRIDE=""

for arg in "$@"; do
  case "$arg" in
    --skip-setup) DO_SETUP=0 ;;
    --setup-only) DO_RUN=0 ;;
    --full) RUN_FULL=1 ;;
    --tests=*) TESTS_OVERRIDE="${arg#--tests=}" ;;
    --help|-h)
      grep -E '^# ' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

banner() {
  echo
  echo "=============================================="
  echo "  $*"
  echo "=============================================="
}

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

setup_sglang() {
  banner "sglang: clone + apply patches"
  if [[ ! -d "$SGLANG_DIR" ]]; then
    git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
  fi
  (
    cd "$SGLANG_DIR"
    git fetch --depth=1 origin "$SGLANG_COMMIT" || true
    git checkout "$SGLANG_COMMIT"
    git reset --hard HEAD
    rm -f python/sglang/srt/speculative/spec_training_info.py
    git apply --recount "$PATCHES_DIR/sglang.patch" || true
    git apply --recount "$PATCHES_DIR/colocate.patch"
  )
}

setup_python() {
  banner "python: $($PYTHON --version) at $(command -v "$PYTHON")"
  $PIP install --upgrade pip wheel setuptools
  if [[ "${COLOCATE_PIN_TORCH:-0}" == "1" ]]; then
    $PIP install "torch==2.5.*" --index-url https://download.pytorch.org/whl/cu124
  else
    $PIP install torch
  fi
  $PIP install \
    "transformers==4.57.1" datasets tqdm wandb accelerate \
    pydantic omegaconf ray openai openai-harmony qwen-vl-utils \
    psutil "numpy<2.4" pyzmq numba cmake ninja packaging \
    setuptools pytest pytest-timeout

  banner "torchspec: pip install -e ."
  $PIP install -e ".[dev]"
  banner "sglang: pip install -e ."
  $PIP install -e "$SGLANG_DIR/python[all]"
}

if [[ $DO_SETUP -eq 1 ]]; then
  setup_sglang
  setup_python
else
  banner "Skipping setup (--skip-setup)"
fi

if [[ $DO_RUN -eq 0 ]]; then
  banner "Setup complete (--setup-only). Re-run without --setup-only to run tests."
  exit 0
fi

# ---------------------------------------------------------------------------
# 2. Pre-flight
# ---------------------------------------------------------------------------

banner "Pre-flight: GPU + MPS"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — host has no NVIDIA driver. Aborting." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
echo "GPU count: $GPU_COUNT"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "Need at least 1 GPU; found $GPU_COUNT." >&2
  exit 1
fi

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "nvidia-cuda-mps-control NOT FOUND — install the CUDA toolkit "  \
       "(it ships the MPS daemon)." >&2
  exit 1
fi
echo "MPS daemon binary: $(command -v nvidia-cuda-mps-control)"

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

# Pick which test files to run.
if [[ -n "$TESTS_OVERRIDE" ]]; then
  IFS=',' read -ra TEST_FILES <<< "$TESTS_OVERRIDE"
elif [[ $RUN_FULL -eq 1 ]]; then
  # 4×H100-class hosts: run the tiny + every MPS-gated full test. Each
  # test self-skips if its preconditions aren't met (e.g. has_h100_quad
  # for the Qwen3-8B tests; mps_works for everything), so this is safe
  # to run on a 1-GPU host too — the 4-GPU tests just SKIP cleanly.
  TEST_FILES=(
    "tests/colocate/test_colocate_tiny.py"
    "tests/colocate/test_one_step.py"
    "tests/colocate/test_grad_parity.py"
    "tests/colocate/test_stability.py"
    "tests/colocate/test_convergence.py"
  )
else
  TEST_FILES=(
    "tests/colocate/test_colocate_tiny.py"
  )
fi

banner "pytest: ${TEST_FILES[*]}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
# Default CUDA_VISIBLE_DEVICES depends on whether we're running --full
# (multi-GPU) or just the tiny smoke. Don't override an already-set value.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  if [[ $RUN_FULL -eq 1 ]] && [[ "$GPU_COUNT" -ge 4 ]]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$REPO_ROOT"
PYTEST_RC=0
$PYTHON -m pytest -xvs "${TEST_FILES[@]}" || PYTEST_RC=$?

banner "Smoke run complete (pytest exit=$PYTEST_RC)."
exit "$PYTEST_RC"
