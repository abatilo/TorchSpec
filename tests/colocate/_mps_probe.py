# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Shared helpers for the colocate phase tests.

Centralised here because every Phase-4+ test needs the same two
preconditions (>=4 GPUs *and* a working MPS daemon), and the MPS
probe is a 50-line subprocess dance we don't want to copy four times.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def has_h100_quad() -> bool:
    """Detect whether we're on a Modal H100:4 (or any 4+ GPU box)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return len([g for g in out.splitlines() if g.strip()]) >= 4


def mps_works() -> bool:
    """True iff nvidia-cuda-mps-control is on PATH and the per-GPU
    server can actually start a CUDA context. False on hosts where
    the MPS server reports 'operation not supported' (e.g. Modal
    sandbox H100 nodes without --ipc=host); see
    docs/colocate/implementation_log.md for the full story.

    Implementation mirrors
    ``torchspec.colocate.mps._probe_mps_server_works`` but is kept
    here so test files don't need to import torchspec just to gate
    their pytest ``skipif``.
    """
    if not shutil.which("nvidia-cuda-mps-control"):
        return False
    pipe_dir = "/tmp/nvidia-mps"
    log_dir = "/tmp/nvidia-log"
    try:
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        env = {
            **os.environ,
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
        }
        if not os.path.exists(os.path.join(pipe_dir, "control")):
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                env=env, timeout=10,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
        probe_code = (
            "import ctypes, sys\n"
            "cuda = ctypes.CDLL('libcuda.so.1')\n"
            "rc = cuda.cuInit(0)\n"
            "if rc != 0:\n    sys.exit(rc)\n"
            "cnt = ctypes.c_int(0)\n"
            "rc = cuda.cuDeviceGetCount(ctypes.byref(cnt))\n"
            "sys.exit(rc)\n"
        )
        proc = subprocess.run(
            ["python3", "-c", probe_code],
            env=env, timeout=20,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False
