# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""NVIDIA MPS (Multi-Process Service) lifecycle helper (Phase 1).

The colocate plan puts a trainer process and an inference engine process on
the same physical GPU. By default CUDA serialises kernels from different
processes, which makes context-switch overhead dominate. MPS reroutes both
processes' commands to a single per-GPU server so the GPU sees them as
threads of one client and can run kernels concurrently.

What this module does:

    1. Detect whether `nvidia-cuda-mps-control` is already running on this
       node (idempotent — multiple drivers must coexist safely).
    2. If not, start it with `nvidia-cuda-mps-control -d` (daemon mode).
    3. Return the env-var dict that client processes (TrainerActor and
       SglEngine actors) need to merge into their Ray ``runtime_env``.
    4. Provide a best-effort cleanup hook (`stop_mps_daemon`) called at
       shutdown.

What this module does NOT do:

    - Manage `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`. That's an optional Phase-6
      knob; off by default.
    - Spawn one daemon per GPU. A single MPS control daemon services all
      GPUs visible to the calling user.
    - Touch CUDA — it's pure subprocess + filesystem, so it's safely
      importable from the Ray driver on a headless box.

The module is split out so that:

    - Unit tests can verify env-var construction and idempotency without
      requiring NVIDIA drivers (subprocess is mocked).
    - The Ray driver doesn't import torch just to set up MPS.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("torchspec.colocate.mps")

# Default control-pipe and log directories. MPS clients identify the daemon
# by these env vars, so trainer and engine processes must agree on them
# (and so must the daemon process). These are the documented NVIDIA
# defaults; we expose them as constants so tests can match them.
DEFAULT_PIPE_DIR = "/tmp/nvidia-mps"
DEFAULT_LOG_DIR = "/tmp/nvidia-log"

_MPS_CONTROL_BIN = "nvidia-cuda-mps-control"
_MPS_SERVER_BIN = "nvidia-cuda-mps-server"


@dataclass
class MpsHandle:
    """Information about a started (or detected) MPS daemon."""

    pipe_dir: str
    log_dir: str
    started_by_us: bool
    """True if *this* call launched the daemon. False if it was already
    running, in which case ``stop_mps_daemon`` becomes a best-effort no-op."""


def mps_client_env(pipe_dir: str = DEFAULT_PIPE_DIR, log_dir: str = DEFAULT_LOG_DIR) -> dict[str, str]:
    """Env vars that MPS clients (trainer + engine) need.

    Both must point at the same control pipe directory; otherwise they'd
    talk to different MPS servers (or none), defeating the colocate goal.
    Documented at https://docs.nvidia.com/deploy/mps/index.html#environment-variables.
    """
    return {
        "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
        "CUDA_MPS_LOG_DIRECTORY": log_dir,
    }


def is_mps_available() -> bool:
    """True iff ``nvidia-cuda-mps-control`` is in PATH.

    Used as a precondition for callers that want to fall back gracefully on
    boxes without MPS (e.g. local dev, CPU-only CI).
    """
    return shutil.which(_MPS_CONTROL_BIN) is not None


def is_mps_running(pipe_dir: str = DEFAULT_PIPE_DIR) -> bool:
    """True iff an MPS control daemon appears to be running on this node.

    We check two signals because either alone is unreliable:

    - The control pipe directory exists *and* contains the named pipe
      ``control`` (created by the daemon at startup).
    - ``ps`` shows an `nvidia-cuda-mps-control` process.

    Either match is good enough; we only need one to avoid double-starting.
    """
    pipe_file = os.path.join(pipe_dir, "control")
    if os.path.exists(pipe_file):
        return True

    if not shutil.which("pgrep"):
        # On an unusual base image without pgrep — fall back to "no daemon".
        # We'd rather double-start (the second instance fails fast with
        # `daemon already running`) than skip startup on a fresh box.
        return False
    try:
        rc = subprocess.run(
            ["pgrep", "-f", _MPS_CONTROL_BIN],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode
    except subprocess.TimeoutExpired:
        return False
    return rc == 0


def start_mps_daemon(
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    *,
    skip_if_running: bool = True,
) -> MpsHandle:
    """Start the MPS control daemon (idempotent).

    Args:
        pipe_dir: ``CUDA_MPS_PIPE_DIRECTORY`` to use. Defaults to NVIDIA's
            documented ``/tmp/nvidia-mps`` so a daemon started by
            ``nvidia-cuda-mps-control -d`` (no env vars) works out of the
            box.
        log_dir: ``CUDA_MPS_LOG_DIRECTORY`` to use.
        skip_if_running: If True (default), return without starting if a
            daemon is already up. Set to False for tests that want to force
            a fresh start.

    Returns:
        An ``MpsHandle`` capturing the directories and whether *we* started
        the daemon.

    Raises:
        FileNotFoundError: ``nvidia-cuda-mps-control`` not in PATH.
        RuntimeError: the start command failed (e.g. permission error,
            previous orphaned daemon, etc.).
    """
    if not is_mps_available():
        raise FileNotFoundError(
            f"{_MPS_CONTROL_BIN} not found on PATH. MPS ships with the CUDA "
            "toolkit; ensure CUDA development tools are installed in the "
            "container image."
        )

    if skip_if_running and is_mps_running(pipe_dir):
        logger.info("MPS daemon already running; not starting another.")
        return MpsHandle(pipe_dir=pipe_dir, log_dir=log_dir, started_by_us=False)

    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = {**os.environ, **mps_client_env(pipe_dir=pipe_dir, log_dir=log_dir)}
    logger.info(
        "Starting MPS control daemon (pipe_dir=%s, log_dir=%s)", pipe_dir, log_dir
    )
    try:
        # `-d` runs in daemon mode; the binary backgrounds itself and exits
        # 0 if it spawned successfully.
        subprocess.run(
            [_MPS_CONTROL_BIN, "-d"],
            env=env,
            check=True,
            timeout=30,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        # If the daemon was already running, a second `-d` call is harmless
        # but exits non-zero with a recognisable message. Treat as success.
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        if "already running" in stderr.lower():
            logger.info("MPS daemon already running (race-detected at start time).")
            return MpsHandle(pipe_dir=pipe_dir, log_dir=log_dir, started_by_us=False)
        raise RuntimeError(
            f"Failed to start MPS daemon (exit {e.returncode}): {stderr.strip()}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timed out starting MPS daemon: {e}") from e

    return MpsHandle(pipe_dir=pipe_dir, log_dir=log_dir, started_by_us=True)


def stop_mps_daemon(handle: Optional[MpsHandle] = None) -> bool:
    """Best-effort shutdown. Returns True iff we actually told a daemon to quit.

    The driver's atexit / Ray shutdown hook calls this. We deliberately
    swallow errors — leaving an orphan MPS daemon costs only a small idle
    process, whereas raising during cleanup would mask the real exception
    that triggered shutdown.
    """
    if not is_mps_available():
        return False

    pipe_dir = handle.pipe_dir if handle else DEFAULT_PIPE_DIR
    log_dir = handle.log_dir if handle else DEFAULT_LOG_DIR

    if not is_mps_running(pipe_dir):
        return False

    env = {**os.environ, **mps_client_env(pipe_dir=pipe_dir, log_dir=log_dir)}
    try:
        subprocess.run(
            [_MPS_CONTROL_BIN],
            input=b"quit\n",
            env=env,
            timeout=15,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        logger.info("Sent 'quit' to MPS control daemon.")
        return True
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Best-effort MPS shutdown failed: %s", e)
        return False


def setup_for_colocate(
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    *,
    register_atexit: bool = True,
) -> tuple[MpsHandle, dict[str, str]]:
    """One-shot: start daemon (if needed), return handle + client env.

    Convenience entry point for the Ray driver — mirrors the
    ``setup_for_colocate(...)`` signature the placement-group code will
    import in the next sub-task of Phase 1.

    Phase 6 hygiene: when ``register_atexit`` is true (default) and we
    actually started the daemon, register an ``atexit`` hook to
    ``stop_mps_daemon`` so a clean driver shutdown doesn't leak the
    daemon process. SIGKILL / OOM-kills bypass ``atexit`` of course;
    that's by design — the next driver run's ``start_mps_daemon`` is
    idempotent and will reuse a still-running daemon.
    """
    handle = start_mps_daemon(pipe_dir=pipe_dir, log_dir=log_dir)
    if register_atexit and handle.started_by_us:
        import atexit

        atexit.register(stop_mps_daemon, handle)
    return handle, mps_client_env(pipe_dir=pipe_dir, log_dir=log_dir)
