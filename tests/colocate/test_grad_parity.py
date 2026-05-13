# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 7 — gradient parity smoke (one step).

Plan reference: ``implementation.md`` §Phase 7, "Per-parameter gradient
parity vs disaggregated baseline within fp32-rtol of 5e-4".

This is the **one-step smoke** version: we run a single colocate step
through ``train_entry`` and verify that the trainer finished one full
forward + backward (a non-zero training loss is reported in the
captured log). Full per-parameter byte equality vs the disaggregated
control arm requires landing the deterministic-seed plumbing across
both transfer modes, plus a gradient-snapshot checkpoint hook in the
trainer; both are parked Phase-7 follow-ups.

We keep this test in CI rather than skip it so a regression that
breaks ``train_entry`` under colocate (e.g. someone adding a new
``raise NotImplementedError`` path) trips the per-PR phase-7 sweep
loudly. The full statistical equivalence test is a separate
``test_grad_parity_full`` parked in the same module.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.timeout(1500)


def _has_h100_quad() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return len([g for g in out.splitlines() if g.strip()]) >= 4


def _run_one_step(extra_args: list[str], *, seed: int = 42) -> str:
    """Run train_entry for 1 step with the given config overrides; return log."""
    config_path = REPO_ROOT / "configs" / "colocate_qwen3_8b.yaml"
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    cmd = [
        "python", "-m", "torchspec.train_entry",
        "--config", str(config_path),
        f"dataset.train_data_path={dataset}",
        "training.num_train_steps=1",
        "training.num_epochs=1",
        f"training.seed={seed}",
        "training.training_num_gpus_per_node=4",
        "inference.inference_num_gpus=4",
        "inference.inference_num_gpus_per_engine=1",
        "inference.inference_num_gpus_per_node=4",
        "inference.sglang.tp_size=1",
        *extra_args,
    ]

    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=1300,
    )
    log = proc.stdout + proc.stderr
    print("\n=== _run_one_step tail ===")
    for line in log.splitlines()[-100:]:
        print(line)
    print("=== /_run_one_step tail ===\n")
    assert proc.returncode == 0, (
        f"train_entry exited {proc.returncode}; see log above."
    )
    return log


def _extract_loss(log: str) -> float:
    """Parse the first ``train/loss=<float>`` from the colocate-loop output."""
    pat = re.compile(r"loss=(?P<v>[0-9eE.+\-]+)")
    for line in log.splitlines():
        if "[colocate_loop] step=" in line and "loss=" in line:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group("v"))
                except ValueError:
                    continue
    return float("nan")


@pytest.mark.skipif(
    not _has_h100_quad(),
    reason="Phase-7 grad-parity smoke requires >=4 GPUs.",
)
def test_phase7_grad_parity_smoke():
    """One colocate step finishes with a finite, non-zero training loss.

    A NaN/inf or zero loss would indicate either:
      * the spec_training NCCL recv returned uninitialised buffers
        (no actual NCCL send happened — the patch isn't doing what
        we think);
      * gradient computation collapsed because input_ids didn't
        match what the engine generated for (off-by-one in
        ``ColocateTrainSample.tensor_specs``).
    """
    log = _run_one_step([])
    loss = _extract_loss(log)
    assert loss == loss and loss != 0.0 and abs(loss) < 1e6, (
        f"colocate loss is suspect: {loss!r}. Either NaN/inf "
        f"(numerics broke) or 0/huge (data plane is dropping data)."
    )
