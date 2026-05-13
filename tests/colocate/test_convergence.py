# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 7 — short-run convergence (slow).

Plan reference: ``implementation.md`` §Phase 7, "Short-horizon
convergence: 1k step training loss curve overlaps within 2% of the
disaggregated baseline."

This is the slow (``@pytest.mark.slow``) counterpart to
``test_grad_parity.py``. It runs a short colocate training horizon
and asserts the loss curve trends downward (i.e., training is making
real progress — not a no-op or constant signal). The full disagg
side-by-side comparison (within 2 % at every step) requires running
two configs back-to-back on the same Modal job; that's a separate
``test_convergence_disagg_overlap`` parked here as a follow-up.

Default horizon: 50 steps. Override with ``PHASE7_CONVERGE_STEPS``
(the plan's reference is 1000 but that's an hour of compute under
MPS; CI only needs to see a clear downward trend).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.colocate._mps_probe import has_h100_quad, mps_works

REPO_ROOT = Path(__file__).resolve().parents[2]

NUM_STEPS = int(os.environ.get("PHASE7_CONVERGE_STEPS", "50"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(60 * 60),
]


def _losses_from_log(log: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    pat = re.compile(
        r"\[colocate_loop\] step=(?P<step>\d+).*?loss=(?P<v>[0-9eE.+\-]+)"
    )
    for line in log.splitlines():
        m = pat.search(line)
        if m:
            try:
                out.append((int(m.group("step")), float(m.group("v"))))
            except ValueError:
                continue
    return out


@pytest.mark.skipif(
    not has_h100_quad(),
    reason="Phase-7 convergence requires >=4 GPUs.",
)
@pytest.mark.skipif(
    not mps_works(),
    reason=(
        "Phase-7 convergence needs the colocate path to actually run, "
        "which needs working NVIDIA MPS (see tests/colocate/_mps_probe.py)."
    ),
)
def test_phase7_convergence_loss_decreases():
    """After ``NUM_STEPS`` colocate steps the average late-window loss
    is below the average early-window loss. Drives the same loop as
    Phase 4 / 6 but for many steps; this is the cheapest e2e signal
    that the gradient is actually flowing (the trainer is updating
    weights from real engine-supplied hidden states)."""

    config_path = REPO_ROOT / "configs" / "colocate_qwen3_8b.yaml"
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    proc = subprocess.run(
        [
            "python", "-m", "torchspec.train_entry",
            "--config", str(config_path),
            f"dataset.train_data_path={dataset}",
            f"training.num_train_steps={NUM_STEPS}",
            "training.num_epochs=1",
            "training.training_num_gpus_per_node=4",
            "inference.inference_num_gpus=4",
            "inference.inference_num_gpus_per_engine=1",
            "inference.inference_num_gpus_per_node=4",
            "inference.sglang.tp_size=1",
        ],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        timeout=60 * 60 - 30,
    )

    log = proc.stdout + proc.stderr
    print("\n=== last 200 lines ===")
    for line in log.splitlines()[-200:]:
        print(line)
    print("=== /last 200 lines ===\n")
    assert proc.returncode == 0, f"train_entry exited {proc.returncode}"

    losses = _losses_from_log(log)
    assert len(losses) >= max(2, NUM_STEPS // 10), (
        f"only captured {len(losses)} loss points; expected at least "
        f"~{NUM_STEPS // 10}. The colocate loop's metric flush "
        f"may have changed format."
    )
    early = sum(v for _, v in losses[: max(1, len(losses) // 4)])
    late = sum(v for _, v in losses[-max(1, len(losses) // 4):])
    early /= max(1, len(losses) // 4)
    late /= max(1, len(losses) // 4)
    assert late < early, (
        f"loss did not decrease: early={early:.4f} late={late:.4f}. "
        f"Either the gradient isn't flowing (NCCL recv buffers are "
        f"uninitialised) or LR/dtype is wrong for the colocate path."
    )
