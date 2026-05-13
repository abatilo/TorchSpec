# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 7 — convergence parity over 1k steps (slow skeleton).

Plan reference: ``implementation.md`` §Phase 7 sub-task 2.

Goal: 1000 steps on ``qwen3-8b-single-node`` with both transfer modes,
assert per-step training loss within 1-2% across modes.

This is the long-run cousin of ``test_grad_parity``. It catches drift
that a single-step parity check would miss (e.g., subtle ordering bugs
that don't surface until enough optimizer steps have accumulated).

Depends on:
  - Upstream sglang patch (Phase 4 ``docs/colocate/sglang_patch.md``).
  - 1000-step run on each mode (~30 min × 2 on 8×H100).
  - Loss-curve persistence + comparison utility.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.slow

pytest.skip(
    "Phase 7 convergence depends on the upstream sglang patch "
    "(see docs/colocate/sglang_patch.md) and is a multi-hour run. "
    "Drop this skip once the patch is in and you have a budget for "
    "two 1000-step runs.",
    allow_module_level=True,
)


def test_phase7_convergence_curves_match_within_2pct():
    """Per-step loss is within 2% between disagg and colocate.

    Implementation outline (post-patch):

    1. Run 1000 steps disagg with deterministic data ordering; persist
       ``loss_per_step_disagg.csv``.
    2. Run 1000 steps colocate with the same seed; persist
       ``loss_per_step_colocate.csv``.
    3. For each step:
         |loss_disagg[i] - loss_colocate[i]| / loss_disagg[i] < 0.02
       (looser bar than per-parameter gradient parity because:
        - cumulative numerical drift over 1000 optimizer steps,
        - any sampling-related noise in the data path).
    """
    raise NotImplementedError(
        "Phase 7 convergence skeleton — wait for upstream sglang patch."
    )


def test_phase7_eval_loss_matches():
    """Eval loss on cached eval batches matches between modes.

    Same eval batches, same vocab mapping, same draft model state
    (loaded from a fixed colocate checkpoint). Eval loss must agree
    to within tokenizer-deterministic noise (≈ 1e-4 absolute).
    """
    raise NotImplementedError(
        "Phase 7 eval-loss skeleton — wait for upstream sglang patch."
    )
