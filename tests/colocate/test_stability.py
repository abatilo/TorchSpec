# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 6 — long-run memory stability skeleton (1000 steps).

Plan reference: ``implementation.md`` §Phase 6, "1000-step stability run
with `dflash_trainer` config: ``peak_alloc(step=10) ≈ peak_alloc(step=999)``
within 1%."

This is the slow (`@pytest.mark.slow`) counterpart to ``test_one_step``.
It depends on the same upstream sglang patch — without it, the engine
side of the union world never lights up and the test will hang on its
first ``recv_step``. The skeleton is parked here so the human submitter
can run it once the patch lands; the assertions are concrete (so they
won't silently pass) but the engine wiring is a TODO marker.

To run:

    modal run --detach --env sandbox \
        scripts/modal/modal_colocate_smoke.py::phase6_stability

When the upstream patch is in, drop the ``pytest.skip`` at the top.
"""

from __future__ import annotations

import os

import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")


# Default scale: trim for CI, override at the entrypoint level.
NUM_STEPS = int(os.environ.get("PHASE6_STABILITY_STEPS", "1000"))
SAMPLE_STEPS = (10, NUM_STEPS - 1)
PEAK_ALLOC_TOLERANCE = 0.01  # 1% per the plan.


pytest.skip(
    "Phase 6 stability run depends on the upstream sglang patch (see "
    "docs/colocate/sglang_patch.md). Once the patch is wired, drop this "
    "skip and the test will drive a 1000-step run and assert peak-alloc "
    "flatness.",
    allow_module_level=True,
)


def test_phase6_peak_alloc_flatness_over_1000_steps():
    """Drive ``NUM_STEPS`` colocate training steps; peak-alloc must be
    flat (within 1%) between step 10 and step ``NUM_STEPS - 1``.

    Implementation outline (post-patch):

    1. Spin up a 4×H100 placement group via the same fixture as
       ``test_one_step.py``.
    2. Wire trainer + engine actors with ``transfer_mode='nccl'``.
    3. Loop ``NUM_STEPS`` times:
         - controller.dispatch_colocate_batch.remote()
         - engines.generate_one_step()  # blocks until P2P send
         - trainers.train_one_step()    # blocks until P2P recv + step
         - every 100 steps: read trainer 0's peak_alloc metric
    4. Assert the last sampled peak-alloc is within 1% of the
       step-10 peak-alloc.

    The metric path (`Trainer._train_core_from_queue` already records
    ``perf/peak_bytes_allocated`` on every step; this test just samples
    it twice and compares.
    """
    raise NotImplementedError(
        "Phase 6 stability skeleton — wait for upstream sglang patch."
    )


def test_phase6_no_oom_under_load():
    """Under MPS+colocate, neither side should OOM during the 1000-step
    run. Test surface: the same loop above wrapped in a try/except for
    ``torch.cuda.OutOfMemoryError`` plus a check that
    ``ray.get_runtime_context().get_node_id`` is still alive at the end.
    """
    raise NotImplementedError(
        "Phase 6 stability skeleton — wait for upstream sglang patch."
    )
