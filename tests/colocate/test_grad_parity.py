# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 7 — gradient parity between disagg and colocate (skeleton).

Plan reference: ``implementation.md`` §Phase 7 sub-task 1.

Goal: same prompts, same seed; one training step on disagg mode and one
on colocate mode → ``torch.allclose(g_disagg, g_colocate, atol=1e-6,
rtol=0)`` per parameter. (NCCL is bit-deterministic given identical
reduction order; we don't change the order, so we expect exact match
modulo floating-point reduce ordering.)

This depends on:
  - The upstream sglang patch (Phase 4 docs/colocate/sglang_patch.md)
    so the colocate path can run a full training step.
  - The disagg control config (existing dflash_trainer config) running
    one step too, with the same seed.
  - A small enough model that we can dump per-parameter gradients
    (``torch.save`` of every named_parameter.grad) — the plan suggests
    Qwen3-8B but for the unit-test sized parity check we'd use the
    smaller examples/qwen3-1.7b-eagle3 config or similar.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

pytest.skip(
    "Phase 7 grad parity depends on the upstream sglang patch "
    "(see docs/colocate/sglang_patch.md). Once both modes can run "
    "one step end-to-end, drop this skip and the test will dump and "
    "compare per-parameter gradients.",
    allow_module_level=True,
)


def test_phase7_grad_parity_per_parameter():
    """Per-parameter gradient parity between disagg and colocate.

    Implementation outline (post-patch):

    1. Load fixed RNG seed (``torch.manual_seed(args.seed)``).
    2. Run one training step in disagg mode → call
       ``extract_gradients(trainer.draft_model)`` and persist to
       ``/tmp/grad_disagg.pt``.
    3. Restart with same seed in colocate mode → run one step →
       ``extract_gradients`` again → persist to
       ``/tmp/grad_colocate.pt``.
    4. For each named parameter:
         assert torch.allclose(g_disagg[name], g_colocate[name],
                               atol=1e-6, rtol=0)

    The two runs share everything except the transfer mode: same
    optimizer init, same data ordering, same RNG. NCCL reduction
    order is the only thing that changes (Mooncake → memory; NCCL
    → P2P send), and at the per-rank level the trainer-side
    arithmetic is identical (FSDP all-gather + local backward).
    Hence: exact bit-equality is the right bar.
    """
    raise NotImplementedError(
        "Phase 7 grad parity skeleton — wait for upstream sglang patch."
    )
