# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Union NCCL world bootstrap for colocate mode (Phase 2).

The colocate plan puts trainer and engine processes on the same physical
GPUs. To send hidden states from the engine to the trainer over NCCL P2P,
both sides must be members of one NCCL world of size ``2 * N`` (N =
training_world_size). This module provides:

- A small ``UnionWorldSpec`` dataclass capturing rendezvous params.
- ``rank_for_role(world_size, role, role_rank)`` — the canonical
  rank-assignment scheme from ``implementation.md`` §Phase 2: trainer ranks
  ``0..N-1``, engine ranks ``N..2N-1``.
- ``init_union_world(spec)`` — initialises the **default** torch.distributed
  PG for the calling process so it sees a 2N-rank world, plus exposes the
  FSDP-only subgroup ``ranks=[0..N-1]`` and a gloo CPU subgroup spanning
  all ranks (for step-metadata broadcast).

**Important**: the trainer side is the easy half. The engine side has a
known wrinkle — sglang internally calls ``dist.init_process_group`` for
its own TP group, and PyTorch only allows one *default* PG per process.
``init_union_world`` writes a small marker into the env so a later
sglang-patch hook can:

  - Skip its own ``init_process_group`` call when our union world is
    already the default (``TORCHSPEC_UNION_WORLD_INITIALIZED=1``), or
  - Reconstruct sglang's TP via ``dist.new_group`` against our union world
    using the rank list it would have used otherwise.

That patch lives in ``patches/_sglang/`` (Phase 2 sub-task 5) and is
exercised by the Phase 2 Modal smoke test.

For Phase 2 we ship:

  1. This helper, fully unit-tested against torch.distributed semantics.
  2. A trainer-side init path that uses it.
  3. A standalone NCCL barrier test: 4 trainer-shape + 4 engine-shape
     processes (no sglang), all join the union world, all
     ``dist.barrier()``.

Phase 2 *does not* require sglang to use the union world for its own TP
yet — that's Phase 4's hidden-state hook. We just need the mechanism to
exist and the 8-rank barrier to succeed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

logger = logging.getLogger("torchspec.colocate.world")

# Roles for the union-world rank-assignment helper. Names match the
# ``role`` argument passed to ``RayTrainGroup.async_init`` /
# ``SglEngine.init`` so the call sites read naturally.
ROLE_TRAINER = "training"
ROLE_ENGINE = "inference"

# Marker we set in os.environ once the union world is up. Read by the
# sglang patch (or any other downstream code) to know the default PG is
# already a 2N-rank world and not a vanilla per-process one.
UNION_WORLD_ENV_MARKER = "TORCHSPEC_COLOCATE_UNION_WORLD"


@dataclass(frozen=True)
class UnionWorldSpec:
    """Parameters needed to bootstrap the union NCCL world on every rank.

    The driver computes this once and broadcasts it to all 2N actors via
    Ray. Ranks join collectively.
    """

    n_per_role: int
    """Number of ranks per role (trainer count == engine count == N)."""

    master_addr: str
    """IP/hostname of the rendezvous master (any 1 actor's IP works)."""

    master_port: int
    """Free TCP port on master_addr; pre-checked by the driver."""

    timeout_minutes: int = 30
    """init_process_group timeout. NCCL default is 10 min, which is too
    short for cold starts where one side might be slower to boot."""

    @property
    def world_size(self) -> int:
        return 2 * self.n_per_role

    @property
    def init_method(self) -> str:
        return f"tcp://{self.master_addr}:{self.master_port}"


def rank_for_role(spec: UnionWorldSpec, role: str, role_rank: int) -> int:
    """Map (role, role_rank) → global rank in the union world.

    Trainers occupy ranks ``[0, N)``, engines occupy ``[N, 2N)``.

    Raises:
        ValueError: unknown role, or role_rank out of range.
    """
    if role == ROLE_TRAINER:
        if not 0 <= role_rank < spec.n_per_role:
            raise ValueError(
                f"trainer role_rank {role_rank} out of range [0, {spec.n_per_role})"
            )
        return role_rank
    if role == ROLE_ENGINE:
        if not 0 <= role_rank < spec.n_per_role:
            raise ValueError(
                f"engine role_rank {role_rank} out of range [0, {spec.n_per_role})"
            )
        return spec.n_per_role + role_rank
    raise ValueError(
        f"unknown role {role!r}; expected {ROLE_TRAINER!r} or {ROLE_ENGINE!r}"
    )


def trainer_global_ranks(spec: UnionWorldSpec) -> list[int]:
    """Convenience: union-world ranks held by trainers (= [0..N))."""
    return list(range(spec.n_per_role))


def engine_global_ranks(spec: UnionWorldSpec) -> list[int]:
    """Convenience: union-world ranks held by engines (= [N..2N))."""
    return list(range(spec.n_per_role, 2 * spec.n_per_role))


@dataclass
class UnionWorld:
    """Live handle to the initialised union world for one rank.

    Returned by ``init_union_world``. Holds references to the subgroups so
    callers can pass them to FSDP / collective ops without re-deriving.
    """

    spec: UnionWorldSpec
    role: str
    role_rank: int
    global_rank: int
    fsdp_group: object  # torch.distributed.ProcessGroup
    """Subgroup of just trainer ranks; pass to FSDP DeviceMesh.

    On engine ranks this is set to ``None`` because the engine is not in
    the FSDP group; calling collectives on it from an engine would hang."""
    meta_group: object  # torch.distributed.ProcessGroup
    """Gloo subgroup spanning all 2N ranks. Used for CPU-side step
    metadata broadcast (cheap dict broadcast, no GPU needed)."""


def init_union_world(spec: UnionWorldSpec, role: str, role_rank: int) -> UnionWorld:
    """Collective: initialise the union world from this process.

    All 2N ranks must call this with consistent ``spec`` (same master_addr,
    master_port, n_per_role) and the right ``role`` / ``role_rank``.

    Side-effects:
        - Calls ``dist.init_process_group(backend='nccl', world_size=2N, …)``.
          The default PG of this process becomes the union world.
        - Calls ``dist.new_group`` twice (collective on all 2N ranks):
          once for the trainer-only NCCL subgroup, once for the gloo
          all-rank metadata subgroup.
        - Sets ``TORCHSPEC_COLOCATE_UNION_WORLD`` env marker so downstream
          code (e.g. sglang patches) can detect the union-world setup.

    Returns:
        UnionWorld handle with the subgroup references.

    Raises:
        RuntimeError: if a default PG is already initialised. This is the
            integration-with-sglang risk flagged in implementation.md
            §Phase 2 risk register.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed default group is already initialised. The colocate "
            "union world must be the default group; call init_union_world *before* "
            "any other framework (FSDP, sglang, etc.) initialises its own world. "
            "Set role=engine and patch sglang to skip its own init_process_group "
            "when TORCHSPEC_COLOCATE_UNION_WORLD=1."
        )

    global_rank = rank_for_role(spec, role, role_rank)

    logger.info(
        "Initialising union world: role=%s role_rank=%d global_rank=%d "
        "world_size=%d init_method=%s",
        role, role_rank, global_rank, spec.world_size, spec.init_method,
    )

    dist.init_process_group(
        backend="nccl",
        world_size=spec.world_size,
        rank=global_rank,
        init_method=spec.init_method,
        timeout=timedelta(minutes=spec.timeout_minutes),
    )

    # Subgroups are collective: every rank must call new_group with the
    # same args, even ranks not in the resulting subgroup.
    fsdp_ranks = trainer_global_ranks(spec)
    fsdp_group = dist.new_group(ranks=fsdp_ranks, backend="nccl")
    if role != ROLE_TRAINER:
        # Engines aren't in the FSDP group; expose None so calling
        # FSDP collectives on this is a clear error rather than a hang.
        fsdp_group_for_role: Optional[object] = None
    else:
        fsdp_group_for_role = fsdp_group

    meta_group = dist.new_group(
        ranks=list(range(spec.world_size)), backend="gloo"
    )

    os.environ[UNION_WORLD_ENV_MARKER] = "1"

    return UnionWorld(
        spec=spec,
        role=role,
        role_rank=role_rank,
        global_rank=global_rank,
        fsdp_group=fsdp_group_for_role,
        meta_group=meta_group,
    )


def union_world_ready() -> bool:
    """Cheap query for downstream code (e.g. the sglang patch hook)."""
    return os.environ.get(UNION_WORLD_ENV_MARKER) == "1"
