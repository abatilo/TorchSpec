# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""NCCL P2P data fetcher for colocate mode (Phase 3).

This is the trainer-side counterpart to the engine's hidden-state writer.
Whereas the disaggregated path goes engine → Mooncake store → trainer
(``MooncakeDataFetcher``), the colocate path is engine → NCCL P2P send →
trainer recv into a pre-allocated buffer on the same physical GPU.

Phase 3 ships only the minimal building block:

    NcclDataFetcher(
        src_rank=engine_rank,
        shape=(B_eng_per_tp, S, H),
        dtype=torch.bfloat16,
        device=torch.device('cuda'),
    )
    tensor = fetcher.recv()  # blocks on dist.recv

The buffer is pre-allocated and re-used across calls so the per-step cost
is one ``cudaMemcpyDtoD`` (when ``clone=True``) or zero (when the caller
promises not to mutate the returned tensor).

Phase 4 will wrap this to also receive the aux-layer hidden states and
``last_hidden_states`` and assemble them into the same batch-dict shape
``MooncakeDataFetcher`` produces, so ``Eagle3Trainer._train_step`` doesn't
need to know which fetcher is wired up.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger("torchspec.training.nccl_data_fetcher")


class NcclDataFetcher:
    """Single-tensor NCCL P2P receiver with a pre-allocated buffer.

    Args:
        src_rank: Global rank to receive from (the paired engine rank in
            the union world).
        shape: Tensor shape to allocate. Must match exactly what the
            sender sends or NCCL will silently corrupt / hang.
        dtype: Tensor dtype.
        device: CUDA device to allocate on. Must be a real CUDA device
            because NCCL refuses CPU tensors.
        group: Optional ``ProcessGroup`` to use; defaults to the world
            (default PG). Tests pass a subgroup; production passes the
            union world's default PG.
        clone_on_return: If ``True`` (default), ``recv()`` returns a
            ``buffer.clone()`` so the caller can mutate freely. If
            ``False``, returns the buffer itself; the caller must finish
            using it before the next ``recv()`` call.
    """

    def __init__(
        self,
        src_rank: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
        clone_on_return: bool = True,
    ):
        if device.type != "cuda":
            raise ValueError(
                f"NcclDataFetcher requires a CUDA device; got device={device}"
            )

        self._src_rank = int(src_rank)
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device
        self._group = group
        self._clone = bool(clone_on_return)

        # Pre-allocate the recv buffer. Phase 6 will verify that this
        # allocation lives in expandable_segments territory so it
        # doesn't fragment the pool.
        self._buffer = torch.empty(self._shape, dtype=self._dtype, device=self._device)

        logger.debug(
            "NcclDataFetcher initialised: src_rank=%d shape=%s dtype=%s device=%s "
            "clone_on_return=%s",
            self._src_rank, self._shape, self._dtype, self._device, self._clone,
        )

    @property
    def buffer_shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def src_rank(self) -> int:
        return self._src_rank

    def recv(self) -> torch.Tensor:
        """Block on a single P2P recv from ``src_rank``.

        Uses ``dist.batch_isend_irecv`` rather than ``dist.recv`` because
        unbatched send/recv on a large parent group serialises through
        NCCL's lazy 2-rank sub-communicator init, which can deadlock
        across multiple pairs (PyTorch warns
        ``ProcessGroupNCCL.cpp:4004``). Batched P2P is its own primitive
        class and always handled correctly by NCCL.

        Returns:
            The received tensor (a clone by default; the underlying
            buffer if ``clone_on_return=False``).
        """
        op = dist.P2POp(dist.irecv, self._buffer, peer=self._src_rank, group=self._group)
        works = dist.batch_isend_irecv([op])
        for work in works:
            work.wait()
        return self._buffer.clone() if self._clone else self._buffer


def make_dummy_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic dummy tensor used as the Phase 3 send payload.

    Uses ``torch.arange`` rather than ``torch.rand`` so byte-equality is
    well-defined (no RNG state to coordinate). The optional ``seed``
    offsets every element so successive iterations send distinct payloads
    — that catches a class of bugs where the receiver "passes" simply
    because the buffer didn't change between iterations.
    """
    n = 1
    for d in shape:
        n *= d
    flat = (torch.arange(n, device=device, dtype=torch.float32) + float(seed))
    return flat.reshape(shape).to(dtype)


def send_dummy(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    dst_rank: int,
    *,
    seed: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Engine-side helper that builds a deterministic tensor and sends it.

    Mirrors ``NcclDataFetcher.recv``: uses batched P2P to side-step the
    lazy-init pathology of unbatched send on large parent groups.

    Returns the tensor it sent (so a caller can keep it alive until the
    receive completes if they care to verify locally).
    """
    tensor = make_dummy_tensor(shape, dtype=dtype, device=device, seed=seed)
    op = dist.P2POp(dist.isend, tensor, peer=dst_rank, group=group)
    works = dist.batch_isend_irecv([op])
    for work in works:
        work.wait()
    return tensor
