# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from argparse import Namespace
from datetime import timedelta

import torch.distributed as dist

from torchspec import AutoDraftModelConfig
from torchspec.colocate.world import (
    ROLE_TRAINER,
    UnionWorldSpec,
    init_union_world,
)
from torchspec.models.draft.dflash import DFlashConfig
from torchspec.ray.ray_actor import RayActor
from torchspec.training.eagle3_trainer import Eagle3Trainer
from torchspec.utils.distributed import init_gloo_group, init_usp_groups
from torchspec.utils.logging import setup_file_logging


# Port offset used by the colocate union-world rendezvous so it doesn't
# clobber the trainer's own MASTER_PORT (used by FSDP / gloo
# initialisation when transfer_mode == 'mooncake'). Phase 4 picks +5000;
# trainer port range is (20000, 21000), engine port allocation lives
# above that, so 25000+ stays clear.
_COLOCATE_UNION_WORLD_PORT_OFFSET = 5000


class TrainerActor(RayActor):
    def __init__(self, world_size: int, rank: int, master_addr: str, master_port: int):
        self._world_size = world_size
        self._rank = rank

        self.setup_master(master_addr, master_port, port_range=(20000, 21000))

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)

        self.setup_gpu()
        setup_file_logging("training", self._rank)

    def _init_distributed_colocate(self, args: Namespace) -> None:
        """Phase 4: bring up the union NCCL world as the default PG.

        In colocate (`transfer_mode='nccl'`) mode the trainer + engine
        ranks share one default PG of size ``2N`` so the engine can do a
        ``dist.send`` to its paired trainer with no shared store.

        The trainer process is the easy half. The engine side must be
        bootstrapped from inside sglang's TP scheduler subprocess by an
        upstream sglang patch (see ``docs/colocate/sglang_patch.md``).
        We surface the rendezvous params via env vars so the patch can
        read them out of the scheduler subprocess's env without needing
        a side-channel:

          - ``TORCHSPEC_COLOCATE_UNION_MASTER_ADDR``
          - ``TORCHSPEC_COLOCATE_UNION_MASTER_PORT``
          - ``TORCHSPEC_COLOCATE_UNION_WORLD_SIZE`` (= 2N)
          - ``TORCHSPEC_COLOCATE_UNION_N_PER_ROLE`` (= N)
          - ``TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN``

        Setting these on the *trainer* process won't affect the engine
        subprocesses directly — that's what the SglEngine env-export +
        sglang patch is for. We set them here for parity / debugging.
        """
        spec = UnionWorldSpec(
            n_per_role=self._world_size,
            master_addr=self.master_addr,
            master_port=int(self.master_port) + _COLOCATE_UNION_WORLD_PORT_OFFSET,
            timeout_minutes=int(getattr(args, "distributed_timeout_minutes", 30)),
        )

        os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_ADDR"] = spec.master_addr
        os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_PORT"] = str(spec.master_port)
        os.environ["TORCHSPEC_COLOCATE_UNION_WORLD_SIZE"] = str(spec.world_size)
        os.environ["TORCHSPEC_COLOCATE_UNION_N_PER_ROLE"] = str(spec.n_per_role)
        os.environ["TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN"] = str(spec.timeout_minutes)

        union = init_union_world(spec, role=ROLE_TRAINER, role_rank=self._rank)
        self._union_world = union

    def init(self, args: Namespace, role: str, mooncake_config=None, with_ref: bool = False) -> int:
        self.args = args
        self._union_world = None

        transfer_mode = getattr(args, "transfer_mode", None) or "mooncake"
        is_colocate_nccl = transfer_mode == "nccl"

        if is_colocate_nccl:
            # Colocate path: union world is the default PG. We do NOT
            # call dist.init_process_group separately — init_union_world
            # owns that.
            self._init_distributed_colocate(args)
        else:
            backend = getattr(args, "distributed_backend", "nccl")
            if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "fsdp_cpu_backend", None):
                cpu_backend = args.fsdp_cpu_backend
                backend = f"cpu:{cpu_backend},cuda:{backend}"

            dist.init_process_group(
                backend=backend,
                timeout=timedelta(minutes=getattr(args, "distributed_timeout_minutes", 30)),
            )

        if getattr(args, "attention_backend", None) == "usp":
            if is_colocate_nccl:
                # USP+colocate is explicitly punted in implementation.md
                # §"Out-of-scope". The validation in colocate/config.py
                # also rejects this combo before we get here, but
                # belt-and-braces the check here so a stale config
                # doesn't silently produce wrong gradients.
                raise RuntimeError(
                    "USP attention + colocate (transfer_mode='nccl') is not "
                    "supported. Set training.attention_backend to a non-USP "
                    "backend, or switch to transfer_mode='mooncake'."
                )
            init_usp_groups(
                sp_ulysses_size=getattr(args, "sp_ulysses_size", 1),
                sp_ring_size=getattr(args, "sp_ring_size", 1),
            )

        if is_colocate_nccl:
            # init_union_world already built an all-rank gloo subgroup
            # (meta_group). Bind it as the module-global GLOO_GROUP so
            # downstream get_gloo_group() returns it. This avoids
            # creating yet another gloo group on the 2N-rank union
            # world, which would trigger an extra TCP rendezvous.
            from torchspec.utils import distributed as _dist_utils

            _dist_utils.GLOO_GROUP = self._union_world.meta_group

            # In colocate mode, the default PG is the 2N-rank union
            # world, but FSDP / per-trainer code assumes
            # ``args.rank ∈ [0, N)`` and ``args.world_size == N``.
            # Override here so all downstream rank-arithmetic stays in
            # the trainer subgroup space. The union-world handle is
            # accessible via ``self._union_world`` if anything needs the
            # 2N view (e.g. the colocate data fetcher to compute the
            # paired engine rank).
            args.rank = self._union_world.role_rank
            args.world_size = self._union_world.spec.n_per_role
        else:
            init_gloo_group()

            args.rank = dist.get_rank()
            args.world_size = dist.get_world_size()

        draft_model_config = getattr(args, "draft_model_config_obj", None)
        if draft_model_config is None and getattr(args, "draft_model_config", None):
            draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

        # Config-based trainer dispatch: DFlashConfig → DFlashTrainer, else Eagle3
        if isinstance(draft_model_config, DFlashConfig):
            from torchspec.training.dflash_trainer import DFlashTrainer

            self._trainer = DFlashTrainer(args)
        else:
            self._trainer = Eagle3Trainer(args)

        target_model_path = getattr(args, "target_model_path", None)

        if draft_model_config is not None:
            self._trainer.init_model(
                draft_model_config=draft_model_config,
                target_model_path=target_model_path,
                mooncake_config=mooncake_config,
            )

        # Forward the union-world handle to the trainer so its
        # set_train_queue / set_eval_queue can build the colocate
        # NcclMultiTensorFetcher with the right paired engine rank.
        # No-op for the disaggregated path (Trainer ignores it).
        if hasattr(self._trainer, "set_union_world"):
            self._trainer.set_union_world(self._union_world)

        return 0

    def train_from_queue(self, step: int, num_batches: int) -> dict:
        return self._trainer.train_from_queue(step, num_batches)

    def set_train_queue(self, queue, mooncake_config=None, per_dp_rank_batch_size: int = 1):
        return self._trainer.set_train_queue(
            queue, mooncake_config=mooncake_config, per_dp_rank_batch_size=per_dp_rank_batch_size
        )

    def get_union_world_paired_rank(self) -> int:
        """Return the paired engine global rank in the union world.

        Trainer-side colocate clients (the controller, mostly) use this
        to assert the engine-side env got configured with the matching
        rank. Raises if colocate isn't initialised on this actor.
        """
        if self._union_world is None:
            raise RuntimeError(
                "TrainerActor.get_union_world_paired_rank called but the "
                "union world is not initialised on this actor. Either "
                "transfer_mode != 'nccl' or init() hasn't run yet."
            )
        return self._union_world.paired_global_rank

    def get_global_step(self) -> int:
        return self._trainer.global_step

    def save_model(self, step: int, force_sync: bool = False) -> None:
        self._trainer.save_model(step, force_sync)

    def save_draft_model_for_serving(self, output_dir: str) -> None:
        self._trainer.save_draft_model_for_serving(output_dir)

    def set_vocab_buffers(self, d2t, t2d) -> None:
        if hasattr(self._trainer, "draft_model") and hasattr(
            self._trainer.draft_model, "set_vocab_buffers"
        ):
            self._trainer.draft_model.set_vocab_buffers(d2t, t2d)
        else:
            raise AttributeError(
                "set_vocab_buffers called but draft model does not support vocab pruning. "
                "DFlash training should not use vocab pruning — check train_entry config."
            )

    def set_eval_queue(self, queue, mooncake_config=None, per_dp_rank_batch_size: int = 1):
        return self._trainer.set_eval_queue(
            queue, mooncake_config=mooncake_config, per_dp_rank_batch_size=per_dp_rank_batch_size
        )

    def cache_eval_samples(self, count: int) -> int:
        return self._trainer.cache_eval_samples(count)

    def save_eval_cache(self, cache_dir: str) -> None:
        return self._trainer.save_eval_cache(cache_dir)

    def load_eval_cache(self, cache_dir: str) -> int:
        return self._trainer.load_eval_cache(cache_dir)

    def eval_from_cache(self) -> dict:
        return self._trainer.eval_from_cache()
