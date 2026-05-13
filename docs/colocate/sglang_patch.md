# Upstream sglang patch surface for the colocate (NCCL) path

> Phase 4 of [`implementation.md`](implementation.md) requires a small
> set of changes inside sglang itself. This doc enumerates the exact
> patch surface so a human submitter can drive the upstream PR (or, in
> the meantime, maintain a fork).

## Motivation

In disaggregated mode, sglang's spec_training callback writes hidden
states to a Mooncake KV store keyed by a UUID, then the trainer reads
from Mooncake. In colocate mode (`transfer_mode=nccl`) the trainer +
engine ranks share one **union NCCL world** of size `2N` (N trainers
+ N engine TP workers, paired by rank). The engine writes hidden states
**directly** to its paired trainer rank via `dist.batch_isend_irecv` on
that union world — no shared store, no serialisation overhead.

The TorchSpec side of the wire is already in this repo:

- Engine-side sender:
  [`torchspec/inference/engine/nccl_hidden_states_connector.py`](../../torchspec/inference/engine/nccl_hidden_states_connector.py)
  — `NcclHiddenStatesConnector(dst_global_rank).send(tensors)`.
- Trainer-side receiver:
  [`torchspec/training/nccl_data_fetcher.py`](../../torchspec/training/nccl_data_fetcher.py)
  — `NcclMultiTensorFetcher(src_global_rank, device).recv_step(specs)`.
- Union-world bootstrap:
  [`torchspec/colocate/world.py`](../../torchspec/colocate/world.py).

What's missing is the **engine-process side of the bootstrap**: sglang
itself must (a) skip its own `dist.init_process_group` when our union
world is already up, or (b) join the union world and re-derive its TP
group from a slice of it; and (c) route the spec_training callback to
the new `NcclHiddenStatesConnector` instead of the Mooncake writer.

## Env-var contract

The TorchSpec driver exports the following env vars before launching
sglang. Read them from inside sglang's TP scheduler subprocess:

| env var | meaning |
|---|---|
| `TORCHSPEC_COLOCATE_TRANSFER_MODE` | Set to `"nccl"` when colocate is on. Set the spec_training callback path accordingly. Empty / unset means stay on the legacy Mooncake path. |
| `TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK` | Global rank in the union world to send hidden states to. |
| `TORCHSPEC_COLOCATE_UNION_MASTER_ADDR` | Rendezvous host for `init_process_group`. |
| `TORCHSPEC_COLOCATE_UNION_MASTER_PORT` | Rendezvous port. |
| `TORCHSPEC_COLOCATE_UNION_WORLD_SIZE` | `2N` — total ranks in the union world. |
| `TORCHSPEC_COLOCATE_UNION_N_PER_ROLE` | `N` — number of trainer / engine ranks. The engine TP scheduler is at union global rank `N + sglang_tp_rank`. |
| `TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN` | `init_process_group` timeout in minutes. Use this exact value — the trainer side already booted the rendezvous and will wait this long. |
| `TORCHSPEC_COLOCATE_UNION_WORLD` | Set to `"1"` once the union world is initialised. The patch can use this as a "torch.dist already brought up" sentinel. |

## Patch points

The patch is small but lives in three sglang files. Pseudo-paths are
shown for the layout that's been stable in sglang since ~mid-2024; they
may shift slightly if the upstream refactor changes.

### 1. Distributed init: `sglang/srt/distributed/parallel_state.py` (or equivalent)

When the scheduler subprocess boots, it normally calls
`torch.distributed.init_process_group` to bring up its TP world. In
colocate mode, the union world is the default PG; sglang should join it
instead of creating a new default.

Pseudocode:

```python
import os
import torch.distributed as dist
from datetime import timedelta

def _maybe_join_torchspec_union_world():
    if os.environ.get("TORCHSPEC_COLOCATE_TRANSFER_MODE") != "nccl":
        return False  # disaggregated path — no-op

    if dist.is_initialized():
        # Trainer's init_union_world already ran in this process —
        # nothing to do. (This branch fires when the engine and
        # trainer happen to share a Python process; not the common
        # case but possible in tests.)
        return True

    addr = os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_ADDR"]
    port = int(os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_PORT"])
    world_size = int(os.environ["TORCHSPEC_COLOCATE_UNION_WORLD_SIZE"])
    n_per_role = int(os.environ["TORCHSPEC_COLOCATE_UNION_N_PER_ROLE"])
    timeout = int(os.environ.get("TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN", "30"))

    # Engines occupy ranks [N, 2N). The current TP rank determines our
    # offset within the engine block.
    tp_rank = int(os.environ.get("TP_RANK", os.environ.get("RANK", "0")))
    global_rank = n_per_role + tp_rank

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        init_method=f"tcp://{addr}:{port}",
        timeout=timedelta(minutes=timeout),
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )

    # The TP group sglang would normally create with new_group is now a
    # subgroup of the 2N-rank default PG; the rank list is contiguous.
    tp_world_ranks = list(range(n_per_role, 2 * n_per_role))
    tp_group = dist.new_group(ranks=tp_world_ranks, backend="nccl")
    return True, tp_group
```

The exact integration pattern depends on how sglang's distributed init
is structured. The key invariants:

- Default PG must be the 2N-rank union world after this runs.
- sglang's TP group is `dist.new_group(ranks=range(N, 2N))` — a
  contiguous slice of the engine half of the union world.
- All trainer ranks have already joined the rendezvous via
  `init_union_world` (TorchSpec side); the engine joining is what
  unblocks them.

### 2. spec_training callback: `sglang/srt/managers/scheduler.py` (or wherever `enable_spec_training_mooncake` is consumed)

The callback today writes to `EagleMooncakeStore` keyed by `mooncake_key`.
In colocate mode, route to the NCCL connector instead. Pseudo-code:

```python
import os

def _build_hidden_states_writer():
    transfer_mode = os.environ.get("TORCHSPEC_COLOCATE_TRANSFER_MODE", "")
    if transfer_mode == "nccl":
        from torchspec.inference.engine.nccl_hidden_states_connector import (
            NcclHiddenStatesConnector,
        )
        dst = int(os.environ["TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK"])
        return NcclHiddenStatesConnector(dst_global_rank=dst)
    else:
        return _build_mooncake_writer()  # existing path
```

In the callback itself:

```python
def on_spec_training_step(hidden_states, aux_hidden_states, last_hidden_states, target_logits):
    if isinstance(writer, NcclHiddenStatesConnector):
        writer.send({
            "hidden_states": hidden_states,
            "aux_hidden_states": aux_hidden_states,
            "last_hidden_states": last_hidden_states,
            "target_logits": target_logits,
        })
    else:
        writer.put(mooncake_key, ...)  # existing Mooncake path
```

The **dict key set** must match what TorchSpec's controller ships in
`ColocateTrainSample.tensor_specs` — see
[`torchspec/training/data_fetcher.py`](../../torchspec/training/data_fetcher.py)
`class ColocateTrainSample`. Both sides walk `sorted(keys)` so insertion
order doesn't matter.

The tensors **must be contiguous and on CUDA**. The connector raises
`ValueError` otherwise.

The callback runs **only on TP rank 0** today (it's the rank that
coordinates the Mooncake write). For colocate, every TP rank participates
in the P2P send because the trainer side has one fetcher per trainer
rank (paired 1:1 with engine TP ranks). Either:

  - Move the callback to fire on every TP rank, OR
  - Do an all-gather on TP rank 0 first and then send the shards out.

The former is simpler and matches the way the trainer expects to
receive (one shard per trainer rank). The Phase-4 plan in
`implementation.md` §"sglang patch" §1 makes this explicit:
*"Local-chunks: shard_i = hidden_states[i*B_eng/TP : (i+1)*B_eng/TP]
where i = engine.tp_rank."*

### 3. (Optional) Skip the Mooncake setup completely

When `enable_spec_training_mooncake=False`, sglang's existing flag flow
already skips the Mooncake bootstrap. TorchSpec sets the flag from
[`torchspec/inference/engine/sgl_engine.py`](../../torchspec/inference/engine/sgl_engine.py)
based on `transfer_mode`. No extra patch needed here as long as the flag
is honoured.

## Verification

After the patch lands:

```bash
modal run --env sandbox \
    scripts/modal/modal_colocate_smoke.py::phase4_one_step
```

This runs `tests/colocate/test_one_step.py` end-to-end on a 4×H100 box:
1 engine × TP=4 + 4 trainers × FSDP=4, all sharing GPUs via MPS, hidden
states moving over the union world. The plan's §Phase 4 done-criterion
("loss is finite and non-zero") is checked there.

Without the patch, that test will **hang on the first P2P recv** because
the engine's spec_training callback is still writing to a (now disabled)
Mooncake store and the trainer's `NcclMultiTensorFetcher.recv_step` is
waiting for tensors that never arrive. This hang is the diagnostic — if
you see it, the patch isn't being picked up.

## Test surface available without the patch

`tests/colocate/test_p2p_multi_tensor.py` exercises the connector +
fetcher + union-world integration **without** sglang involvement
(both sides are Ray actors that call the connector directly). Modal
entrypoint: `phase4_multi_tensor`. This is the maximal e2e check that
runs in this repo today.
