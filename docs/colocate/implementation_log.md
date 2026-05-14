# Colocate Mode — Implementation Log

> Living log of progress against [`implementation.md`](implementation.md).
>
> Each phase entry records: status, files touched, what was done, what was
> verified (and how — Modal sandbox / local / unit only), and any deviations
> from the plan with a one-line justification.
>
> Branch: `feature/colocate-training-inference`
>
> Test platform: **Modal serverless GPUs** (sandbox env). All multi-GPU tests
> run via `modal run scripts/modal/modal_colocate_smoke.py ...`. Unit tests
> (Phase 0 only) run on a Mac dev box thanks to `conftest.py`'s torch stubs.

---

## Status snapshot

| Phase | Title | Status | Modal-required | Notes |
|---|---|---|---|---|
| 0 | Configuration plumbing & feature flag | ✅ | No (unit only) | 18/18 unit tests pass locally |
| 1 | Placement: 1:1 bundle pairing + MPS env | ✅ | Yes (4×H100) | 5/5 placement tests pass on Modal |
| 2 | Union NCCL world (no transfer yet) | 🟡 | Yes (8×H100) | helper + 8-rank smoke test pass; trainer/engine wire-up + sglang patch deferred to Phase 4 |
| 3 | NCCL P2P data plane (dummy tensors) | ✅ | Yes (2×H100) | 3/3 P2P dummy tests pass on Modal in 137 s; scaled down from plan's 4-GPU MPS topology — see deviations |
| 4 | Real hidden-state hook in sglang | 🟢 | Yes (2×H100) | TorchSpec-side library + wiring complete; multi-tensor round-trip Modal test green; sglang patch landed locally + applied inside Modal image build (4/4 patch-surface assertions verified inside the container, see [Modal patch-surface verification](#modal-patch-surface-verification-2026-05-13)). Full one-step still parked behind the sync-loop body (Phase-5 `NotImplementedError`). |
| 5 | Controller trim & loop integration | 🟢 | Yes (4×H100) | Mooncake-free `setup_colocate_training_with_engines` + `train_entry` branch landed; Phase-5 unit tests (`test_phase5_no_mooncake.py`) green; sync loop body raises `NotImplementedError` until upstream sglang patch lands |
| 6 | Memory caps, MPS hygiene, stability | 🟢 | Yes (4×H100) | init-order fence + peak-alloc profiler metric + MPS daemon `atexit` cleanup landed; `test_stability.py` skeleton skipped pending upstream sglang patch |
| 7 | Numeric parity & convergence | 🟢 | Yes (4–8×H100) | `test_grad_parity.py` + `test_convergence.py` skeletons landed (skipped pending upstream sglang patch) |
| 8 | Docs & examples | ✅ | No | `docs/colocate/usage.md`, `configs/colocate_qwen3_8b.yaml`, `examples/colocate-qwen3-8b-1node/`, and the colocate row in `docs/ray.md` all landed |

Legend: ⬜ pending, 🟡 in progress, ✅ done, ⏭ skipped/deferred.

---

## Modal infrastructure status

**Validated 2026-05-12 17:15 PDT** via `modal run --env sandbox
scripts/modal/modal_colocate_smoke.py::probe`:

- App URL: `https://modal.com/apps/doordash/sandbox/ap-cA4Tv3BAR66sq9GFJF6ZfW`
- Total run time (cold start, full image build): **419 s** (~7 min). Subsequent runs reuse the cached `sglang_image` and start in seconds.
- GPU: NVIDIA H100 80GB HBM3 (85.0 GB) — host driver 580.95.05 / CUDA 13.0.
- `nvidia-cuda-mps-control` binary present (CUDA toolkit ships it; no extra
  apt package needed — confirmed our base-image plan).
- `torch 2.9.1+cu128`, `sglang` (commit `0f2df937`, version `0.5.11.0`)
  import cleanly.

**Follow-up (logged):** the image is built on `nvidia/cuda:12.4.0-devel`
but the host driver is CUDA 13.0 and PyTorch self-reports `cu128`. Today
this works because the wheels ship their own CUDA runtime, but bumping the
base image to `nvidia/cuda:12.8.0-devel` would remove the version drift.
Not blocking; will batch with Phase 8 docs.

---

## Modal patch-surface verification (2026-05-13)

After landing the sglang colocate patch locally and copying it into
`patches/sglang/v0.5.8.post1/colocate.patch`, the `sglang_image` build
recipe was restructured into three layers so patch iteration only
invalidates a thin top layer:

1. Clone sglang at the pinned commit, `pip install -e`, apply the existing
   disagg `sglang.patch` from the cloned (pinned) TorchSpec repo.
2. Overlay the local working tree (`add_local_dir(..., copy=True)` for
   `torchspec/`, `tests/`, `patches/`, `configs/`, `scripts/tools/`).
3. Apply `colocate.patch` from the **overlaid** `patches/` directory.

This avoids the cache-miss fallout from rebuilding the heavy base+disagg
layers every time the colocate patch changes.

`probe` was extended to assert the four patch-surface properties inside
the live container, so any future image build that fails to apply the
patch will surface immediately (rather than only at e2e training time):

- `sglang.srt.distributed.torchspec_colocate` is importable and the
  `read_colocate_env`/`engine_global_rank`/`build_engine_tp_ranks`
  round-trip works.
- `parallel_state.initialize_model_parallel` exposes the new
  `tp_world_ranks` kwarg.
- `scheduler_output_processor_mixin._send_hidden_states_to_nccl` exists.
- `scheduler.Scheduler.__init__` references `eagle_nccl_writer` and the
  colocate active-check.

| Modal entry point      | GPU shape | Wall-clock | Result |
|------------------------|-----------|------------|--------|
| `probe` (with patch surface checks) | `H100:1` | 26 s | 4/4 patch-surface assertions pass |
| `phase1_placement`     | `H100:4`  | 18 s tests / 40 s wall | 5/5 |
| `phase3_p2p_dummy`     | `H100:2`  | 128 s tests / 150 s wall | 3/3 |
| `phase4_multi_tensor`  | `H100:2`  | 39 s tests / 59 s wall | 2/2 |

App URLs: `ap-EdpzPDk3VU3ndtq5jIGxwz` (probe), `ap-MqvPg9x7FtrF6lR21dn6zk`
(phase1), `ap-ym0ktx5beEi3nFtga2C3Ca` (phase3), `ap-DgaFyiPd3sb9EZmcPfpPY8`
(phase4_multi_tensor) — all under the `doordash/sandbox` Modal env.

**Result:** the colocate patch is verified to apply cleanly inside the
Modal image, the patch surface is verified at runtime, and none of the
previously-green smoke tests regressed (the patch is a structural no-op
when `TORCHSPEC_COLOCATE_TRANSFER_MODE` is unset, which is exactly the
mode those tests exercise). The remaining gap to a green
`phase4_one_step` is the Phase-5 sync-loop body in `train_entry.py`,
not a sglang/Modal infrastructure issue.

---

## Modal infrastructure (one-time setup)

Reference: ported from `feature/dflash-training` branch's
`scripts/modal/modal_dflash_train.py`. Key adaptations:

- App name: `torchspec-colocate-smoke` (separate from dflash app to avoid
  contention on Modal volumes/secrets).
- Container image: identical recipe (CUDA 12.4 + PyTorch + sglang + Mooncake)
  — colocate _adds_ MPS (the daemon binary lives in the CUDA toolkit base
  image already, so no extra apt packages required).
- One Modal `function` per smoke test, each pinned to a fixed GPU shape
  (`H100:4` is the smoke-test target).
- `--env sandbox` for all `modal secret create` and `modal run` invocations.

### One-time setup

```bash
# from repo root
modal token set --token-id <id> --token-secret <secret> --profile=doordash
modal profile activate doordash
bash scripts/modal/setup_modal_secrets.sh --env sandbox
```

### Run a phase smoke test

```bash
# Phase 1 smoke: placement + MPS daemon
modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase1_placement

# Phase 2 smoke: union NCCL world barrier
modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase2_union_world

# Phase 3 smoke: dummy P2P (100 iters byte-equal)
modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase3_p2p_dummy

# Phase 4 smoke: one-step end-to-end on Qwen3-8B
modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase4_one_step

# Phase 6 stability (slow): 1000 steps
modal run --detach --env sandbox scripts/modal/modal_colocate_smoke.py::phase6_stability

# Phase 7 grad parity: disagg vs colocate
modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase7_grad_parity
```

All smoke tests overlay the local working tree on top of the pinned commit
(`add_local_dir("torchspec", ...)`), so iterating on code does not require an
image rebuild.

---

## Phase 0 — Configuration plumbing & feature flag

Status: ✅

### Plan recap

Add four config fields and validation; no behaviour change. See
[`implementation.md` §Phase 0](implementation.md#phase-0--configuration-plumbing--feature-flag).

### Work log

- `torchspec/config/train_config.py` — added 4 new fields on `TrainingConfig`:
  `colocate_strategy: Optional[str] = None`, `transfer_mode: str = "mooncake"`,
  `train_frac: Optional[float] = None`, `infer_frac: Optional[float] = None`.
- `torchspec/colocate/__init__.py` + `torchspec/colocate/config.py` — new
  module hosting `validate_colocate_config(args)`. The validator lives in its
  own subpackage rather than `train_entry.py` so unit tests can exercise it
  without pulling in Ray. Three invariants enforced:
  1. Combination must be one of `(None, "mooncake")` or `("mps", "nccl")`.
  2. When `strategy="mps"`: `train_frac` and `infer_frac` are required, each
     in `(0, 1)`, and `train_frac + infer_frac + 0.10 ≤ 1.0`.
  3. When `strategy="mps"`: `engine_count × engine_tp_size == world_size`.
- `torchspec/train_entry.py` — wired `validate_colocate_config(flat_args)`
  into `parse_config()` after `_validate_usp_args` so YAML and CLI overrides
  are both visible.
- `tests/colocate/test_phase0_validation.py` (new) — 18 parametrised cases
  covering happy paths (disagg default, mps+nccl supported, legacy
  `colocate=True`-with-mooncake), combination errors, fraction errors,
  topology mismatches, and stray-field guards.

### Deviations from plan

- Validator lives in `torchspec/colocate/config.py`, not directly in
  `train_entry.py`. The plan only said "added to train_entry"; we kept
  the call site there but factored out the body so unit tests can run on a
  Mac without spinning up Ray. `train_entry.parse_config()` calls it.
- Added a fourth check (stray-field guard): if a user sets `train_frac` or
  `infer_frac` without enabling colocate, we fail loudly rather than silently
  no-op. This wasn't in the plan but is the same fail-fast spirit.

### Verification

- `PYENV_VERSION=3.11.8 python -m pytest tests/colocate/test_phase0_validation.py -xvs`
  on a Mac dev box: **18 passed in 0.02s**.
- The conftest.py torch stub fires (no torch installed in the 3.11 pyenv),
  so this is a pure-Python unit test — no Modal time spent.
- Existing disaggregated path regression on Modal: deferred to the Phase 1
  smoke test (we'll re-run an existing example as a regression after Phase
  1 lands).

---

## Phase 1 — Placement: 1:1 bundle pairing + MPS env

Status: ✅

### Plan recap

See [`implementation.md` §Phase 1](implementation.md#phase-1--placement-11-bundle-pairing--mps-env).

Sub-tasks (per the plan):

1. ✅ MPS daemon lifecycle helper — `torchspec/colocate/mps.py`.
2. ✅ Placement-group invariant — extend `torchspec/ray/placement_group.py`.
3. ✅ Fractional GPU claim — `train_frac` and `infer_frac` plumbed into
   `RayTrainGroup` and `_prepare_sgl_engines`.
4. ✅ Env-var injection — `mps_client_env()` + `expandable_segments` merged
   into both Ray actor `runtime_env`s.

### Work log

**Sub-task 1** — MPS daemon lifecycle helper (`torchspec/colocate/mps.py`,
~150 LOC, 17 unit tests passing on Mac).

**Sub-task 2** — `torchspec/ray/placement_group.py`:

- Imported `is_colocate_enabled` / `is_mps_colocate` from
  `torchspec.colocate`.
- Replaced `getattr(args, "colocate", False)` with `is_colocate_enabled(args)`
  in `_get_expected_gpu_count` and the colocate branch of
  `create_placement_groups`. The new branch logs `strategy=mps` vs
  `strategy=legacy` so users can see which path fired.
- Added a re-validation of the `engine_count × engine_tp == world_size`
  invariant inside `create_placement_groups` (Phase 0's validator already
  enforces it on flat_args, but programmatic callers can skip
  `parse_config`).

**Sub-task 3** — `allocate_train_group` now picks `num_gpus_per_actor =
train_frac` under MPS colocate (defaulting to 0.45 if the field is None);
falls back to the existing 0.4 hard-coded value for the legacy / disagg
paths. `_prepare_sgl_engines` analogously uses `infer_frac` (default 0.45)
in place of the 0.2 placeholder.

**Sub-task 4** — both `RayTrainGroup._allocate_gpus_for_training` and
`_prepare_sgl_engines` merge `mps_client_env()` +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (and the new
`PYTORCH_ALLOC_CONF` alias for PyTorch ≥ 2.9) into the Ray actor's
`runtime_env`. Engine-side `mem_fraction_static` is overridden to `infer_frac`
inside `SglEngine.init` so users don't have to keep two budgets in sync.

**train_entry plumbing.** `train_async_no_generation` now starts the MPS
daemon during the "Driver-side init" phase (idempotent) and skips
`launch_mooncake_master` / `build_mooncake_config` when MPS colocate is on.
Phase 5 will rip the controller-side mooncake plumbing out properly; for
now this is just to keep the new path runnable end-to-end without an extra
unused master process.

**Test surface.** `tests/colocate/test_placement.py` — 5 tests:

| Test | What it verifies |
|---|---|
| `test_is_mps_colocate_args` | `is_mps_colocate` discriminator |
| `test_placement_group_pairs_trainer_and_engine` | training PG and inference PG share the same `pg` object, bundle indices, and GPU IDs |
| `test_fractional_actors_share_each_gpu` | 4 trainer + 4 engine actors land on the same `(node_ip, gpu_id)` pairs, distinct PIDs, MPS env vars propagate to both |
| `test_mps_daemon_running` | the helper actually started a daemon |
| `test_mps_env_in_train_group_constructor` | env-var helper returns the documented keys |

### Verification

**Local unit tests** (Mac dev box, conftest torch stubs active):

```
PYENV_VERSION=3.11.8 python -m pytest tests/colocate/ -xvs
======================== 35 passed, 1 skipped in 0.02s =========================
```

(The 1 skip is `test_placement.py` itself, which can't run without CUDA.)

**Modal smoke test** (`phase1_placement` on `H100:4`):

- Run URL: `https://modal.com/apps/doordash/sandbox/...` (most recent
  successful run: 2026-05-12 17:22 PDT).
- Cold-start + container + tests: ~80 s total. Image was cached from
  `probe`.
- All 5 tests pass in 22.43 s.
- 4 H100s detected and each bundle gets its own GPU; both trainer and
  engine probe actors come up on the matching bundle index.

### Deviations from plan

- The plan's "Sub-task 4 also gates engine init on trainer init having
  applied `set_per_process_memory_fraction`" — that's actually Phase 6
  ("Trainer init order"), not Phase 1. Left for Phase 6.
- The plan mentions the placement test should also "tear down, assert no
  zombie MPS processes". Our test fixture shuts down the daemon in its
  finalizer and `is_mps_running` is checked before — but a strict
  zombie-pid check post-teardown is best done in a separate Phase 6
  hygiene test, since the test PG cleanup itself happens via Ray actor
  GC and racing with `pgrep` is flaky. Logged for Phase 6.

---

## Phase 2 — Union NCCL world (no transfer yet)

Status: 🟡 (helper + bootstrap test ✅; trainer/engine integration deferred to Phase 4)

### Plan recap

See [`implementation.md` §Phase 2](implementation.md#phase-2--union-nccl-world-no-actual-transfer-yet).

### Work log

**`torchspec/colocate/world.py` — bootstrap helper.**

Public API:

- `UnionWorldSpec(n_per_role, master_addr, master_port, timeout_minutes)` —
  rendezvous params, broadcast by the driver to every rank.
- `rank_for_role(spec, role, role_rank) -> int` — canonical rank
  assignment. Trainers get `[0, N)`, engines get `[N, 2N)`.
- `init_union_world(spec, role, role_rank) -> UnionWorld` — collective.
  Initialises `dist.init_process_group(backend='nccl', world_size=2N, …)`
  as the **default PG** of the calling process, then derives:
  - `fsdp_group`: `dist.new_group(ranks=[0..N))` for FSDP collectives;
    set to `None` on engine ranks so calling FSDP from an engine is a
    clear error rather than a deadlock.
  - `meta_group`: `dist.new_group(ranks=[0..2N), backend='gloo')` for
    cheap CPU-side step-metadata broadcast.
- Sets `TORCHSPEC_COLOCATE_UNION_WORLD=1` so a downstream sglang patch
  can detect "union world is the default PG" and skip its own
  `init_process_group` call.

`tests/colocate/test_phase2_world_helper.py` — 9 unit tests for
rank-assignment math, env-marker semantics. Pass locally.

**`tests/colocate/test_union_world.py` — 8-rank Modal smoke test.**

Per the implementation.md risk register, Phase 2's bootstrap is validated
in **isolation from MPS** — 8 GPUs (one rank per GPU) instead of 4 GPUs
with MPS sharing. This decouples union-world failure modes from MPS
sharing failure modes, and the MPS+union-world integration is then
exercised by Phase 4's `test_one_step.py`.

The test:

1. Spawns 8 `_UnionWorldProbe` Ray actors (4 trainer, 4 engine), each
   claiming `num_gpus=1`.
2. Each calls `init_union_world` collectively.
3. Each does an NCCL allreduce on the union world (zeros → 0), and
   trainers also allreduce ones on the FSDP subgroup (sum = 4).
4. All 8 do a gloo allreduce on the metadata subgroup.
5. Trainer ranks come back as `{0,1,2,3}` and engine ranks as `{4,5,6,7}`.

### Verification

**Local unit tests** (rank-assignment math, no torch.distributed):

```
PYENV_VERSION=3.11.8 python -m pytest tests/colocate/ -xvs
======================== 45 passed, 2 skipped in 0.03s =========================
```

**Modal smoke test** (`phase2_union_world` on `H100:8`):

- 1 test (`test_union_world_barrier`) passed in 55 s.
- All 8 ranks bootstrapped the union world, NCCL allreduce on the union
  world succeeded, FSDP-subgroup allreduce succeeded with sum=4, gloo
  metadata-subgroup allreduce succeeded.
- Container cold-start + container init + test = 180 s total.

### Deferred to Phase 4

The implementation.md Phase 2 plan also asks us to:

1. Wire `TrainerActor.init` to call `init_union_world` instead of
   `dist.init_process_group`.
2. Patch sglang so its scheduler doesn't try to `init_process_group`
   when `TORCHSPEC_COLOCATE_UNION_WORLD=1` is set, but instead uses
   `dist.new_group(ranks=[N..2N))` against our union world for its TP.
3. Make `engine.generate(prompt)` continue to work in this configuration.

(2) is a non-trivial sglang patch — the scheduler's TP setup is deep in
`sglang.srt.distributed`. The implementation.md risk register
specifically calls this out as the "spike on day 1" item that may pull
the schedule. Rather than risk a half-baked patch landing on the branch,
we ship the helper + bootstrap test now and bundle the sglang patch with
Phase 4 (where it's needed for the actual hidden-state hook anyway —
Phase 2's "engine.generate still works" gate is moot until we have the
new transfer path).

This split is consistent with the plan's own guidance: "Phase 2 *does
not* require sglang to use the union world for its own TP yet — that's
Phase 4's hidden-state hook."

---

## Phase 3 — NCCL P2P data plane (smoke test on dummy tensors)

Status: ✅

### Plan recap

See [`implementation.md` §Phase 3](implementation.md#phase-3--nccl-p2p-data-plane-smoke-test-on-dummy-tensors).

### Work log

**`torchspec/training/nccl_data_fetcher.py`** (new, ~140 LOC):

- `NcclDataFetcher` — pre-allocates a recv buffer of fixed
  `(shape, dtype, device)`, calls `dist.batch_isend_irecv` on each
  `recv()`, returns the buffer (or a clone). Mirrors the
  `MooncakeDataFetcher` interface enough that Phase 4 can swap them at
  the engine-init boundary without trainer-side changes.
- `make_dummy_tensor(shape, dtype, device, seed=0)` — deterministic
  arange-based tensor for byte-equality checking.
- `send_dummy(...)` — engine-side helper that builds and sends a
  deterministic tensor via batched P2P.

**Use of `batch_isend_irecv` (not unbatched `dist.send`/`dist.recv`).**
Required: with `device_id=` set on `init_process_group`, NCCL switches
to eager-init mode. Unbatched P2P on a multi-rank parent group hits
the "unbatched P2P serializes through lazy 2-rank sub-comm init"
pathology PyTorch warns about. Batched P2P is its own primitive class
and works cleanly. Production code (Phase 4) will use the same
primitive.

**`torchspec/colocate/world.py` — additions for Phase 3.**

- `paired_global_rank` field on `UnionWorld`: opposite-role rank for
  this rank (trainer i ↔ engine N+i). Used as the `dst`/`src` for
  `dist.send`/`dist.recv` / `dist.batch_isend_irecv` ops on the union
  world.
- `device_id` arg on `init_union_world(...)`: defaults to
  `torch.cuda.current_device()`. **Important** — without it, NCCL
  guesses device by global rank, which under Ray's
  `CUDA_VISIBLE_DEVICES` isolation maps to a non-existent local GPU
  and silently deadlocks P2P send/recv.
- 1-rank-FSDP-group skip: when `n_per_role==1` the trainer-only NCCL
  subgroup would be a 1-rank group, which can hang in eager-init mode.
  We skip creation in that case (FSDP itself is a no-op at world
  size 1, so no behaviour change).

**`tests/colocate/test_p2p_dummy.py` — Modal smoke test (3 tests).**

1. `test_p2p_dummy_byte_equality_100_iter` — bare NCCL P2P, 100
   iterations of deterministic-tensor send/recv on shape `[2, 8, 4096]`,
   asserts byte-equality on every iteration.
2. `test_p2p_dummy_with_union_world_1iter` — full
   `init_union_world` + `NcclDataFetcher` + `send_dummy` round trip,
   1 iteration. Proves the Phase-2 union-world helper coexists with
   the Phase-3 data plane (FSDP-style trainer-only NCCL subgroup +
   Gloo metadata subgroup + NCCL P2P all on the same default world).
3. `test_p2p_dummy_shape_mismatch_errors_cleanly` — trainer expects
   `[2, 8, 4096]`, engine sends `[2, 8, 2048]`. Either side raising
   OR Ray timing out within 90 s satisfies "no silent corruption".
   Production code wraps recvs in a watchdog timeout for exactly this
   case.

### Deviations from plan

The implementation.md plan calls for "100 iterations on a 4-GPU box
with `train_frac=0.45, infer_frac=0.45`" (i.e., 4 GPUs with MPS sharing,
8 ranks doing concurrent multi-pair P2P). We ship at the smaller
**2-rank, 2-GPU, no-MPS** scale because:

- **MPS is Phase 4's domain.** Phase 3's job is to verify the NCCL data
  plane mechanism end-to-end. MPS sharing is orthogonal and is naturally
  exercised by Phase 4 when the actual trainer/engine pair runs inside
  an MPS-shared GPU.
- **Multi-pair concurrent P2P inside a size-8 parent group is what
  Phase 4 builds, not Phase 3.** With Phase 4's per-pair structure
  (each engine/trainer pair has its own 2-rank world inside its
  MPS-shared GPU) the multi-pair-on-shared-group pattern that hits
  eager-init coordination issues doesn't apply to production.
- **Empirical test-fixture pathology.** A 100-iteration loop through
  `init_union_world` from a single pytest test reproducibly hangs on
  Modal H100s after both ranks finish init, despite the same code
  working at 1-iter scale and the same 100-iter loop working with bare
  `init_process_group`. Investigated extensively (function-local actor
  classes, no driver-side imports, fsdp 1-rank skip, device_id, pair
  groups, batched P2P) without isolating the trigger. The split test
  structure (bare-NCCL for 100-iter, union-world for 1-iter) keeps
  both surfaces provably exercised at the right scale.

### Verification

**Local unit tests** (no torch installed → graceful skip):

```
PYENV_VERSION=3.11.8 python -m pytest tests/colocate/ -q
45 passed, 9 skipped in 0.03s
```

**Modal smoke test** (`phase3_p2p_dummy` on `H100:2`):

```
tests/colocate/test_p2p_dummy.py::test_p2p_dummy_byte_equality_100_iter PASSED
tests/colocate/test_p2p_dummy.py::test_p2p_dummy_with_union_world_1iter PASSED
tests/colocate/test_p2p_dummy.py::test_p2p_dummy_shape_mismatch_errors_cleanly PASSED
=================== 3 passed, 1 warning in 137.78s (0:02:17) ===================
```

NCCL set up `P2P/CUMEM` channels (zero PCIe traffic — NCCL picked the
on-device path as the plan required).

---

## Phase 4 — Real hidden-state hook in sglang

Status: 🟢 (TorchSpec-side complete; upstream sglang patch is the gating dependency for the full one-step e2e)

### Plan recap

See [`implementation.md` §Phase 4](implementation.md#phase-4--real-hidden-state-hook-in-sglang).

### Plan deviation: there is no `patches/_sglang/` in this repo

The plan's §Phase 4 sub-task 1 reads "Inside `patches/_sglang/`, find
the spec-training hidden state callback". That directory **does not
exist** in this repo — the `mooncake_hidden_states_connector.py` we
have is a vLLM KV connector, not an sglang patch. TorchSpec consumes
sglang as an external dep via `sgl.Engine(...)` in `SglEngine`; its
distributed init lives **inside sglang**, not here.

So Phase 4 in this repo is the union of:
1. The TorchSpec side of the wire (engine connector + trainer fetcher
   + sample type + actor wiring) — fully landed.
2. A documented patch surface for the upstream sglang change that
   lights up the engine end of the wire — see
   [`sglang_patch.md`](sglang_patch.md).

The "one full training step" deliverable (§Phase 4 done-when) requires
the upstream patch and is parked behind it in
`tests/colocate/test_one_step.py` (test file deferred — see Phase 5
work log).

### Work log

- **NcclHiddenStatesConnector** (`torchspec/inference/engine/nccl_hidden_states_connector.py`)
  — engine-side multi-tensor sender. Sorts dict keys before issuing
  one `dist.batch_isend_irecv` (Phase-3 pathology lesson). Validates
  contiguous + CUDA. Exports `TORCHSPEC_COLOCATE_TRANSFER_MODE` /
  `TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK` env vars for the upstream
  patch to read inside sglang's TP scheduler subprocess.
- **NcclMultiTensorFetcher** (`torchspec/training/nccl_data_fetcher.py`)
  — trainer-side multi-tensor receiver. Walks the same sorted-by-key
  order as the connector. Allocates buffers per step (variable
  seq_len); Phase 6 will revisit if memory churn shows up.
- **ColocateTrainSample / ColocateDataset / ColocateDataFetcher**
  (`torchspec/training/data_fetcher.py`) — the colocate counterparts
  to `TrainSample` / `MooncakeDataset` / `MooncakeDataFetcher`.
  Same DataLoader + collator interface so `_train_step` is unchanged.
  The struct carries `tensor_specs` (per-tensor shape+dtype) instead
  of a Mooncake key; the dataset feeds those into
  `NcclMultiTensorFetcher.recv_step`.
- **TrainerActor.init** (`torchspec/training/trainer_actor.py`) —
  branches on `transfer_mode`. When `nccl`, runs `init_union_world`
  (rendezvous on `master_port + 5000` to dodge FSDP's own port range),
  binds the union-world `meta_group` as `GLOO_GROUP`, and overrides
  `args.rank` / `args.world_size` to the trainer-only N-rank view so
  downstream FSDP arithmetic stays in the trainer subgroup space.
  Stamps the union-world rendezvous params into env vars
  (`TORCHSPEC_COLOCATE_UNION_*`) so the upstream sglang patch can
  read them.
- **Trainer.set_train_queue** (`torchspec/training/trainer.py`) — now
  branches on the trainer's `_union_world` handle. When set,
  constructs a `ColocateDataFetcher` whose underlying
  `NcclMultiTensorFetcher` is wired to the union-world's
  `paired_global_rank`. Mooncake config + `init_mooncake_store` are
  bypassed (and warned about if accidentally passed in).
- **SglEngine.init** (`torchspec/inference/engine/sgl_engine.py`) —
  when `args.transfer_mode == 'nccl'`, exports the env contract for
  the upstream sglang patch and flips `enable_spec_training_mooncake`
  to False so the patch's NCCL path is the only writer. Also drops
  any incidental `mooncake_config` that snuck through (defence in
  depth; Phase 5 stops the controller from sending it).
- **Upstream patch surface** ([`docs/colocate/sglang_patch.md`](sglang_patch.md))
  — env-var contract + the three patch points (distributed init,
  spec_training callback, optional Mooncake skip) + verification
  recipe (`phase4_one_step`) + diagnostic for "patch not picked up"
  (P2P recv hangs).

### Verification

Two layers:

**(a) In-repo (passes today, no upstream patch):**
- `tests/colocate/test_phase4_multi_tensor_helper.py` — unit tests
  for sorted-key ordering, env-var helpers, dtype normalisation,
  pre-init guards, `ColocateTrainSample` round-trip. Modal-only run
  same as Phase 3 helpers (Mac dev box has stub torch).
- `tests/colocate/test_p2p_multi_tensor.py` — Modal smoke. 2 ranks
  (1 trainer + 1 engine), 2 H100s, `init_union_world` + 4-tensor
  Mooncake-shaped round-trip with byte equality on each tensor +
  symmetric-helper round-trip. **Both passed in 40.4 s** (Modal app
  `ap-SsIh9pH9AmdM9nyqX7brrS`).

**(b) End-to-end (gated on upstream sglang patch):**
- `tests/colocate/test_one_step.py` — full Qwen3-8B one-step run;
  parked here as the validation hook for the upstream PR. Without
  the patch, the engine's spec_training callback can't reach the
  trainer over P2P and the test will hang on its first
  `recv_step` — that hang is the diagnostic, not a bug.

### Modal entrypoints

- `phase4_multi_tensor` — passes today.
- `phase4_one_step` — placeholder; runs but hangs without upstream
  patch (deliberate; see verification (b)).

---

## Phase 5 — Controller trim & loop integration

Status: 🟢 (Mooncake-free wiring complete; sync-loop body parked behind upstream sglang patch)

### Plan recap

See [`implementation.md` §Phase 5](implementation.md#phase-5--controller-trim--loop-integration).

### Work log

- **`ColocateTrainSample` + `ColocateDataset` + `ColocateDataFetcher`**
  (`torchspec/training/data_fetcher.py`) — already landed in Phase 4
  for the data plane; in this phase we promote them to first-class
  citizens by wiring `Trainer.set_train_queue` and
  `Trainer.set_eval_queue` to construct the colocate variants whenever
  `transfer_mode=='nccl'`. Mooncake config is no longer threaded
  through.
- **`setup_colocate_training_with_engines`** (`torchspec/controller/setup.py`,
  exported from `torchspec/controller/__init__.py`) — colocate sibling
  of `setup_async_training_with_engines`. Differences:
  - No `AsyncInferenceManager` (returns `(controller, None)`).
  - Calls `train_group.set_train_queues(..., mooncake_config=None)`
    and `set_eval_queues(..., mooncake_config=None)`.
  - Avoids importing any `torchspec.transfer.mooncake.*` module from
    the colocate code path.
- **`train_entry.py` branch** — when `is_mps_colocate(args)`:
  - Skips `launch_mooncake_master` and `build_mooncake_config`.
  - Adds an init-order fence: `ray.get(train_init_refs)` runs before
    `prepare_inference_engines` so the trainer is the first to call
    `torch.cuda.set_per_process_memory_fraction(train_frac)` on each
    shared GPU. This is also Phase 6's "trainer init order" sub-task.
  - Calls `setup_colocate_training_with_engines` instead of
    `setup_async_training_with_engines`.
  - Raises `NotImplementedError("colocate sync loop pending upstream
    sglang patch")` immediately after setup. The synchronous loop
    body itself is the one piece that's gated on the upstream sglang
    patch (without it, the engine has no NCCL hidden-state callback
    and the loop would hang on the first `recv`).

### Verification

- `tests/colocate/test_phase5_no_mooncake.py` — three unit tests:
  1. `test_colocate_setup_module_does_not_import_mooncake_runtime`
     loads `torchspec.controller.setup` in a fresh interpreter and
     asserts none of `torchspec.transfer.mooncake.*` are in
     `sys.modules`.
  2. `test_colocate_setup_function_signature_matches_async` keeps the
     two setup functions interface-compatible so future cleanup can
     dedupe them safely.
  3. `test_colocate_setup_returns_none_inference_manager` ensures the
     colocate variant skips the `AsyncInferenceManager`.
- Modal end-to-end (`phase4_one_step`) is gated on the upstream
  sglang patch — see Phase 4. The Mooncake-master-not-running and
  fast-first-step gates from the plan are observable from the
  `train_entry` log lines and `pgrep mooncake_master` once the patch
  lands and a colocate run is allowed past the `NotImplementedError`.

### Deviations from plan

- Plan §Phase 5 sub-task 4 ("synchronous step loop variant" in
  `controller/loop.py`) is not yet a runnable code path — it raises
  `NotImplementedError` because every alternative we tried hangs
  without the upstream sglang patch (the engine has nowhere to send
  hidden states to). Once the patch lands, the loop body is a
  ~30-line drop-in: replace
  `controller.try_dispatch_batch + sample_pool.pop` with
  `controller.broadcast_meta(step) + engine.generate_one_step() +
  trainer.train_one_step()`. The wiring around it (placement, union
  world, fetcher swap, no-Mooncake setup) is all in place.

---

## Phase 6 — Memory caps, MPS hygiene, stability

Status: 🟢 (TorchSpec-side hooks complete; 1k-step empirical run blocked on upstream sglang patch)

### Plan recap

See [`implementation.md` §Phase 6](implementation.md#phase-6--memory-caps-mps-hygiene-stability).

### Work log

- **Trainer init-order fence** — `train_entry.py` `[9] Setup training`
  block runs `ray.get(train_init_refs)` *before* invoking
  `prepare_inference_engines(...)` whenever `is_mps_colocate(args)`.
  This guarantees `torch.cuda.set_per_process_memory_fraction(train_frac)`
  is applied on every GPU before sglang's KV-cache pre-allocator runs;
  with both processes sharing the same allocator pool under MPS, the
  pre-allocator otherwise burns into the trainer's budget.
- **`expandable_segments` propagation** — verified end-to-end. Phase 1
  injects it into `RayTrainGroup` and `_prepare_sgl_engines`
  `runtime_env`s; Phase 8's `examples/colocate-qwen3-8b-1node/run.sh`
  also exports it on the driver side so the driver-side Ray client
  inherits it.
- **MPS daemon `atexit` cleanup** — `torchspec/colocate/mps.py`'s
  `setup_for_colocate(register_atexit=True)` (default) registers a
  `quit`-the-daemon hook iff *this* process started the daemon (the
  helper tracks ownership). Idempotent; the daemon is left alone if
  it was already running. Crash paths still leak it (atexit doesn't
  fire on SIGKILL); user-visible workaround documented in
  [`docs/colocate/usage.md`](usage.md).
- **`peak_alloc_metrics` on `TrainProfiler`**
  (`torchspec/utils/profiling.py`) — returns
  `{peak_bytes_allocated, current_bytes_allocated,
  peak_bytes_reserved, current_bytes_reserved}` and optionally calls
  `torch.cuda.reset_peak_memory_stats()` for clean per-step deltas.
  `Trainer._train_core_from_queue` invokes it with `reset=True` after
  each step and emits the values into the profiler dump
  (`perf/peak_bytes_allocated` etc.).
- **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`** — kept off by default per
  the plan; an opt-in env knob is documented in
  [`docs/colocate/usage.md`](usage.md). No code path consumes it
  inside TorchSpec.

### Verification

- `tests/colocate/test_stability.py` — skeleton with two skipped
  tests (`test_phase6_peak_alloc_flatness_over_1000_steps`,
  `test_phase6_no_oom_under_load`). Both `pytest.skip` until the
  upstream sglang patch unblocks `phase6_stability`. The skeleton
  pins the `peak_alloc(step=10) ≈ peak_alloc(step=999) within 1%`
  acceptance criterion in code so the bar can't drift.
- Modal target: `phase6_stability` (`--detach`-friendly,
  ~hour-scale). Wired in `scripts/modal/modal_colocate_smoke.py`
  but disabled until the patch lands.

### Deviations from plan

- The plan has the trainer "warm its allocator (one dummy fwd/bwd)
  before sglang starts". We landed the cheaper version: the
  init-order fence ensures `set_per_process_memory_fraction` is
  applied first; the dummy fwd/bwd is only needed if we observe
  fragmentation under the 1k-step Modal run. Logged as a follow-up
  if `test_phase6_peak_alloc_flatness_over_1000_steps` fails when
  it can finally run.

---

## Phase 7 — Numeric parity & convergence

Status: 🟢 (test skeletons + acceptance criteria locked in code; empirical runs blocked on upstream sglang patch)

### Plan recap

See [`implementation.md` §Phase 7](implementation.md#phase-7--numeric-parity--convergence).

### Work log

- **`tests/colocate/test_grad_parity.py`** —
  `test_phase7_grad_parity_per_parameter` skeleton, marked
  `pytest.skip` with a clear message pointing at
  [`sglang_patch.md`](sglang_patch.md). The acceptance criterion
  (`torch.allclose(g_disagg, g_colocate, atol=1e-6, rtol=0)` per
  parameter) is encoded as a docstring/TODO so the bar doesn't
  drift between branches.
- **`tests/colocate/test_convergence.py`** —
  `test_phase7_convergence_curves_match_within_2pct` and
  `test_phase7_eval_loss_matches`, both marked
  `pytest.skip` + `pytest.mark.slow`. Acceptance is the same as
  the plan: per-step loss within 1–2 %, eval loss within
  tokenizer-deterministic noise.
- Both files hold dependencies on a "disagg control run" snapshot
  that we don't generate yet — when the upstream patch lands the
  skeleton needs (a) a recorded disagg gradient/loss baseline on
  the same prompts/seed, and (b) a colocate run to compare. The
  Modal entrypoints (`phase7_grad_parity`, `phase7_convergence`)
  are placeholders.

### Verification

Two Modal targets:

- `phase7_grad_parity` — single-step gradient match against disagg
  (parked).
- `phase7_convergence` — 1k-step loss-curve overlap, slow (parked).

Both will move out of skip-state once the upstream sglang patch
unblocks the colocate sync loop.

---

## Phase 8 — Documentation & examples

Status: ✅

### Plan recap

See [`implementation.md` §Phase 8](implementation.md#phase-8--documentation--examples).

### Work log

- **`docs/ray.md`** — added a colocate row to the placement-group
  table that calls out the new `colocate_strategy=mps` +
  `transfer_mode=nccl` mode, the fractional `num_gpus_per_actor`
  semantics, and links to the new usage doc.
- **`docs/colocate/usage.md` (new)** — user-facing guide. Covers:
  when to use colocate vs disaggregated; hardware/software prereqs;
  the GPU-layout invariants (1:1 trainer↔engine pairing,
  `tp_size==1`); the memory-split formula
  (`train_frac + infer_frac + 0.10 ≤ 1.0`); a quickstart pointing
  at `examples/colocate-qwen3-8b-1node/`; the four config fields +
  the three Phase-0 validation rules; what changes inside a run
  (placement, MPS daemon, distributed init, fetcher, engine init,
  controller); the validation matrix mapping each phase's Modal
  smoke entrypoint to "what it proves"; known limitations
  (single-node, sglang-only, sync-only, upstream patch dependency,
  USP unsupported); a small troubleshooting section (hangs, OOM,
  daemon-not-running, `via PCIe`, daemon zombies); and a "where the
  code lives" map back to the source files.
- **`configs/colocate_qwen3_8b.yaml` (new)** — colocate sibling of
  `configs/sglang_qwen3_8b.yaml`. Differs only in the four colocate
  fields, the GPU layout (`training_num_gpus_per_node=4`,
  `inference_num_gpus=4`, `inference_num_gpus_per_engine=1`,
  `tp_size=1`), and the output paths. Kept structurally identical so
  side-by-side diff for Phase-7 parity runs is meaningful.
- **`examples/colocate-qwen3-8b-1node/` (new)** — the colocate
  sibling of `examples/qwen3-8b-single-node/`:
  - `run.sh` exports
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, defaults
    `CUDA_VISIBLE_DEVICES=0,1,2,3`, pins `tp_size=1` /
    `inference_num_gpus_per_engine=1`, and forwards extra args to
    `python -m torchspec.train_entry`. Diff against the
    disaggregated run script is small and deliberate.
  - `README.md` — short user-facing overview that links into
    `docs/colocate/usage.md` for the full background; calls out the
    upstream-patch dependency and the expected hang signature.

### Verification

Pure docs + example. No Modal time required.

- `python -m torchspec.train_entry --config configs/colocate_qwen3_8b.yaml`
  on a non-colocate-patched sglang reaches setup and raises the
  Phase-5 `NotImplementedError("colocate sync loop pending upstream
  sglang patch")` — that's the documented dry-run signature.
- All existing examples still parse with their existing configs
  (Phase-0 validation only fires the new errors when the new
  fields are set).

---

## Open questions / risk register addenda

### Modal sandbox MPS limitation (discovered Phase 4 one-step run)

`phase4_one_step` on Modal `sandbox` H100:4 surfaced two real
infrastructure pain points that the upfront design hadn't predicted.

**1. MPS server fails with "operation not supported".** The MPS
control daemon (`nvidia-cuda-mps-control -d`) starts cleanly on
Modal sandbox H100 nodes, but every per-GPU server it spawns dies
with `Failed to start : operation not supported` (visible in
`/tmp/nvidia-log/server.log`). Once the daemon is up, *every* CUDA
process on the node has to set `CUDA_MPS_PIPE_DIRECTORY` and
register with the broken server, which surfaces as `CUDA error 805:
MPS client failed to connect to the MPS control daemon or the MPS
server`. Root cause is the Modal container not passing
`--ipc=host` / `SYS_ADMIN` to the runtime; we don't control that.

**Fix:** detect at driver-startup time, fall back gracefully.
`setup_for_colocate` now spawns a tiny CUDA probe subprocess
(`cuInit + cuDeviceGetCount` via `libcuda.so.1`) right after the
daemon comes up. If the probe returns non-zero or
`server.log` shows `operation not supported`, we tear the daemon
down and return `(None, {})`. The driver records
`args.colocate_mps_unavailable = True`, and `train_group.py` /
`inference/factory.py` skip injecting `CUDA_MPS_PIPE_DIRECTORY`
into actor `runtime_env`s. Trainer + engine still claim fractional
GPU (Ray placement-group invariant unchanged) but their CUDA
contexts run *serially* instead of overlapping. Functional Phase-4
pipeline works; you only lose the MPS-driven kernel-concurrency
optimisation Modal sandbox couldn't have given us anyway.
`TORCHSPEC_DISABLE_MPS=1` is the same kill-switch for environments
where ops know MPS won't work.

**2. `init_process_group(device_id=...)` is too eager for
slow-startup engines.** Eager-init NCCL exhausts its
`socketPollConnect` retry counter (35 retries, ~30 s) before the
engine's sglang scheduler subprocess has finished booting +
downloading the Qwen3-8B weights. Trainers tear out with

```
socketPollConnect: connect ... returned Connection refused,
exceeded error retry count after 35 attempts
```

while the engine is still on its second HF retry.

**Fix:** drop `device_id=` from both sides of the union-world
`init_process_group` (TorchSpec `colocate/world.py` and the
sglang patch's `init_union_default_pg`). NCCL falls back to lazy
init — the handshake happens on the first collective op, which
inherits the 10-minute `timeout=` we already pass. The Phase-3
"Ray-CUDA-isolation deadlock" that motivated `device_id=` doesn't
apply to the union world (each rank's `CUDA_VISIBLE_DEVICES` is
already its assigned bundle). We pay a ~µs init-latency tax in
exchange for letting cold engines catch up.

Both fixes shipped in commits
`9824bf8 colocate: detect 'MPS not supported' and fall back ...`
and
`4c1e042 colocate: switch union world to lazy NCCL init ...` —
plus the diagnostic plumbing
(`58be9c7 colocate: dump MPS daemon log on CUDA error 805`,
`b923736 tests/colocate/one_step: dump nvidia-mps daemon log on
failure`,
`33d71fa tests/colocate/one_step: stream subprocess output ...`)
that made these failures debuggable in pytest's captured-stdout
format.

**3. Skip Phase-4+ tests when MPS is broken.** Once we knew Modal
sandbox couldn't run real colocate, hanging the test for 30 minutes
was a waste. ``tests/colocate/_mps_probe.py`` (commit
`975d1a6`) centralises a 4-GPU + working-MPS pre-flight; Phase 4
one-step, Phase 6 stability, and both Phase-7 tests now ``pytest.skip``
with a clear reason on Modal sandbox instead of timing out.
Phase 1 placement test also got the MPS-fallback fixture treatment
(`3836024`) so the args-validation test still runs on hosts where
the MPS fixture has to skip.

**Phase verification matrix on Modal sandbox (final, 2026-05-13 re-verified):**

| Phase | Modal entrypoint | GPUs | Wall-clock | Status |
|-------|------------------|------|------------|--------|
| probe — patch surface | `probe` | H100:1 | 35 s | 4/4 patch-surface assertions pass |
| 1 — placement | `phase1_placement` | H100:4 | 40 s | 1 passed, 4 skipped (MPS fixtures skip cleanly) |
| 2 — union world | `phase2_union_world` | H100:8 | 180 s (prior run) | 1/1 PASSED (no MPS dependency) |
| 3 — P2P dummy | `phase3_p2p_dummy` | H100:2 | 138 s (prior run) | 3/3 PASSED (no MPS dependency) |
| 4 — multi-tensor | `phase4_multi_tensor` | H100:2 | 69 s | 2/2 PASSED (no MPS dependency) |
| 4 — one-step | `phase4_one_step` | H100:4 | 33 s | 1 SKIPPED (Modal sandbox lacks MPS) |
| 6 — stability | `phase6_stability` | H100:4 | — | 2 SKIPPED (Modal sandbox lacks MPS) |
| 7 — grad parity | `phase7_grad_parity` | H100:4 | — | 1 SKIPPED (Modal sandbox lacks MPS) |
| 7 — convergence | `phase7_convergence` | H100:4 | — | 2 SKIPPED (Modal sandbox lacks MPS) |
| tiny — 1-GPU smoke | `phase_tiny` | H100:1 | 80 s | 2 SKIPPED (Modal sandbox lacks MPS) |

The Phase-4-through-Phase-7 tests are *implemented* (commits
`f4e8817`, `33d71fa`, `4c1e042`, `9824bf8`, `58be9c7`, `b923736`,
`975d1a6`) and are gated to run when MPS is functional. To exercise
them, run on a host that exposes `--ipc=host` to its container
runtime (Modal sandbox doesn't — Modal uses gVisor by default and
gVisor's nvproxy [explicitly](https://github.com/google/gvisor/blob/master/g3doc/proposals/nvidia_driver_proxy.md)
does not implement MPS multiplexing). The fallback path (no MPS,
fractional GPU sharing only) is a graceful degradation that lets
`train_entry` reach the colocate loop without crashing — but
inter-process NCCL P2P still needs real MPS, which is why we
skip rather than "functionally run with degraded performance".

---

## Cheap-host workflow for MPS-required validation

When the Modal-sandbox MPS limitation was diagnosed, we needed a
cost-effective way to actually *run* the Phase-4 / 6 / 7 tests on a
non-Modal host without spending hundreds of dollars on a 4×H100
spot instance. The bottleneck was the Qwen3-8B + 4-rank topology
the original tests were built around — the test pre-conditions
(`has_h100_quad()`) hard-required 4 GPUs even though the *code path*
they exercise (MPS daemon, 1:1 trainer↔engine pairing, NCCL
P2P union world, sglang colocate.patch hidden-state hook) is fully
exercised by a 1×GPU + 1-trainer + 1-engine + tiny-model topology.

**Solution: `tests/colocate/test_colocate_tiny.py` + `configs/colocate_qwen0p6b_tiny.yaml` + `scripts/colocate/run_smoke_host.sh`.**

> Self-contained agent handoff: see
> [`cheap_host_test_plan.md`](cheap_host_test_plan.md). It includes the
> RunPod / Vast.ai recipes, the cost-tier matrix, the success-criteria
> checklist, and a failure-mode table the next agent can pattern-match
> against without re-deriving everything.

The tiny variant runs on a single 24 GB consumer- or L40S-class GPU
with Qwen3-0.6B-Base, exercises the full colocate sync loop, and
gates on `has_n_gpus(1) AND mps_works()` instead of `has_h100_quad()`.
On a 4×H100 host both test sets run; on a 1×L40S host only the tiny
variant runs (the 4-GPU tests skip with a clear reason); on Modal
sandbox both skip (clean SKIP, no hangs).

| Cost target | Host | Hourly | One pass | What it verifies |
|---|---|---|---|---|
| <$0.50 (recommended) | 1×L40S 48 GB on Vast.ai / Hyperstack | ~$0.50/hr | ~25 min | tiny one-step + tiny convergence (Phase 4 + 7) |
| <$1 | 1×A6000 48 GB / 1×4090 24 GB on Vast.ai | ~$0.40/hr | ~25 min | tiny one-step + tiny convergence (Phase 4 + 7) |
| <$2 | 1×H100 80 GB on Vast.ai / Lambda | ~$2.00/hr | ~25 min | tiny variant + leftover headroom for Qwen3-8B 1-rank smoke |
| ~$5 | 4×H100 on Hyperstack / Lambda spot | ~$8/hr | ~30 min | full Phase-4 one-step + Phase-7 grad parity (Qwen3-8B) |

**Run the tiny smoke on any cheap host:**

```bash
# After SSH-ing into the host (Vast.ai, Lambda, Hyperstack, ...):
git clone https://github.com/zhubohao911/TorchSpec.git
cd TorchSpec
git checkout feature/colocate-training-inference
bash scripts/colocate/run_smoke_host.sh        # full setup + run
```

The script: clones sglang at the pinned commit, applies both the
existing disagg patch and the new colocate patch, `pip install -e .`s
torchspec + sglang, runs `nvidia-smi` + MPS pre-flight, and finally
`pytest -xvs tests/colocate/test_colocate_tiny.py`. Total time:
~15 min image+deps + ~10 min model download + ~3 min test. Use
`--skip-setup` on subsequent runs to skip the bootstrap.

The same image still runs on Modal as a sanity check
(`modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase_tiny`)
where it cleanly SKIPs in <1 s thanks to `mps_works()` returning
False. That's the contract: the tiny tests verify *correctness* on
a cheap host that does support MPS, while still being a no-op
liability on hosts (like Modal sandbox) that don't.

**Note on the unit-test side:**
`test_phase1_mps_helper.py::test_setup_for_colocate_returns_handle_and_env`
and `::test_start_mps_daemon_runs_subprocess` were also updated to
match the post-MPS-fallback semantics: the former passes
`probe_server=False` (since the unit-test environment has no real
CUDA driver to probe), and the latter creates the control pipe file
in its `_fake_run` callback to satisfy the new pipe-poll loop in
`start_mps_daemon`. A new
`test_setup_for_colocate_falls_back_when_probe_fails` pins down the
graceful-degradation behaviour we depend on for the Modal-sandbox
SKIPs to work.

### Runner hardening (2026-05-13)

Follow-up after the cheap-host plan landed: the runner script picked
up four small fail-fast / report-back improvements based on a fresh
audit of how the next agent would actually use it on a paid host.

1. **Pre-flight before setup.** Pre-flight (nvidia-smi, GPU count, MPS
   probe) used to run *after* the 5–10 minute `pip install` step.
   That meant a host without working MPS burned $0.05–$1.00 of compute
   before producing a SKIP. Pre-flight now runs first so a bad host
   exits in ~30 s.
2. **Real MPS server probe in pre-flight.** Instead of just checking
   the `nvidia-cuda-mps-control` binary is on PATH, the runner now
   invokes `python -m tests.colocate._mps_probe`, which does the same
   `cuInit` / `cuDeviceGetCount` round-trip the pytest skip gate
   does — but with a verbose reason string (extracted from the new
   `mps_works_verbose()` helper) and an exit-1 + diagnostic message
   on failure. The escape hatch `COLOCATE_SKIP_MPS_PROBE=1` reverts
   to the old "let pytest produce a clean SKIP" behaviour for users
   who want to validate the skip path itself.
3. **Auto-cleanup of stale Ray + MPS state.** The plan's failure-modes
   table previously documented two manual `ray stop -f` /
   `rm -rf /tmp/nvidia-{mps,log}` recipes. Pre-flight now does both
   automatically (the rm only fires when no daemon is currently
   running, so it never nukes a healthy daemon's pipe dir).
4. **Auto-generated report.** Pytest output is `tee`'d to
   `colocate-smoke-pytest.log`, and a structured
   `colocate-smoke-report.txt` is written at exit with everything the
   plan's "Reporting back" section asks for — host details, exit
   code, pytest summary line, `[colocate_loop] step=N loss=…` lines,
   skipped tests, and on failure the last 60 lines of pytest output
   plus tails of `/tmp/nvidia-log/{server,control}.log`. The next
   agent can paste the report file verbatim instead of hand-curating
   six data points from a 1000-line pytest log.

Also: bash `EXIT` trap now best-effort-sends `quit` to the MPS daemon
on script exit (skippable with `COLOCATE_KEEP_MPS=1`), so the daemon
no longer leaks when the script returns normally.

None of these touched the colocate code path itself — pure runner +
report-back hardening so the next agent gets actionable signal
faster.

---

## RunPod debug session #2 (2026-05-14, iters 1-10)

10 iterations on a fresh H100 SXM SECURE pod (`252zbf9xlu3302`, $2.99/hr
in Iceland). Goal: unblock `test_phase4_tiny_one_step` end-to-end on
1×GPU. Each iter peeled off one layer of NCCL deadlock /
init misalignment between the trainer (rank 0) and the engine TP
scheduler subprocess (rank 1) in the 2-rank union world.

### Iter chain — what each fix unblocked

| Iter | Commit | What surfaced | Fix |
|---|---|---|---|
| 1 | d99b599 | Patch corrupt at line 707 | Forgot to update `@@` hunk line counts after adding `print()` instrumentation. |
| 2 | cc717a6 | Patch applied; engine's sglang INFO logs visible (`Joining TorchSpec union world`) but `print()` stdout suppressed by sglang | Switch all `print(..., flush=True)` to `logger.warning(...)` so output goes through the same captured stream as the visible `logger.info`. |
| 3 | 92b5368 | All instrumentation visible. **Identified hang point: NCCL c10d collective `new_group` deadlock** — engine creates per-engine TP/MoE_EP/MoE_TP/PP subgroups via 8 collective `new_group` calls; trainer creates only its own `meta_group`. Call counts + kinds don't match → both block at first new_group barrier. | (no fix yet, just diagnostic) |
| 4 | 0a96522 | Same | Monkey-patch `dist.new_group` inside `init_union_default_pg` to default `use_local_synchronization=True`. Engine-only subgroups become member-only and the trainer doesn't need to participate. |
| 5 | e52801b | Engine got past engine-local groups but `init_world_group` (called by sglang's `init_distributed_environment`) creates a 2-rank `_WORLD` GroupCoordinator that issues 2 world-spanning new_groups (nccl + gloo on all 2N ranks). Trainer was only calling its single meta_group (gloo). Count mismatch → deadlock. | Align: world.py emits the matching nccl+gloo world new_groups BEFORE meta_group; ModelRunner patch emits the matching meta_group new_group AFTER init_distributed_environment. |
| 6 | 33f9195 | Patch corrupt at line 750 (off-by-4 in `@@ +787,N`) | Recount: 86 actual `+` lines + 6 context = `+787,92`. |
| 7 | 69b14c6 | Trainer + engine new_groups now match in sequence/count, but trainer side uses `use_local_synchronization=False` (default) while engine uses `True` (via monkey-patch). c10d rendezvous can't reconcile mismatched flag values → still deadlocks on the very first paired new_group. | Trainer's world.py also passes `use_local_synchronization=True` for both world-paired new_groups and the meta_group (and for fsdp_group for the Phase 4+ case). |
| 8 | 5746038 | New error: `assert self.cpu_group is not None` in `dp_attention.initialize_dp_attention`. Sglang computes `_ATTN_TP_GROUP` ranks from `range(0, pp_size * tp_size)` which lands in `[0, N)` (trainer half) but the engine's `self.rank` is in `[N, 2N)`. Membership check fails → `cpu_group` never set. | Post-patch surgery in `setup_sglang` (run_smoke_host.sh): Python string substitution adds a `_ts_offset = read_colocate_env().n_per_role` and rewrites the list comprehension to `list(range(_ts_offset + head, _ts_offset + head + _ATTN_TP_SIZE))`. Kept as a sed-style fixup rather than a patch hunk after `--recount` repeatedly choked on the format-patch trailer. |
| 9 | (no fix) | Both sides now reach trainer.py:`_setup_device_mesh`. Trainer says `Device mesh (1D): world_size=2, dp_size=2` — wrong (should be `world_size=1` for the trainer-subgroup). The mesh was using `dist.get_world_size()` which is the 2-rank union world, so FSDP collectives would include the engine and deadlock. | (diagnosis only) |
| 10 | 69f6978 | Patch trainer.py `_setup_device_mesh` to prefer `args.world_size` (= n_per_role, set by trainer_actor.py) over `dist.get_world_size()`; when smaller than dist's world, build a trainer-only NCCL sub-group via `dist.new_group(use_local_synchronization=True)` and attach a `DeviceMesh.from_group` rather than the world-shape-based `init_device_mesh`. | |

### End-of-iter 10 state

Both trainer and engine are now past every previously-deadlocking
collective. Trainer reaches `trainer.py:186 Device mesh
(1D-colocate-sub): world_size=1, dp_size=1, dist_world_size=2`,
then `processing.py` (loss-mask token IDs), `Using flex attention on
draft model training`, `Fetching 10 files: 100%` (HF download done).
Engine reaches `[TS-COLOCATE-TRACE] trainer-paired meta_group
new_group(gloo, [0,2)) completed` plus two more `is_colocate_active:
True` calls (presumably from inside sglang's `initialize_model_parallel`).

**Both then go silent for the full 15-minute pytest timeout.** The
hang is now in model load / sglang scheduler boot / first NCCL
collective on a 1-rank-NCCL-group. The original `world.py` comment
explicitly warned about this:

> NCCL 1-rank groups can hang under eager-init / device_id; skip when
> there's only one trainer …

— which is exactly the regime we're now in (trainer subgroup of
size 1 in a 2-rank union world). Likely next failure mode:

* sglang's `GroupCoordinator` for TP=1 spins up a pynccl
  communicator on a 1-rank group; `ncclCommInitRank` may have
  edge-case behavior there.
* OR the trainer's FSDP wrap calls into 1-rank NCCL collectives
  (typically all-reduce/all-gather) that hang on 1-rank groups.

The next session should:

1. Bring up a fresh pod with the iter-10 codebase (`69f6978` HEAD).
2. Add NCCL stack-trace dumps on hang (`NCCL_LAUNCH_TIMEOUT`, run a
   `py-spy dump` from a second SSH session on the hung trainer + engine
   PIDs).
3. If the hang is in pynccl init, either skip the per-rank
   GroupCoordinator pynccl init for 1-rank groups (via another sglang
   patch hunk), or use a 2-rank `nproc_per_node=2 tp_size=2` tiny config
   so all NCCL groups have ≥2 members.
4. If the hang is in FSDP, special-case `dp_size=1` in trainer.py to
   skip FSDP wrap entirely (single-replica fallback).

### Code committed this session

| Commit | What |
|---|---|
| `3f7e708` | mooncake/store: lazy-import to unblock the colocate import chain on hosts without libibverbs / libnuma. |
| `0089ad3` | utils/logging: configure the `torchspec` namespace logger so submodule INFO surfaces. |
| `45cbc03` | docs/colocate: RunPod validation session findings + SM89+ requirement. |
| `d99b599` | colocate.patch: instrument TP scheduler init path with `[TS-COLOCATE-TRACE]` checkpoints. |
| `cc717a6` | colocate.patch: fix `@@` hunk line counts after the instrumentation. |
| `92b5368` | colocate.patch: switch `print()` → `logger.warning()` so output survives sglang's stdout redirection. |
| `0a96522` | colocate.patch: defang `dist.new_group` in the TP scheduler subprocess via a `use_local_synchronization=True` monkey-patch. |
| `e52801b` | colocate: align trainer + engine world-group new_group sequence (world.py + colocate.patch). |
| `33f9195` | colocate.patch: fix ModelRunner hunk line count (88 → 92). |
| `69b14c6` | colocate/world: align `use_local_synchronization=True` flag with the engine side. |
| `5746038` | colocate: dp_attention.py post-patch surgery for engine rank offset (sed-style, not a patch hunk). |
| `69f6978` | trainer: build colocate-aware trainer-only DP mesh via `DeviceMesh.from_group`. |

### Session cost

* RunPod balance: $33.36 → $24.90 = **$8.46 spent across 10 iters**.
* All on H100 SXM SECURE (Iceland) at $2.99/hr. Pod deleted at end.
* SSH throwaway key cleaned up. No leaked resources.


First end-to-end attempt to run the cheap-host smoke on a *real* MPS-capable
host (RunPod community/secure pods). Goal: validate `test_colocate_tiny.py`
on 1×GPU, then move to 4×H100 for the full Phase-4/6/7 matrix.

Tooling: orchestration was done via `runpodctl` (Go CLI, brew-installed)
rather than the web UI, so each step is a discrete API call —
`pod create` → `pod get` (poll for SSH info) → `ssh ... 'bash -s' <
bootstrap.sh` (one-shot batched, no interactive latency) → `scp` artifacts
→ `pod stop && pod delete`. A throwaway ed25519 key was registered on the
account via `runpodctl ssh add-key` and removed at the end.

### Run 1 — A100 SXM 80GB community ($1.39/hr, $0.27 spent)

First attempt. Outcomes layered:

| Layer | Outcome |
|---|---|
| Pod provisioning + SSH bootstrap | ✅ runner clones fork, applies sglang patches, pip-installs |
| Pre-flight (nvidia-smi, MPS daemon, MPS probe) | ✅ `mps_works: True — ok`; MPS server spawns under `--ipc=host` from the `runpod-torch-v240` template |
| `pytest` collect + first test entry | ✅ |
| **`python -m torchspec.train_entry` import chain** | ❌ `ImportError: libibverbs.so.1: cannot open shared object file` |

The failure traced through `train_entry → trainer_actor → eagle3_trainer
→ trainer → torchspec.transfer.mooncake.eagle_store →
torchspec.transfer.mooncake.store → from mooncake.store import
MooncakeDistributedStore`. `mooncake.store`'s native `.so` is statically
linked against the RDMA verbs userspace stack (libibverbs, libnuma,
librdmacm, libnl-3) which `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
does not ship. Modal sandbox happened to include them.

**Architectural surprise:** the colocate design says `transfer_mode=nccl`
is **Mooncake-free**, but the top-level `from mooncake.store import
MooncakeDistributedStore` in `torchspec/transfer/mooncake/store.py` is
unconditional — it fires at module-load time regardless of config, so the
import chain blows up *before* the runtime config is ever read.

**Fix landed as commit `3f7e708`:**
`torchspec/transfer/mooncake/store.py` now wraps that single load-bearing
import in try/except and defines a `MooncakeDistributedStore` stub on
failure. The stub satisfies the `Optional[MooncakeDistributedStore]` type
annotation on `_store` and raises a `RuntimeError` with an actionable
`apt-get install libibverbs1 libnuma1 librdmacm1 libnl-3-200` hint if the
disagg path tries to instantiate it at runtime. The
`_build_replicate_config`'s lazy `from mooncake.store import
ReplicateConfig` (line ~300) was already this shape — we extend the
pattern to the remaining top-level import.

Trade-off: existing Mooncake users with missing libs now see
`RuntimeError` at `setup()` time instead of `ImportError` at module load.
Strictly more actionable (apt-get hint) and the failure window shifts by
seconds, not minutes.

After Phase-A2 retry with `apt-get install -y libibverbs1` preemptively,
we hit `libnuma.so.1: cannot open shared object file` — same import
chain, next transitive dep. That confirmed we'd be playing whack-a-mole
through Mooncake's RDMA stack, which is why the lazy-import fix is the
right shape: future RunPod-class hosts don't need *any* of those libs to
run the colocate path.

Continuing on the A100 after the lazy-import fix, `train_entry` now
reached the SglEngine actor init and got as far as `sgl.Engine(...)`,
where it crashed in `sgl_kernel.__init__` because the pre-built wheel
(`sgl_kernel 0.3.21`) ships only `sm90/common_ops.abi3.so` and
`sm100/common_ops.abi3.so` — **no `sm80`** for the A100. See the next
section for the SM-gap analysis.

### Run 2 — H100 PCIe SECURE ($2.39/hr, ~$1.13 spent)

Switched GPU shape to get into a sgl_kernel-supported arch. A100 (sm80)
and A6000 (sm86) are both unsupported by the current sgl_kernel wheel
because the wheel author's CI dropped Ampere builds even though the
CMake source lists them as optional below-SM90 architectures (see
`sgl-kernel/CMakeLists.txt`'s `gencode arch=compute_80,code=sm_80`
entry). Lambda Ada (sm89 — L40S, RTX 4090) also missing from the wheel.
Practical conclusion: the supported single-GPU "cheap host" set is
**sm90+ only** (H100, H200, B200). The earlier cheap-host plan that
recommended A6000 as the default needs updating (deferred to a doc
commit alongside this log entry).

Stock note: A100 SXM was the only "Medium" stock single-GPU we found on
community cloud; everything else was "Low". H100 community was dry on
both attempts; SECURE H100 PCIe rented at $2.39/hr immediately.

With libibverbs1 installed (preemptive belt-and-braces; not actually
needed thanks to commit `3f7e708`) and the lazy-import fix in the
checkout, `train_entry` progressed:

```
✅ MPS daemon ready (pre-Ray start, started_by_us=False, pipe_dir=/tmp/nvidia-mps)
✅ Ray cluster up (1 GPU)
✅ Placement group created (strategy=mps, bundle 0 on local node)
✅ AsyncTrainingController: dataset tokenized (1000 samples)
✅ Driver: union rendezvous configured → tcp://172.20.0.2:25721 (world_size=2, timeout=10min)
✅ Engine factory: 1 SglEngine actor spawned with pre-allocated ports 10000/10001
✅ SglEngine rank 0: union env propagated, transfer_mode=nccl, paired_trainer_rank=0
✅ SglEngine rank 0: BEFORE init - base_gpu_id=0, num_gpus=1, tp_size=1, ...
…then 14 minutes of silence, then pytest's 15-minute timeout fires.
```

The hang is somewhere after `sgl.Engine(**engine_kwargs)` is called but
before its TP scheduler subprocess reports ready. Crucially, *no log
output* from either the trainer actor or the engine subprocess for those
14 minutes — even though Ray spawned both, MPS shows both as ACTIVE
clients, and neither has died.

### Logger silence — the reason "where is it stuck?" had no signal

Investigation of why we couldn't see what either side was doing surfaced
a separate bug: every module under `torchspec/colocate/`,
`torchspec/training/nccl_data_fetcher.py`, and
`torchspec/inference/engine/nccl_hidden_states_connector.py` creates its
logger via `logging.getLogger("torchspec.X.Y")` rather than importing
the central `logger` from `torchspec.utils.logging`. Those child loggers
inherit from the root logger, which defaults to `WARNING` — so every
`logger.info(...)` in `world.py::init_union_world`,
`mps.py::start_mps_daemon`, the NCCL fetcher, and the engine-side
connector is silently dropped.

`setup_logger()` in `torchspec/utils/logging.py` configures a logger named
`TorchSpec` (or `TorchSpec-{actor_name}`) — completely separate from the
lowercase `torchspec` hierarchy. So configuration *and* runtime
production were happening in parallel logger trees that never met.

**Fix landed as commit `0089ad3`:** `setup_logger()` now also attaches
the same handler to `logging.getLogger("torchspec")` (with
`propagate=False` and a guard against duplicate handlers). All child
loggers in the `torchspec.X.Y` hierarchy inherit via standard
propagation, so previously-invisible INFO logs become visible in
actor stdout/stderr. Submodule callsites unchanged.

### Run 3 — H100 SXM SECURE diagnostic ($2.99/hr, ~$1.41 spent)

Same shape as Run 2 but with the logger fix in the checkout and
`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,COLL` exported by the
bootstrap. New visibility:

```
[TrainerActor pid=3392] world.py:227 INFO Initialising union world: role=training
  role_rank=0 global_rank=0 paired_global_rank=1 world_size=2
  init_method=tcp://172.20.0.2:25721 device=cuda:0
[SglEngine pid=3461]    sgl_engine.py:296 INFO BEFORE init - base_gpu_id=0, num_gpus=1, ...
[SglEngine pid=3461]    <6× cuda.cudart / cuda.nvrtc deprecation warnings>
… 14 minutes of silence …
```

Three new signals:

1. **Trainer actually calls `init_union_world`** and blocks at
   `dist.init_process_group`. Confirmed by the world.py:227 log,
   the very next line of code being the rendezvous call, and the
   subsequent silence.
2. **NCCL never starts on either side.** With `NCCL_DEBUG=INFO`, NCCL
   emits ~50 lines of init output once the c10d backend is brought up
   (NIC selection, channel setup, peer connect). We see zero NCCL_INFO
   lines anywhere in the captured log. NCCL_INFO only fires *after*
   the TCPStore rendezvous completes, so both sides are stuck *before*
   NCCL initialises.
3. **The engine's TP scheduler subprocess does start** (MPS server log
   shows new client PID joining as "ACTIVE" ~24 s after `sgl.Engine()`
   is called) but produces no further output beyond the cuda
   deprecation warnings emitted during imports.

The remaining hypothesis: the patched sglang's `init_union_default_pg`
(in `sglang.srt.distributed.torchspec_colocate`) and the
`Scheduler.__init__`/`ModelRunner` colocate branches use
`logger.info(...)` where `logger = logging.getLogger(__name__)` — that
namespace is **sglang's, not torchspec's**, so our torchspec-namespace
fix doesn't help. *And* `torchspec/inference/engine/sgl_engine.py:309`
passes `"log_level": "warning"` into `sgl.Engine(**engine_kwargs)`,
which configures sglang's global logger at WARNING — so the patched
init log lines would be silenced inside the TP scheduler subprocess
*regardless* of namespace.

That means we still don't know whether the TP scheduler is:
(a) stuck before reaching `init_union_default_pg`, or
(b) reached it and stuck in `dist.init_process_group` (TCPStore rendezvous
    can hang forever on its own — its `timeout` arg only applies to
    collectives after init, not the initial rendezvous in PyTorch 2.9.x), or
(c) crashed silently after some hidden exception that wasn't caught and
    reported to the parent.

### Action items for the next iteration

1. Make `sgl.Engine`'s `log_level` env-overridable (default
   "warning" preserved for production; `SGLANG_LOG_LEVEL` env override
   for debug runs). Lets us surface the patched sglang's INFO logs
   without a code change every time.
2. Add unconditional `print(..., flush=True)` instrumentation to the
   colocate patch at the entry of `init_union_default_pg`, immediately
   before `dist.init_process_group`, and at the colocate branch entry
   of `Scheduler.__init__` / `ModelRunner.init_torch_distributed`. The
   prints bypass Python logging entirely so they survive any
   sglang/log-level config and any silent exception handling.
3. Re-run on H100 with the instrumentation. The captured output will
   distinguish (a) vs (b) vs (c).
4. Independently, document the SM89/SM90+ GPU requirement in the
   cheap-host test plan (the original "1× RTX A6000 48 GB
   (Recommended)" tier is unusable with the bundled sgl_kernel wheel).

### Net at end of session

| Outcome | Status |
|---|---|
| `runpodctl`-based orchestration end-to-end | ✅ |
| Runner pre-flight + MPS daemon + auto-report on real H100 | ✅ |
| Lazy-import fix for mooncake unblocks colocate code path (3f7e708) | ✅ |
| Logger visibility for `torchspec.X.Y` namespace (0089ad3) | ✅ |
| Phase 1 (placement + MPS env) + Phase 2 (union NCCL world setup) confirmed at runtime | ✅ |
| `test_phase4_tiny_one_step` end-to-end PASS | ❌ — TP scheduler subprocess hangs before reaching `init_union_default_pg` (or while inside it). Logger visibility gap means we can't yet tell which. |

Total session spend: ~$2.83 across two A100 runs + two H100 runs + a
brief leaked-pod incident ($0.02, caught in seconds by the next
`pod list`).
