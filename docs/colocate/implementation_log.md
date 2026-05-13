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
| 3 | NCCL P2P data plane (dummy tensors) | ⬜ | Yes (4×H100) | |
| 4 | Real hidden-state hook in sglang | ⬜ | Yes (4×H100) | most of sglang patch |
| 5 | Controller trim & loop integration | ⬜ | Yes (4×H100) | |
| 6 | Memory caps, MPS hygiene, stability | ⬜ | Yes (4×H100) | slow 1000-step |
| 7 | Numeric parity & convergence | ⬜ | Yes (4–8×H100) | needs disagg control run |
| 8 | Docs & examples | ⬜ | No | |

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

Status: ⬜

### Plan recap

See [`implementation.md` §Phase 3](implementation.md#phase-3--nccl-p2p-data-plane-smoke-test-on-dummy-tensors).

### Work log

_(populated as work progresses)_

### Verification

Modal target: `phase3_p2p_dummy`.

- 100 iterations, byte-equality every iteration on shape `[2, 8, 4096]`.
- `nvidia-smi` reports zero PCIe / NVLink traffic during transfers (NCCL
  picked the on-device path).
- Shape-mismatch test errors cleanly without deadlock.

---

## Phase 4 — Real hidden-state hook in sglang

Status: ⬜

### Plan recap

See [`implementation.md` §Phase 4](implementation.md#phase-4--real-hidden-state-hook-in-sglang).

### Work log

_(populated as work progresses)_

### Verification

Modal target: `phase4_one_step` on Qwen3-8B with TP=4 engine + 4 FSDP
trainers.

- Loss is finite and non-zero.
- No Mooncake calls happen (mocked store fails the test if touched).

---

## Phase 5 — Controller trim & loop integration

Status: ⬜

### Plan recap

See [`implementation.md` §Phase 5](implementation.md#phase-5--controller-trim--loop-integration).

### Work log

_(populated as work progresses)_

### Verification

Modal target: extends `phase4_one_step`.

- `pgrep mooncake_master` returns nothing post-run.
- First training step starts within ~seconds of init (no async ramp-up).

---

## Phase 6 — Memory caps, MPS hygiene, stability

Status: ⬜

### Plan recap

See [`implementation.md` §Phase 6](implementation.md#phase-6--memory-caps-mps-hygiene-stability).

### Work log

_(populated as work progresses)_

### Verification

Modal target: `phase6_stability` (slow, `--detach` recommended).

- `peak_alloc(step=10)` ≈ `peak_alloc(step=999)` within 1 %.
- No process-side OOM, no system-side hang.

---

## Phase 7 — Numeric parity & convergence

Status: ⬜

### Plan recap

See [`implementation.md` §Phase 7](implementation.md#phase-7--numeric-parity--convergence).

### Work log

_(populated as work progresses)_

### Verification

Two Modal targets:

- `phase7_grad_parity` — single-step gradient match against disagg.
- `phase7_convergence` — 1k-step loss-curve overlap (slow).

---

## Phase 8 — Documentation & examples

Status: ⬜

### Plan recap

See [`implementation.md` §Phase 8](implementation.md#phase-8--documentation--examples).

### Work log

_(populated as work progresses)_

---

## Open questions / risk register addenda

_(none yet — populate when blockers surface during execution)_
