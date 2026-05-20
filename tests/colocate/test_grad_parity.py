# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 7 — gradient parity.

Plan reference: ``implementation.md`` §Phase 7 / the design doc's
validation plan — "Compare per-layer gradients against the Mooncake
baseline on identical prompts + seeds; require numerical match up to
NCCL non-determinism."

Three tests, increasing in cost and strength:

* ``test_phase7_grad_parity_smoke`` — one colocate step, assert a
  finite non-zero loss. Cheap regression guard for ``train_entry``.
* ``test_phase7_grad_parity_determinism`` — run the colocate tiny
  config twice with the same seed and assert the dumped per-parameter
  gradients are **bit-identical**. Proves the colocate path (gloo
  CPU-staged transfer included) injects no non-determinism. Needs only
  the colocate deps (1 GPU + MPS).
* ``test_phase7_grad_parity_full`` — the design doc's actual ask: run
  the disaggregated (Mooncake) tiny config and the colocate tiny config
  with the same seed and assert per-parameter draft-model gradients
  match. Both arms are dp_size=1 so FSDP is a no-op and the only thing
  that can differ is the hidden-state transport. Needs >=2 GPUs, MPS,
  and an importable Mooncake (RDMA userspace libs); skips cleanly
  otherwise.

The gradient snapshot is the existing ``debug.save_debug_train_data``
dump (``torchspec/utils/train_dump.py``); the deterministic-seed
plumbing is ``torchspec/colocate/determinism.py``, engaged on both arms
via ``TORCHSPEC_GRAD_PARITY=1``.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.colocate._mps_probe import has_h100_quad, has_n_gpus, mps_works

REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-parameter gradient match tolerance. Both arms compute the same
# thing deterministically, so the expectation is near-bit-identical; the
# small rtol absorbs a possible 1-ULP bf16 difference between the
# Mooncake and gloo transport copies. Override via env for tuning on a
# real host without a code change.
GRAD_ATOL = float(os.environ.get("GRAD_PARITY_ATOL", "1e-6"))
GRAD_RTOL = float(os.environ.get("GRAD_PARITY_RTOL", "2e-3"))


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _disagg_runnable() -> bool:
    """True iff the Mooncake store can actually be imported.

    The disagg baseline arm needs ``mooncake.store``, whose native .so
    links the RDMA verbs stack (libibverbs / libnuma / librdmacm /
    libnl-3). On hosts without those the import raises at load time;
    probe in a subprocess so a hard failure doesn't poison this process.
    """
    probe = "import mooncake.store  # noqa\nprint('ok')\n"
    try:
        proc = subprocess.run(
            ["python3", "-c", probe],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "ok" in proc.stdout


# ---------------------------------------------------------------------------
# Arm runner
# ---------------------------------------------------------------------------

def _run_arm(
    config_name: str,
    *,
    dump_dir: Path,
    visible_devices: str,
    seed: int = 42,
    extra_args: list[str] | None = None,
    timeout_s: int = 1800,
    disable_mps: bool = False,
) -> str:
    """Run train_entry for 1 step, dumping per-parameter gradients.

    Returns the captured combined stdout+stderr log.

    ``disable_mps`` is for the disaggregated arm: it is a non-colocate
    run and must not be caught by an MPS daemon left running by the
    colocate arm / earlier tests (its actors otherwise fail MPS's
    CUDA_VISIBLE_DEVICES validation and the worker dies).
    """
    config_path = REPO_ROOT / "configs" / config_name
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"
    dump_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    # Engage the strict deterministic-kernel path in seed_everything on
    # both arms (see torchspec/colocate/determinism.py).
    env["TORCHSPEC_GRAD_PARITY"] = "1"
    if disable_mps:
        env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        env.pop("CUDA_MPS_LOG_DIRECTORY", None)
        env["TORCHSPEC_DISABLE_MPS"] = "1"

    cmd = [
        "python", "-m", "torchspec.train_entry",
        "--config", str(config_path),
        f"dataset.train_data_path={dataset}",
        "training.num_train_steps=1",
        "training.num_epochs=1",
        f"training.seed={seed}",
        # Deterministic prompt order so both arms see the same batch.
        "dataset.shuffle_dataset=false",
        # dump_eagle3_batch formats {step}/{rank}/{batch_idx} into this.
        f"debug.save_debug_train_data={dump_dir}/g_{{step}}_{{rank}}_{{batch_idx}}.pt",
        *(extra_args or []),
    ]

    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=timeout_s,
    )
    log = proc.stdout + proc.stderr
    print(f"\n=== _run_arm({config_name}) tail ===")
    for line in log.splitlines()[-80:]:
        print(line)
    print(f"=== /_run_arm({config_name}) tail ===\n")
    assert proc.returncode == 0, (
        f"train_entry({config_name}) exited {proc.returncode}; see log above."
    )
    return log


def _extract_loss(log: str) -> float:
    """Parse the first ``loss=<float>`` from the colocate-loop output."""
    pat = re.compile(r"loss=(?P<v>[0-9eE.+\-]+)")
    for line in log.splitlines():
        if "[colocate_loop] step=" in line and "loss=" in line:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group("v"))
                except ValueError:
                    continue
    return float("nan")


# ---------------------------------------------------------------------------
# Gradient-dump comparison
# ---------------------------------------------------------------------------

def _load_grads(dump_dir: Path) -> dict[str, dict]:
    """Load every gradient dump in a dir, keyed by file name."""
    import torch

    out: dict[str, dict] = {}
    files = sorted(dump_dir.glob("g_*.pt"))
    for f in files:
        data = torch.load(f, weights_only=False, map_location="cpu")
        grads = data.get("gradients")
        if grads:
            out[f.name] = grads
    return out


def _compare_grad_dumps(
    dir_a: Path, dir_b: Path, *, atol: float, rtol: float
) -> tuple[int, list[str]]:
    """Compare per-parameter gradients between two dump dirs.

    Returns ``(n_params_compared, mismatches)`` where each mismatch is a
    human-readable line. Raises AssertionError-grade conditions are left
    to the caller.
    """
    import torch

    grads_a = _load_grads(dir_a)
    grads_b = _load_grads(dir_b)
    assert grads_a, f"no gradient dumps found in {dir_a}"
    assert grads_b, f"no gradient dumps found in {dir_b}"

    common_files = sorted(set(grads_a) & set(grads_b))
    assert common_files, (
        f"no dump files in common: {sorted(grads_a)} vs {sorted(grads_b)}"
    )

    n_compared = 0
    mismatches: list[str] = []
    for fname in common_files:
        ga, gb = grads_a[fname], grads_b[fname]
        common_params = set(ga) & set(gb)
        only_a = set(ga) - set(gb)
        only_b = set(gb) - set(ga)
        if only_a or only_b:
            mismatches.append(
                f"{fname}: param-set mismatch "
                f"(only_a={sorted(only_a)[:3]} only_b={sorted(only_b)[:3]})"
            )
        for name in sorted(common_params):
            ta, tb = ga[name].float(), gb[name].float()
            if ta.shape != tb.shape:
                mismatches.append(
                    f"{fname}:{name}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}"
                )
                continue
            n_compared += 1
            if torch.allclose(ta, tb, atol=atol, rtol=rtol, equal_nan=True):
                continue
            diff = (ta - tb).abs()
            denom = tb.abs().clamp_min(1e-12)
            mismatches.append(
                f"{fname}:{name}: max_abs={diff.max().item():.3e} "
                f"max_rel={(diff / denom).max().item():.3e} "
                f"(shape={tuple(ta.shape)})"
            )
    return n_compared, mismatches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.timeout(2200)
@pytest.mark.skipif(
    not has_h100_quad(),
    reason="Phase-7 grad-parity smoke requires >=4 GPUs.",
)
@pytest.mark.skipif(
    not mps_works(),
    reason=(
        "Phase-7 grad-parity needs the colocate path to actually run, "
        "which needs working NVIDIA MPS (see tests/colocate/_mps_probe.py)."
    ),
)
def test_phase7_grad_parity_smoke():
    """One colocate step finishes with a finite, non-zero training loss."""
    log = _run_arm(
        "colocate_qwen3_8b.yaml",
        dump_dir=Path(tempfile.mkdtemp(prefix="gradsmoke-")),
        visible_devices="0,1,2,3",
        extra_args=[
            "training.training_num_gpus_per_node=4",
            "inference.inference_num_gpus=4",
            "inference.inference_num_gpus_per_engine=1",
            "inference.inference_num_gpus_per_node=4",
            "inference.sglang.tp_size=1",
        ],
        timeout_s=1300,
    )
    loss = _extract_loss(log)
    assert loss == loss and loss != 0.0 and abs(loss) < 1e6, (
        f"colocate loss is suspect: {loss!r}. Either NaN/inf "
        f"(numerics broke) or 0/huge (data plane is dropping data)."
    )


@pytest.mark.timeout(60 * 60)
@pytest.mark.skipif(
    not has_n_gpus(1),
    reason="grad-parity determinism needs >=1 GPU.",
)
@pytest.mark.skipif(
    not mps_works(),
    reason="grad-parity determinism needs working NVIDIA MPS.",
)
def test_phase7_grad_parity_determinism():
    """The colocate path is bit-reproducible across runs of the same seed.

    Run the tiny colocate config twice with seed=42 and assert every
    dumped per-parameter gradient is bit-identical. A mismatch means
    something in the colocate pipeline — the gloo CPU-staged transfer,
    a non-deterministic kernel, or unseeded RNG — injected noise.
    """
    tmp = Path(tempfile.mkdtemp(prefix="graddet-"))
    _run_arm("colocate_qwen0p6b_tiny.yaml", dump_dir=tmp / "run_a",
             visible_devices="0", seed=42)
    _run_arm("colocate_qwen0p6b_tiny.yaml", dump_dir=tmp / "run_b",
             visible_devices="0", seed=42)

    n, mismatches = _compare_grad_dumps(
        tmp / "run_a", tmp / "run_b", atol=0.0, rtol=0.0
    )
    assert n > 0, "no gradients were compared"
    assert not mismatches, (
        f"colocate path is non-deterministic — {len(mismatches)} of {n} "
        f"gradients differ across two same-seed runs:\n  "
        + "\n  ".join(mismatches[:20])
    )
    print(f"[grad-parity] determinism OK: {n} gradients bit-identical")


@pytest.mark.timeout(90 * 60)
@pytest.mark.skipif(
    not has_n_gpus(2),
    reason="grad-parity full needs >=2 GPUs (1 trainer + 1 disagg engine).",
)
@pytest.mark.skipif(
    not mps_works(),
    reason="grad-parity full needs working NVIDIA MPS for the colocate arm.",
)
@pytest.mark.skipif(
    not _disagg_runnable(),
    reason=(
        "grad-parity full needs an importable Mooncake store for the "
        "disagg baseline arm (apt-get install libibverbs1 libnuma1 "
        "librdmacm1 libnl-3-200)."
    ),
)
def test_phase7_grad_parity_full():
    """Per-parameter gradient parity: colocate vs the disagg baseline.

    Both arms are dp_size=1 (single trainer rank) so FSDP is a no-op and
    there is no all-reduce reduction-order term. With the same seed,
    deterministic prompt order, and identical draft-training config, the
    only thing that differs is the hidden-state transport — Mooncake
    (disagg) vs gloo CPU-staged (colocate). Both are lossless copies, so
    the draft-model gradients must match within ``GRAD_ATOL``/``GRAD_RTOL``.

    A mismatch means the colocate transport is *not* delivering the same
    hidden states the disagg path would — the exact failure the design
    doc's validation plan calls for.
    """
    tmp = Path(tempfile.mkdtemp(prefix="gradfull-"))

    # The disagg arm is a non-colocate run. If an MPS daemon is up on
    # this node (the determinism test / colocate arm starts one), the
    # disagg actors get caught by MPS and die on its CUDA_VISIBLE_DEVICES
    # validation. Stop the daemon first; the colocate arm's train_entry
    # restarts it via setup_for_colocate.
    from torchspec.colocate.mps import stop_mps_daemon

    stop_mps_daemon()

    # Disagg baseline arm: 2 GPUs (trainer + engine disjoint), MPS off.
    _run_arm("disagg_qwen0p6b_tiny.yaml", dump_dir=tmp / "disagg",
             visible_devices="0,1", seed=42, disable_mps=True)
    # Colocate arm: 1 GPU (trainer + engine MPS-shared).
    _run_arm("colocate_qwen0p6b_tiny.yaml", dump_dir=tmp / "colocate",
             visible_devices="0", seed=42)

    n, mismatches = _compare_grad_dumps(
        tmp / "disagg", tmp / "colocate", atol=GRAD_ATOL, rtol=GRAD_RTOL
    )
    assert n > 0, "no gradients were compared"
    assert not mismatches, (
        f"grad parity FAILED — {len(mismatches)} of {n} draft-model "
        f"gradients diverge between disagg and colocate "
        f"(atol={GRAD_ATOL}, rtol={GRAD_RTOL}):\n  "
        + "\n  ".join(mismatches[:20])
    )
    print(f"[grad-parity] full OK: {n} gradients match disagg baseline")
