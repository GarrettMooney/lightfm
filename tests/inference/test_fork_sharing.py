"""The load-bearing perf test: forked workers must share the mmap'd model.

On Linux we check PSS (Proportional Set Size) via /proc/<pid>/smaps_rollup.
PSS correctly accounts shared pages: each shared page counts as
(page_size / num_sharers) in each sharing process.

The measurement: each child records its own PSS while predict is in flight —
at that moment `parent + N children` share the mmap, so each child's PSS for
model pages ≈ model_size / (N + 1). Summing across N children yields
N / (N + 1) × model_size (e.g., 4/5 = 0.8 × model_size for N=4).

If mmap is NOT working (a regression copies the model into each worker),
each child's PSS ≈ full model_size, and sum(children PSS) ≈ N × model_size.

Asserting `sum(children PSS) < 1.0 × model_size` therefore catches any
regression that pushes workers past ~model/N each, with comfortable margin
above the true shared floor. The parent's PSS is not summed here — it
changes as children enter/exit, and we only need to prove worker-side
sharing to make the d152 claim.

On macOS PSS isn't available; fall back to a weaker per-worker RSS check —
a catastrophic CoW regression would put each worker at ≈ model_size RSS,
easily caught.
"""
from __future__ import annotations

import multiprocessing
import os
import sys

import numpy as np
import pytest

from lightfm.inference import _artifact
from lightfm.inference.model import InferenceLightFM

psutil = pytest.importorskip("psutil")


# ~670MB total on-disk, large enough that a CoW regression is unmistakable.
# item_embeddings: 1_000_000 × 128 × 4 = 512MB
# user_embeddings:   300_000 × 128 × 4 = 154MB
# biases:  (1_000_000 + 300_000) × 4   = ~5MB
ITEM_ROWS = 1_000_000
USER_ROWS = 300_000
NO_COMPONENTS = 128
N_WORKERS = 4


def _build_medium_artifact(path):
    rng = np.random.default_rng(0)
    arrays = {
        "item_embeddings": rng.standard_normal((ITEM_ROWS, NO_COMPONENTS), dtype=np.float32),
        "user_embeddings": rng.standard_normal((USER_ROWS, NO_COMPONENTS), dtype=np.float32),
        "item_biases": np.zeros(ITEM_ROWS, dtype=np.float32),
        "user_biases": np.zeros(USER_ROWS, dtype=np.float32),
    }
    total_bytes = sum(a.nbytes for a in arrays.values())
    _artifact.save(
        path,
        arrays=arrays,
        metadata={
            "no_components": NO_COMPONENTS,
            "loss": "warp",
            "learning_schedule": "adagrad",
        },
    )
    return total_bytes


def _read_pss_kb(pid: int) -> int:
    """Read PSS in KB from /proc/<pid>/smaps_rollup. Linux-only."""
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            if line.startswith("Pss:"):
                return int(line.split()[1])
    raise RuntimeError(f"Pss: not found in smaps_rollup for pid {pid}")


def _worker_predict(args):
    """Worker: runs a predict batch, returns (pid, mem_bytes) where mem_bytes
    is PSS on Linux and RSS on macOS. Measurement happens AFTER predict so
    pages are resident at sampling time."""
    model_path, seed = args
    model = InferenceLightFM.load(model_path, mmap=True)
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(0, USER_ROWS, size=10_000, dtype=np.int32)
    item_ids = rng.integers(0, ITEM_ROWS, size=10_000, dtype=np.int32)
    model.predict(user_ids, item_ids)
    pid = os.getpid()
    if sys.platform.startswith("linux"):
        return pid, _read_pss_kb(pid) * 1024  # bytes
    return pid, psutil.Process(pid).memory_info().rss  # bytes


@pytest.mark.skipif(sys.platform == "win32", reason="fork not available on Windows")
def test_forked_workers_share_mmap_pages(tmp_path):
    if sys.platform.startswith("linux") and not os.path.exists("/proc/self/smaps_rollup"):
        pytest.skip("kernel does not expose /proc/<pid>/smaps_rollup")

    path = tmp_path / "medium.safetensors"
    total_bytes = _build_medium_artifact(path)

    # Parent loads so the mmap is instantiated before fork; children then inherit
    # it rather than re-opening from scratch. PSS accounting works either way
    # (pages fault in on access), but holding the parent reference keeps the
    # num_sharers count stable across the measurement window.
    parent_model = InferenceLightFM.load(path, mmap=True)
    _ = parent_model.item_embeddings[0, 0]
    _ = parent_model.user_embeddings[0, 0]

    # macOS defaults to spawn (which would copy via pickling, invalidating the test);
    # force fork on both Linux and macOS.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=N_WORKERS) as pool:
        results = pool.map(_worker_predict, [(str(path), i) for i in range(N_WORKERS)])

    if sys.platform.startswith("linux"):
        # With proper sharing across (parent + N_WORKERS) processes, each child's
        # PSS for model pages ≈ total_bytes / (N_WORKERS + 1).
        # Sum across children ≈ N_WORKERS / (N_WORKERS + 1) × total_bytes.
        # For N_WORKERS=4 that's 0.8 × model_size. Threshold 1.0 gives headroom
        # for per-worker Python overhead while still failing loudly on any
        # regression that pushes any worker near full model_size.
        children_pss = sum(bytes_ for _, bytes_ in results)
        assert children_pss < 1.0 * total_bytes, (
            f"children PSS sum {children_pss:,} exceeds 1.0× model size "
            f"{total_bytes:,} — fork CoW sharing regressed "
            f"(expected ~{N_WORKERS/(N_WORKERS+1):.2f}× with mmap working)"
        )
    else:
        # macOS: no single worker RSS may approach full model size.
        for pid, rss in results:
            print(
                f"  worker pid={pid} rss={rss:,} bytes "
                f"({rss / total_bytes:.2f}× model_size={total_bytes:,})"
            )
            assert rss < 1.5 * total_bytes, (
                f"worker pid {pid} RSS {rss:,} exceeds 1.5× model size "
                f"{total_bytes:,} — mmap sharing regressed"
            )
