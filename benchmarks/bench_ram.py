"""Axis D: multi-process RAM comparison.

Variant A: joblib.load(parent) + Pool(fork). Children inherit the full
LightFM instance via CoW. Mirrors the old d152 pattern.

Variant B: InferenceLightFM.load(parent, mmap=True) + Pool(fork). Children
open their own mmap; OS page cache shares the file-backed pages.

Measurement:
- Linux: /proc/self/smaps_rollup PSS (proportional set size — correctly
  accounts shared pages)
- macOS: psutil RSS (counts all resident pages; weaker signal but the best
  available)
- Windows: skipped (fork unavailable)

CRITICAL: Variant A passes the model via fork inheritance (module-level
_MODEL_A global set before Pool()), NOT through Pool.map's arg queue.
Pickling the model through the queue would measure pickle/unpickle
overhead instead of CoW behavior and invalidate the A/B comparison.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

import joblib
import numpy as np

from lightfm.inference.model import InferenceLightFM

# Module-level global — populated by _run_variant_a before Pool() forks. Must
# be at module scope so children inherit it via the fork start method.
_MODEL_A = None


def _sample_memory_bytes() -> tuple[str, int]:
    """Return ('pss'|'rss', bytes) for the current process."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"):
                        return "pss", int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass  # fall through to RSS on older kernels
    import psutil
    return "rss", psutil.Process(os.getpid()).memory_info().rss


def _joblib_worker(seed: int) -> tuple[int, int]:
    """Variant A worker. Reads _MODEL_A from the module namespace — inherited
    via fork, no pickle round-trip."""
    rng = np.random.default_rng(seed)
    n_users = _MODEL_A.user_embeddings.shape[0]
    n_items = _MODEL_A.item_embeddings.shape[0]
    user_ids = rng.integers(0, n_users, size=10_000, dtype=np.int32)
    item_ids = rng.integers(0, n_items, size=10_000, dtype=np.int32)
    _MODEL_A.predict(user_ids, item_ids)
    _, bytes_ = _sample_memory_bytes()
    return os.getpid(), bytes_


def _mmap_worker(args: tuple[str, int]) -> tuple[int, int]:
    """Variant B worker. Opens its own mmap; OS shares file-backed pages."""
    path, seed = args
    model = InferenceLightFM.load(path, mmap=True)
    rng = np.random.default_rng(seed)
    n_users = model.user_embeddings.shape[0]
    n_items = model.item_embeddings.shape[0]
    user_ids = rng.integers(0, n_users, size=10_000, dtype=np.int32)
    item_ids = rng.integers(0, n_items, size=10_000, dtype=np.int32)
    model.predict(user_ids, item_ids)
    _, bytes_ = _sample_memory_bytes()
    return os.getpid(), bytes_


def _run_variant_a(joblib_path: Path, n_workers: int) -> list[tuple[int, int]]:
    global _MODEL_A
    _MODEL_A = joblib.load(str(joblib_path))
    try:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            return pool.map(_joblib_worker, list(range(n_workers)))
    finally:
        _MODEL_A = None


def _run_variant_b(safetensors_path: Path, n_workers: int) -> list[tuple[int, int]]:
    parent = InferenceLightFM.load(str(safetensors_path), mmap=True)
    _ = parent.item_embeddings[0, 0]  # fault in the first page so the mmap is live
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        return pool.map(_mmap_worker, [(str(safetensors_path), i) for i in range(n_workers)])


def run(artifact_dir: Path, n_workers: int = 4) -> dict:
    """Run both variants; return the full axis-D payload."""
    if sys.platform == "win32":
        return {"skipped": True, "reason": "fork unavailable on Windows"}

    artifact_dir = Path(artifact_dir)
    joblib_path = artifact_dir / "model.joblib"
    safetensors_path = artifact_dir / "model.safetensors"

    # Detect measurement type by sampling ourselves once.
    measurement, _ = _sample_memory_bytes()
    platform = "linux" if sys.platform.startswith("linux") else "darwin"

    a_results = _run_variant_a(joblib_path, n_workers)
    b_results = _run_variant_b(safetensors_path, n_workers)

    a_per = [bytes_ for _, bytes_ in a_results]
    b_per = [bytes_ for _, bytes_ in b_results]
    a_total = sum(a_per)
    b_total = sum(b_per)
    sharing = (a_total / b_total) if b_total else 0.0

    caveats = [
        f"Measurement: {measurement.upper()} ({'Linux smaps_rollup' if platform == 'linux' else 'psutil RSS on macOS — counts all resident pages'})",
        "Variant A (joblib + fork) creates anonymous CoW pages in each child; PSS is a lower bound on real footprint",
        "Synthetic model — embedding values are random float32, not trained. Array layout is representative; memory behavior is identical to a trained model.",
    ]

    return {
        "platform": platform,
        "measurement": measurement,
        "n_workers": n_workers,
        "model_bytes": safetensors_path.stat().st_size,
        "variant_a": {"per_worker": a_per, "total": a_total},
        "variant_b": {"per_worker": b_per, "total": b_total},
        "sharing_ratio": sharing,
        "caveats": caveats,
    }
