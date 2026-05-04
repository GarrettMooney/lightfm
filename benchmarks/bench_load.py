"""Axis B: load-time benchmark.

Measures wall-clock time for three load paths:
- joblib.load (the old d152 pattern)
- InferenceLightFM.load(mmap=False) (heap read)
- InferenceLightFM.load(mmap=True) (lazy mmap)

No cold-cache control — reports steady-state warm-cache timing.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import joblib

from lightfm.inference.model import InferenceLightFM


def _time_call(fn, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
        "repeats": repeats,
    }


def run(artifact_dir: Path, repeats: int = 5) -> dict:
    """Time the three load paths; return per-method summaries."""
    artifact_dir = Path(artifact_dir)
    joblib_path = str(artifact_dir / "model.joblib")
    safetensors_path = str(artifact_dir / "model.safetensors")

    return {
        "joblib": _time_call(lambda: joblib.load(joblib_path), repeats),
        "inference_heap": _time_call(
            lambda: InferenceLightFM.load(safetensors_path, mmap=False), repeats
        ),
        "inference_mmap": _time_call(
            lambda: InferenceLightFM.load(safetensors_path, mmap=True), repeats
        ),
    }
