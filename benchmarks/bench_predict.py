"""Axis C: predict throughput.

For each batch size, compare LightFM.predict (via joblib-loaded model) against
InferenceLightFM.predict (heap-loaded). Reports min/median/max seconds and
predictions/sec derived from the median.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import joblib
import numpy as np

from lightfm.inference.model import InferenceLightFM

DEFAULT_BATCH_SIZES = (1_000, 10_000, 100_000, 1_000_000)


def _time_predict(model, user_ids: np.ndarray, item_ids: np.ndarray, repeats: int) -> dict:
    # Warmup
    model.predict(user_ids, item_ids)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(user_ids, item_ids)
        samples.append(time.perf_counter() - t0)
    median = statistics.median(samples)
    per_sec = int(len(user_ids) / median) if median > 0 else 0
    return {
        "min": min(samples),
        "median": median,
        "max": max(samples),
        "per_sec": per_sec,
    }


def run(
    artifact_dir: Path,
    repeats: int = 3,
    batch_sizes: tuple[int, ...] | list[int] = DEFAULT_BATCH_SIZES,
    seed: int = 0,
) -> list[dict]:
    """Run predict timings; return one dict per batch size."""
    artifact_dir = Path(artifact_dir)
    lightfm_model = joblib.load(str(artifact_dir / "model.joblib"))
    inference_model = InferenceLightFM.load(
        str(artifact_dir / "model.safetensors"), mmap=False
    )

    n_users = lightfm_model.user_embeddings.shape[0]
    n_items = lightfm_model.item_embeddings.shape[0]

    rng = np.random.default_rng(seed)
    results = []
    for batch in batch_sizes:
        user_ids = rng.integers(0, n_users, size=batch, dtype=np.int32)
        item_ids = rng.integers(0, n_items, size=batch, dtype=np.int32)
        lf = _time_predict(lightfm_model, user_ids, item_ids, repeats)
        inf = _time_predict(inference_model, user_ids, item_ids, repeats)
        ratio = (inf["per_sec"] / lf["per_sec"]) if lf["per_sec"] else 0.0
        results.append({
            "batch_size": batch,
            "lightfm": lf,
            "inference": inf,
            "ratio": ratio,
        })
    return results
