"""Top-k throughput: BLAS predict_top_k vs current d152 path.

Two paths compared on a d152-shaped synthetic model:
- "current" — replicates d152 prod: model.predict(np.repeat(user_ids, n_items),
  np.tile(np.arange(n_items), n_batch)) followed by reshape + argpartition.
- "topk" — model.predict_top_k(user_ids, k, n_items=...).

Both use no features (matching d152 prod call shape). Reports min/median/max
seconds per batch, predictions/sec on the median, and the speedup ratio.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from benchmarks.models import make_fitted_model

DEFAULT_BATCH_SIZES = (1_000, 2_000, 5_000, 10_000)
DEFAULT_K = 100


def _time_current(model, user_ids: np.ndarray, n_items: int, k: int, repeats: int) -> dict:
    n_batch = len(user_ids)
    user_grid = np.repeat(user_ids, n_items).astype(np.int32)
    item_grid = np.tile(np.arange(n_items, dtype=np.int32), n_batch)

    def _once():
        scores = model.predict(user_grid, item_grid, num_threads=1)
        scores = scores.reshape(n_batch, n_items)
        topk_unsorted = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        topk_scores = np.take_along_axis(scores, topk_unsorted, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        np.take_along_axis(topk_unsorted, order, axis=1)

    _once()  # warmup
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _once()
        samples.append(time.perf_counter() - t0)
    return _summarize(samples, n_batch)


def _time_topk(model, user_ids: np.ndarray, n_items: int, k: int, repeats: int) -> dict:
    def _once():
        model.predict_top_k(user_ids, k=k, n_items=n_items)

    _once()  # warmup
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _once()
        samples.append(time.perf_counter() - t0)
    return _summarize(samples, len(user_ids))


def _summarize(samples: list[float], n_users: int) -> dict:
    median = statistics.median(samples)
    return {
        "min": min(samples),
        "median": median,
        "max": max(samples),
        "users_per_sec": int(n_users / median) if median > 0 else 0,
    }


def run(
    n_users: int = 50_000,
    n_items: int = 118_000,
    no_components: int = 200,
    k: int = DEFAULT_K,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    repeats: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Build a d152-shaped synthetic model and time both paths per batch size."""
    print(
        f"Building synthetic model: n_users={n_users:,} n_items={n_items:,} "
        f"no_components={no_components}",
        flush=True,
    )
    model = make_fitted_model(
        n_users=n_users, n_items=n_items, no_components=no_components, seed=seed
    )
    rng = np.random.default_rng(seed)

    results = []
    for batch in batch_sizes:
        user_ids = rng.integers(0, n_users, size=batch, dtype=np.int32)
        print(f"  batch={batch:,}…", end="", flush=True)
        cur = _time_current(model, user_ids, n_items, k, repeats)
        top = _time_topk(model, user_ids, n_items, k, repeats)
        speedup = cur["median"] / top["median"] if top["median"] > 0 else 0.0
        results.append({
            "batch": batch,
            "current": cur,
            "topk": top,
            "speedup": speedup,
        })
        print(
            f" current={cur['median']*1000:.1f}ms ({cur['users_per_sec']:,} u/s) "
            f"topk={top['median']*1000:.1f}ms ({top['users_per_sec']:,} u/s) "
            f"speedup={speedup:.1f}×",
            flush=True,
        )
    return results


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--n-users", type=int, default=50_000,
                   help="synthetic user count (kept smallish so model fits in RAM)")
    p.add_argument("--n-items", type=int, default=118_000,
                   help="d152 prod has 118,543 items")
    p.add_argument("--no-components", type=int, default=200, help="d152 prod = 200")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=list(DEFAULT_BATCH_SIZES))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=None,
                   help="optional JSON output path")
    args = p.parse_args()

    out = run(
        n_users=args.n_users,
        n_items=args.n_items,
        no_components=args.no_components,
        k=args.k,
        batch_sizes=tuple(args.batch_sizes),
        repeats=args.repeats,
    )
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nresults → {args.out}")
