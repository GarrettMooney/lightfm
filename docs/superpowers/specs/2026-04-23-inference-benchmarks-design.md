# Inference Benchmarks Design

**Date:** 2026-04-23
**Branch:** `feature/inference-artifact` (continues on same branch)
**Status:** Draft, pending spec review

## Problem

The inference-artifact work (spec `2026-04-21-inference-artifact-design.md`) claims:
- Artifact size ~60GB → ~30GB (training state removed)
- Runtime RAM ~8× model size → ~1× model size (mmap shared across forks)

These claims rest on a single PSS/RSS test at ~670MB and a bit-identical parity test. That's enough to prove correctness. It is not enough to size a production rollout, compare across hardware, or detect perf regressions over time.

This spec adds a benchmark suite that measures four axes — artifact size, load time, predict throughput, multi-process RAM — comparing the old `joblib.dump(LightFM)` + `Pool(processes=N)` pattern against `save_for_inference` + `InferenceLightFM.load(mmap=True)` + fork. Output is a printed table plus a JSON sidecar for diffing runs.

## Goals

- Quantify, at multiple model sizes, the size / load / predict / RAM deltas between old and new patterns.
- Provide a reproducible, scriptable benchmark entry point so engineers can run it on representative hardware.
- Produce JSON artifacts that can be diffed across runs, committed to PR descriptions, or fed into a results tracker later.
- Keep benchmarks separate from the correctness test suite — different discipline, different cadence.

## Non-Goals

- **CI regression gating.** v1 is investigation tooling; no exit-code-based regression bands. A future v2 could add asserts on top of the existing JSON output.
- **Training benchmarks.** The inference-artifact work doesn't change `fit()`. No `fit` throughput measurements.
- **Cross-framework comparisons** (PyTorch, JAX). Future-state per inference-artifact spec's v2 hooks.
- **Real-workload traces.** Synthetic batches only — keeps the benchmarks hermetic.
- **Matplotlib charts / markdown reports.** Maintenance burden beyond what's needed; JSON + ad-hoc plotting is enough.
- **Forcing cold page cache.** Best-effort only (drop_caches needs root on Linux; no equivalent on macOS). Document the measurement regime honestly.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Layout | Top-level `benchmarks/` directory | Separate from `tests/` — different discipline (timing vs correctness). Clean CLI story. |
| Invocation | `uv run python -m benchmarks --size {tiny,medium,large,custom}` | Parametric size; one codebase handles CI-fast + local-realistic + workstation-full. |
| Output | Printed table + JSON sidecar | Table for the terminal; JSON for diffs, PR descriptions, cross-hardware comparisons. No charts. |
| Axes | All four (size, load, predict, multi-proc RAM) | Size + RAM are load-bearing claims; load + predict are where surprises hide. |
| Model generation | Synthetic via `LightFM._initialize()` + random fill | Avoids training cost at medium/large sizes; captures the real array layout that matters for size/load/RAM. |
| Multi-proc RAM is A/B | Compare `joblib.load` + Pool vs `mmap` + Pool | The whole point is the delta; measuring only variant B is half the story. |

## Architecture

New top-level `benchmarks/` package:

```
benchmarks/
  __init__.py            # version, public export of runner API
  __main__.py            # CLI entry point
  models.py              # synthetic fitted-model factory + size presets
  bench_size.py          # axis A: artifact bytes
  bench_load.py          # axis B: wall-clock load
  bench_predict.py       # axis C: predictions/sec at various batch sizes
  bench_ram.py           # axis D: multi-process RAM (A/B joblib vs mmap)
  results.py             # shared: table printer + JSON writer
  results/
    .gitkeep
    sample-medium.json   # committed reference run
```

`benchmarks/results/*.json` is `.gitignore`d except `sample-medium.json` (checked in for documentation / PR reference). `.gitkeep` preserves the directory.

## Components

### `models.py` — Synthetic fitted-model factory

```python
# benchmarks/models.py

SIZE_PRESETS = {
    "tiny":   dict(n_users=1_000,     n_items=5_000,      no_components=32),
    "medium": dict(n_users=300_000,   n_items=1_000_000,  no_components=128),
    "large":  dict(n_users=5_000_000, n_items=20_000_000, no_components=128),
}

def make_fitted_model(
    *,
    n_users: int,
    n_items: int,
    no_components: int,
    loss: str = "warp",
    learning_schedule: str = "adagrad",
    seed: int = 0,
) -> LightFM:
    """Return a LightFM instance with all arrays (embeddings, gradients,
    momentum) allocated and filled with pseudo-random float32 data — as if
    `fit()` had run. Skips the actual training cost.
    """
```

Uses `LightFM._initialize(no_components, n_items, n_users)` to set up the 12 arrays with the right shapes, then fills each with `np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)`. The `gradients` arrays for adagrad have the `+= 1` convention that `_initialize` already handles.

Size accounting (float32 = 4 bytes):
- `tiny`: 2×(1K+5K)×32×4 + 4×(1K+5K)×4 + 2×(1K+5K)×32×4 [grads+momentum] ≈ 3.1 MB for arrays, ~800KB-1MB joblib on disk
- `medium`: 2×(300K+1M)×128×4 + ... ≈ 2.0 GB for arrays, ~670MB–~1.3GB on disk (joblib compresses somewhat)
- `large`: ≈ 37GB for arrays — requires a beefy workstation

### `bench_size.py` — Axis A

```python
def run(model: LightFM, artifact_dir: Path) -> dict:
    """Dump the model both ways and return byte counts."""
```

Writes `<artifact_dir>/model.joblib` via `joblib.dump(model, path)` and `<artifact_dir>/model.safetensors` via `model.save_for_inference(path)`. Returns:

```python
{
    "joblib_bytes": int,
    "safetensors_bytes": int,
    "reduction_ratio": safetensors_bytes / joblib_bytes,  # lower is better
    "reduction_pct": 100.0 * (1 - ratio),
}
```

The artifact files are left on disk so `bench_load.py` can reuse them without re-dumping.

### `bench_load.py` — Axis B

```python
def run(artifact_dir: Path, repeats: int = 5) -> dict:
    """Measure wall-clock load time across three methods."""
```

For each method, run `repeats` times, collect timings, return `{min, median, max, repeats}`:

- `joblib.load(artifact_dir / "model.joblib")`
- `InferenceLightFM.load(artifact_dir / "model.safetensors", mmap=False)`
- `InferenceLightFM.load(artifact_dir / "model.safetensors", mmap=True)`

Uses `time.perf_counter()`. No cold-cache control in v1 — documented in the output as "warm cache; measures steady-state load."

### `bench_predict.py` — Axis C

```python
BATCH_SIZES = [1_000, 10_000, 100_000, 1_000_000]

def run(artifact_dir: Path, repeats: int = 3, seed: int = 0) -> list[dict]:
    """Load both models once; for each batch size, time predict() 3x."""
```

1. Load both: `LightFM` from joblib, `InferenceLightFM` from safetensors (`mmap=False` for a fair in-memory comparison).
2. For each batch size:
   - Generate `user_ids`, `item_ids` as random int32 arrays (within the valid range for the model's dimensions).
   - Warmup: 1 `predict` call per model, untimed.
   - Timed: `repeats=3` runs of each model; report `{min, median, max, per_sec = batch_size / median}`.
3. Return a list of per-batch-size results.

For `tiny` models, drop batch sizes exceeding `n_users × n_items` or clamp item_ids to `[0, n_items)`.

### `bench_ram.py` — Axis D (the A/B multi-process test)

```python
def run(artifact_dir: Path, n_workers: int = 4) -> dict:
    """A: joblib.load + Pool. B: InferenceLightFM.load(mmap=True) + Pool.
    Measure per-worker PSS (Linux) / RSS (macOS) and return."""
```

Skips entirely on Windows (no fork). Skips on any platform if `n_workers < 1`.

**Variant A (joblib + fork)** — mirrors the old d152 pattern:
```python
parent_model = joblib.load(joblib_path)
ctx = multiprocessing.get_context("fork")
with ctx.Pool(n_workers) as pool:
    results = pool.map(_joblib_worker, [(parent_model_is_captured_via_fork, seed) for seed in ...])
```

Wait — `pool.map`'s first arg is a callable, not a captured model. The model must be visible to the child at fork time. We accomplish this by binding `parent_model` into a module-level global before forking (or using a closure that `Pool` pickles with `forkserver`). For `fork` start method, children inherit the parent's address space, so the worker function can reference a module-level `_MODEL_A` variable set before `Pool()` is called.

**Variant B (mmap + fork)** — mirrors the `test_fork_sharing.py` approach:
```python
parent_inf = InferenceLightFM.load(safetensors_path, mmap=True)
# pre-touch to make sure mmap is instantiated
_ = parent_inf.item_embeddings[0, 0]
ctx = multiprocessing.get_context("fork")
with ctx.Pool(n_workers) as pool:
    results = pool.map(_mmap_worker, [(safetensors_path, seed) for seed in ...])
```

In both variants, the worker function:
1. Runs a `predict` batch (10K random pairs — small enough to be fast, large enough to touch many pages).
2. Samples its own memory footprint:
   - Linux: read `/proc/self/smaps_rollup` for `Pss:` in KB → bytes.
   - macOS: `psutil.Process(os.getpid()).memory_info().rss`.
3. Returns `(pid, bytes, platform_metric)`.

**Return payload:**
```python
{
    "platform": "linux" | "darwin",
    "measurement": "pss" | "rss",
    "n_workers": int,
    "model_bytes": int,  # from bench_size or re-stat'd
    "variant_a": {
        "per_worker": [int, ...],
        "total": int,  # sum
    },
    "variant_b": {...},
    "sharing_ratio": variant_a_total / variant_b_total,
    "caveats": [...],
}
```

**Caveats** appended to the JSON (string list):
- `"Measurement: PSS via /proc/self/smaps_rollup (Linux)"` or `"Measurement: RSS via psutil (macOS; counts all resident pages)"`
- `"Variant A (joblib + fork) creates anonymous CoW pages in each child; PSS is a lower bound on real footprint"`
- `"Synthetic model — embedding values are random float32, not trained. Array layout is representative; memory behavior is identical to a trained model."`

### `results.py` — Output

Two functions:
- `print_table(results: dict, metadata: dict) -> None` — prints the four-axis table to stdout.
- `write_json(results: dict, metadata: dict, path: Path) -> None` — writes the JSON sidecar.

JSON schema matches Section 4 of the brainstorm:
```json
{
  "metadata": {"timestamp", "size", "config", "host", "lightfm_version", "n_workers"},
  "axes": {"size": {...}, "load": {...}, "predict": [...], "ram": {...}}
}
```

The `host` block includes `platform`, `python`, `cpu_count` — enough to correlate across machines. No sensitive data.

### `__main__.py` — CLI

```
usage: python -m benchmarks [-h] [--size {tiny,medium,large,custom}]
                            [--n-users N] [--n-items N] [--no-components N]
                            [--n-workers N] [--repeats-load N] [--repeats-predict N]
                            [--skip-axis {size,load,predict,ram}]
                            [--output-dir PATH] [--label LABEL]
```

Behavior:
1. Resolve size (preset or custom kwargs).
2. Create a temp dir under `--output-dir` (default `benchmarks/results/`) for artifact files, and a JSON result path `<size>-<iso-date>-<label?>.json`.
3. Build the synthetic model.
4. Run each non-skipped axis in order (size → load → predict → ram).
5. Print table; write JSON.
6. On success: print the JSON path on the final line; exit 0.
7. On benchmark-level failure (e.g., one axis errors): print traceback, continue remaining axes, include error in JSON output, exit 1.
8. On environment error (can't import): exit 2.

## Error Handling

- **Out-of-memory during model build** — `make_fitted_model` catches `MemoryError`, raises a clear message with required bytes and suggests running at a smaller size.
- **Windows — axis D** — `bench_ram.run()` returns `{"skipped": True, "reason": "fork unavailable on Windows"}` with no error. Other axes run normally.
- **Per-axis errors** — caught in `__main__.py`; a single axis failure doesn't abort the others. The JSON output includes a per-axis `"error": "..."` field for any axis that failed.
- **Missing `psutil`** on macOS — `bench_ram.run()` returns `{"skipped": True, "reason": "psutil not installed"}`. `psutil` is already in `[project.optional-dependencies].dev` from the inference-artifact work.

## Testing

Benchmarks themselves are not tested as correctness suites — they test correctness elsewhere. But we do want:

- **One smoke test** in `tests/benchmarks/test_smoke.py` that runs the benchmark CLI at `--size tiny` and asserts it produces a valid JSON file. Catches breakage of the benchmark machinery itself. Skippable via `-m "not slow"` marker if tiny still takes a few seconds.
- **`make_fitted_model` should produce a model whose `.predict()` doesn't raise** — covered by the smoke test.

Not tested: the numbers themselves. These are measurements, not assertions.

## Files Added

- `benchmarks/__init__.py`
- `benchmarks/__main__.py`
- `benchmarks/models.py`
- `benchmarks/bench_size.py`
- `benchmarks/bench_load.py`
- `benchmarks/bench_predict.py`
- `benchmarks/bench_ram.py`
- `benchmarks/results.py`
- `benchmarks/results/.gitkeep`
- `benchmarks/results/sample-medium.json` (committed after first real run on reviewer's hardware)
- `tests/benchmarks/__init__.py`
- `tests/benchmarks/test_smoke.py`

## Files Modified

- `.gitignore` — add `benchmarks/results/*.json` and negate `!benchmarks/results/sample-*.json`
- `pyproject.toml` — no changes required (uses existing `numpy`, `scipy`, `joblib` already present; `psutil` already in dev extras)

## Expected Outcomes

Representative `medium` numbers (estimated from the existing fork-sharing test's 0.43× per-worker RSS):

| Axis | Expected |
|---|---|
| Size reduction | ~50% (drop gradients + momentum, which are ~half the joblib bytes for adagrad) |
| Load time (joblib → mmap) | 10-100× faster mmap cold-start |
| Predict throughput | Within 1% of `LightFM.predict` (dataclass overhead is negligible vs Cython kernel time) |
| RAM sharing ratio | 3-5× on Linux with N=4 workers (joblib-fork ≈ N× model, mmap-fork ≈ 0.8× model) |

These are predictions; the benchmarks produce the real numbers.

## Out of Scope / v2 Ideas

- Cold-cache control (requires root; separate sudo flow)
- Historical results dashboard / GitHub Action that runs benchmarks on PR
- CI regression bands with asserted ratios
- GPU-backend comparison (waits for v2 GPU work)
- float16 / quantization comparisons (waits for v2)
- Benchmark comparing `.predict` parallelism via `num_threads=N` (LightFM Cython parallelism, orthogonal to the fork-sharing story)
