# Inference Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone four-axis benchmark suite (artifact size / load time / predict throughput / multi-process RAM) under `benchmarks/` that A/B-compares the old `joblib.dump(LightFM)` + `Pool` pattern against `save_for_inference` + `InferenceLightFM.load(mmap=True)` + fork. Output is a printed table plus a JSON sidecar.

**Architecture:** New top-level `benchmarks/` package. One module per axis (`bench_size.py`, `bench_load.py`, `bench_predict.py`, `bench_ram.py`), plus `models.py` (synthetic fitted-model factory at parametric sizes), `results.py` (table + JSON output), and `__main__.py` (CLI). Benchmarks live separately from `tests/` — different discipline (timing vs correctness). A single smoke test under `tests/benchmarks/test_smoke.py` guards the benchmark machinery itself.

**Tech Stack:** Python 3.8+, numpy, scipy.sparse, joblib, psutil (dev extras, already present), safetensors (runtime dep, already present), `lightfm.inference` subpackage (already on this branch).

**Spec:** `docs/superpowers/specs/2026-04-23-inference-benchmarks-design.md`

**Branch:** `feature/inference-artifact` (continues on the existing branch from the inference-artifact work)

---

## Notes for the implementer

- **Run commands from `/Users/garrettmooney/git/lightfm`.** User prefers `uv run <cmd>` over bare `python`/`python3`.
- **TDD for behavioral code (bench modules, models).** For scaffolding (empty `__init__.py`, `.gitignore` edits) and CLI plumbing, write the code first then verify with a smoke run.
- **The fork-global trick for Variant A is load-bearing.** The spec explicitly calls out why: passing the model through `Pool.map`'s arg queue would pickle/unpickle per worker, measuring the wrong thing. See Task 7 — treat the pattern as exact-code, don't "improve" it.
- **Axes A→B→D have an implicit artifact dependency.** `bench_size` writes `model.joblib` + `model.safetensors`; `bench_load` and `bench_ram` read them. `__main__.py` must auto-run `bench_size` if its outputs are needed but missing.
- **Skip axes gracefully.** Windows skips axis D with a notice; macOS/Linux run everything. `--skip-axis` flag takes comma-separated names.
- **Never import from `tests/`.** `benchmarks/` is a sibling top-level package. Don't share fixtures across them; `make_fitted_model` is the canonical model factory for benchmarks.

---

## File Structure

```
benchmarks/
  __init__.py            # empty (or version export)
  __main__.py            # CLI entry point
  models.py              # make_fitted_model + SIZE_PRESETS
  bench_size.py          # axis A
  bench_load.py          # axis B
  bench_predict.py       # axis C
  bench_ram.py           # axis D (A/B joblib-fork vs mmap-fork)
  results.py             # table printer + JSON writer
  results/
    .gitkeep
    sample-medium.json   # committed after first real medium run

tests/
  benchmarks/
    __init__.py
    test_smoke.py        # --size tiny smoke under @pytest.mark.slow
```

---

## Chunk 1: Benchmark suite

Single chunk with strict sequential dependencies (models.py before bench_*.py, bench_*.py before __main__.py). The chunk exceeds the 1000-line guideline but splitting would create artificial sub-chunks without changing execution — every task depends on the previous one's output, and the code listings take up most of the line count.

### Task 1: Scaffold the benchmarks package

**Files:**
- Create: `benchmarks/__init__.py` (empty)
- Create: `benchmarks/results/.gitkeep` (empty)
- Create: `tests/benchmarks/__init__.py` (empty)
- Modify: `.gitignore` (ignore `benchmarks/results/*.json`, keep sample)
- Modify: `pyproject.toml` (register `slow` pytest marker)

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p benchmarks/results tests/benchmarks
touch benchmarks/__init__.py benchmarks/results/.gitkeep tests/benchmarks/__init__.py
```

- [ ] **Step 2: Update `.gitignore`**

Read the current `.gitignore`. Append:

```
# Benchmark result files are ignored except committed samples
benchmarks/results/*.json
!benchmarks/results/sample-*.json
```

- [ ] **Step 3: Register `slow` marker in `pyproject.toml`**

Check whether `[tool.pytest.ini_options]` already exists. If yes, add `markers`. If no, append the section. Final state should contain:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 4: Smoke-test the package imports**

Run: `uv run python -c "import benchmarks; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Full test suite still passes**

Run: `uv run pytest tests/ -q`
Expected: 105 passed, 1 skipped (unchanged from prior state).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/ tests/benchmarks/ .gitignore pyproject.toml
git commit -m "Scaffold benchmarks package and register slow marker"
```

---

### Task 2: `models.py` — synthetic fitted-model factory (TDD)

**Files:**
- Create: `benchmarks/models.py`
- Create: `tests/benchmarks/test_models.py`

- [ ] **Step 1: Write the failing tests**

`tests/benchmarks/test_models.py`:

```python
import numpy as np
import pytest

from benchmarks.models import SIZE_PRESETS, make_fitted_model
from lightfm import LightFM


def test_size_presets_shape():
    """Presets have the expected keys and dimensions."""
    assert set(SIZE_PRESETS) == {"tiny", "medium", "large"}
    for preset in ("tiny", "medium", "large"):
        cfg = SIZE_PRESETS[preset]
        assert "n_users" in cfg and "n_items" in cfg and "no_components" in cfg


def test_make_fitted_model_is_a_lightfm():
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    assert isinstance(model, LightFM)


def test_make_fitted_model_arrays_are_initialized():
    """All 12 arrays that FastLightFM needs are present and float32."""
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    for attr in (
        "item_embeddings", "item_embedding_gradients", "item_embedding_momentum",
        "item_biases", "item_bias_gradients", "item_bias_momentum",
        "user_embeddings", "user_embedding_gradients", "user_embedding_momentum",
        "user_biases", "user_bias_gradients", "user_bias_momentum",
    ):
        arr = getattr(model, attr)
        assert arr is not None, attr
        assert arr.dtype == np.float32, attr


def test_make_fitted_model_predict_works():
    """Predict runs without error — the synthetic model is a valid LightFM."""
    model = make_fitted_model(n_users=50, n_items=100, no_components=4, seed=0)
    user_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    item_ids = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    scores = model.predict(user_ids, item_ids)
    assert scores.shape == (5,)
    assert scores.dtype == np.float32


def test_make_fitted_model_deterministic_with_seed():
    m1 = make_fitted_model(n_users=10, n_items=20, no_components=4, seed=42)
    m2 = make_fitted_model(n_users=10, n_items=20, no_components=4, seed=42)
    np.testing.assert_array_equal(m1.item_embeddings, m2.item_embeddings)
    np.testing.assert_array_equal(m1.user_embeddings, m2.user_embeddings)
```

- [ ] **Step 2: Run tests — expect failures**

Run: `uv run pytest tests/benchmarks/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name ...` — `benchmarks.models` doesn't exist.

- [ ] **Step 3: Implement `benchmarks/models.py`**

```python
"""Synthetic LightFM factory for benchmarking.

Builds a LightFM instance whose 12 arrays (embeddings, gradients, momentum,
biases) are allocated and filled with pseudo-random float32 values — as if
`fit()` had run for some number of epochs. Skips actual training cost so
medium/large-scale benchmarks are feasible.
"""
from __future__ import annotations

import numpy as np

from lightfm import LightFM

SIZE_PRESETS: dict[str, dict[str, int]] = {
    "tiny":   {"n_users": 1_000,     "n_items": 5_000,      "no_components": 32},
    "medium": {"n_users": 300_000,   "n_items": 1_000_000,  "no_components": 128},
    "large":  {"n_users": 5_000_000, "n_items": 20_000_000, "no_components": 128},
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
    """Return a LightFM whose arrays are shaped and filled as if `fit()` ran.

    Uses `LightFM._initialize` (the same private method `fit` calls internally)
    to allocate the 12 arrays, then overwrites the embeddings with fresh
    `standard_normal` draws so the model is distinct per seed. Gradients and
    momentum keep `_initialize`'s default values (including the adagrad `+= 1`
    convention). Does not call `fit`.
    """
    model = LightFM(
        no_components=no_components,
        loss=loss,
        learning_schedule=learning_schedule,
        random_state=seed,
    )
    # _initialize(no_components, no_item_features, no_user_features);
    # with identity feature matrices (the default), features == entities.
    model._initialize(no_components, n_items, n_users)

    rng = np.random.default_rng(seed)
    model.item_embeddings = rng.standard_normal(
        (n_items, no_components), dtype=np.float32
    )
    model.user_embeddings = rng.standard_normal(
        (n_users, no_components), dtype=np.float32
    )
    return model
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/benchmarks/test_models.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Full suite sanity check**

Run: `uv run pytest tests/ -q`
Expected: 110 passed, 1 skipped (105 + 5 new).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/models.py tests/benchmarks/test_models.py
git commit -m "Add synthetic fitted-model factory for benchmarks"
```

---

### Task 3: `results.py` — table + JSON output (TDD)

**Files:**
- Create: `benchmarks/results.py`
- Create: `tests/benchmarks/test_results.py`

- [ ] **Step 1: Write failing tests**

`tests/benchmarks/test_results.py`:

```python
import json
from pathlib import Path

import pytest

from benchmarks.results import print_table, write_json


@pytest.fixture
def sample_results():
    return {
        "size": {
            "joblib_bytes": 1_000_000,
            "safetensors_bytes": 500_000,
            "reduction_ratio": 0.5,
            "reduction_pct": 50.0,
        },
        "load": {
            "joblib": {"min": 1.0, "median": 1.1, "max": 1.2, "repeats": 5},
            "inference_heap": {"min": 0.5, "median": 0.55, "max": 0.6, "repeats": 5},
            "inference_mmap": {"min": 0.001, "median": 0.002, "max": 0.003, "repeats": 5},
        },
        "predict": [
            {
                "batch_size": 1_000,
                "lightfm": {"min": 0.01, "median": 0.011, "max": 0.012, "per_sec": 90909},
                "inference": {"min": 0.01, "median": 0.011, "max": 0.012, "per_sec": 90909},
                "ratio": 1.00,
            },
        ],
        "ram": {
            "platform": "darwin",
            "measurement": "rss",
            "n_workers": 4,
            "model_bytes": 500_000,
            "variant_a": {"per_worker": [400_000, 410_000, 390_000, 405_000], "total": 1_605_000},
            "variant_b": {"per_worker": [200_000, 205_000, 195_000, 202_000], "total": 802_000},
            "sharing_ratio": 2.0,
            "caveats": ["test caveat"],
        },
    }


@pytest.fixture
def sample_metadata():
    return {
        "timestamp": "2026-04-23T14:32:17Z",
        "size": "custom",
        "config": {"n_users": 10, "n_items": 20, "no_components": 4},
        "host": {"platform": "darwin", "python": "3.13", "cpu_count": 8},
        "lightfm_version": "1.20",
        "n_workers": 4,
    }


def test_write_json_roundtrip(tmp_path, sample_results, sample_metadata):
    path = tmp_path / "out.json"
    write_json(sample_results, sample_metadata, path)
    loaded = json.loads(path.read_text())
    assert loaded["metadata"]["size"] == "custom"
    assert loaded["axes"]["size"]["reduction_pct"] == 50.0
    assert loaded["axes"]["ram"]["variant_b"]["total"] == 802_000


def test_print_table_does_not_raise(capsys, sample_results, sample_metadata):
    print_table(sample_results, sample_metadata)
    captured = capsys.readouterr()
    assert "Artifact Size" in captured.out
    assert "Load Time" in captured.out
    assert "Predict Throughput" in captured.out
    assert "Multi-Process RAM" in captured.out


def test_write_json_handles_skipped_axes(tmp_path, sample_metadata):
    partial = {"size": {"joblib_bytes": 100, "safetensors_bytes": 50, "reduction_ratio": 0.5, "reduction_pct": 50.0}}
    path = tmp_path / "partial.json"
    write_json(partial, sample_metadata, path)
    loaded = json.loads(path.read_text())
    assert "size" in loaded["axes"]
    assert "load" not in loaded["axes"]


def test_print_table_handles_skipped_axes(capsys, sample_metadata):
    partial = {"size": {"joblib_bytes": 100, "safetensors_bytes": 50, "reduction_ratio": 0.5, "reduction_pct": 50.0}}
    print_table(partial, sample_metadata)
    captured = capsys.readouterr()
    assert "Artifact Size" in captured.out
    assert "Load Time" not in captured.out
```

- [ ] **Step 2: Run tests — expect failures**

Run: `uv run pytest tests/benchmarks/test_results.py -v`
Expected: FAIL — `benchmarks.results` doesn't exist.

- [ ] **Step 3: Implement `benchmarks/results.py`**

```python
"""Shared output formatting for benchmark results.

Two entry points:
- `print_table(results, metadata)` — prints a human-readable table per axis.
- `write_json(results, metadata, path)` — writes the full JSON sidecar.

`results` is a dict with axis keys ("size", "load", "predict", "ram"); any
subset is allowed (axes that were skipped or errored are simply absent).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(results: dict[str, Any], metadata: dict[str, Any], path: Path) -> None:
    """Write the full JSON payload. Missing axes are simply omitted."""
    payload = {"metadata": metadata, "axes": results}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def print_table(results: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Print a human-readable table to stdout, one section per axis present."""
    size_label = metadata.get("size", "?")
    cfg = metadata.get("config", {})
    cfg_str = (
        f"{cfg.get('n_users', '?'):,} users × {cfg.get('n_items', '?'):,} items × "
        f"{cfg.get('no_components', '?')} components"
    )

    if "size" in results:
        _print_size(results["size"], size_label, cfg_str)
    if "load" in results:
        _print_load(results["load"])
    if "predict" in results:
        _print_predict(results["predict"])
    if "ram" in results:
        _print_ram(results["ram"])


def _print_size(data: dict, size_label: str, cfg_str: str) -> None:
    print(f"\n=== Artifact Size ({size_label}: {cfg_str}) ===")
    print(f"{'Method':<30} {'Bytes':>15}   Ratio")
    j = data["joblib_bytes"]
    s = data["safetensors_bytes"]
    print(f"{'joblib.dump(LightFM)':<30} {j:>15,}   1.00x (baseline)")
    print(f"{'save_for_inference':<30} {s:>15,}   {data['reduction_ratio']:.2f}x ({data['reduction_pct']:.1f}% smaller)")


def _print_load(data: dict) -> None:
    print("\n=== Load Time (wall-clock seconds) ===")
    print(f"{'Method':<30} {'Min':>10} {'Median':>10} {'Max':>10}")
    for method, label in (
        ("joblib", "joblib.load"),
        ("inference_heap", "InferenceLightFM mmap=False"),
        ("inference_mmap", "InferenceLightFM mmap=True"),
    ):
        if method in data:
            d = data[method]
            print(f"{label:<30} {d['min']:>10.4f} {d['median']:>10.4f} {d['max']:>10.4f}")


def _print_predict(data: list[dict]) -> None:
    print("\n=== Predict Throughput (predictions/sec, median of 3 runs) ===")
    print(f"{'Batch Size':>12} {'LightFM':>18} {'InferenceLightFM':>18} {'Ratio':>8}")
    for row in data:
        print(
            f"{row['batch_size']:>12,} "
            f"{row['lightfm']['per_sec']:>18,} "
            f"{row['inference']['per_sec']:>18,} "
            f"{row['ratio']:>7.2f}x"
        )


def _print_ram(data: dict) -> None:
    if data.get("skipped"):
        print(f"\n=== Multi-Process RAM ===")
        print(f"SKIPPED: {data.get('reason', 'unknown')}")
        return
    measurement = data["measurement"].upper()
    n = data["n_workers"]
    model_mb = data["model_bytes"] / 1e6
    print(f"\n=== Multi-Process RAM (N={n} workers, {data['platform']} {measurement}) ===")
    print(f"Model size: {model_mb:,.1f} MB")
    a = data["variant_a"]["total"]
    b = data["variant_b"]["total"]
    print(f"Variant A (joblib.load + Pool):  total {measurement} = {a/1e6:,.1f} MB ({a/data['model_bytes']:.2f}x model)")
    print(f"Variant B (mmap + Pool):         total {measurement} = {b/1e6:,.1f} MB ({b/data['model_bytes']:.2f}x model)")
    print(f"Sharing ratio (A/B):             {data['sharing_ratio']:.2f}x")
    if data.get("caveats"):
        print("Caveats:")
        for c in data["caveats"]:
            print(f"  - {c}")
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/benchmarks/test_results.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: 114 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/results.py tests/benchmarks/test_results.py
git commit -m "Add benchmark results table printer and JSON writer"
```

---

### Task 4: `bench_size.py` — Axis A (TDD)

**Files:**
- Create: `benchmarks/bench_size.py`
- Create: `tests/benchmarks/test_bench_size.py`

- [ ] **Step 1: Failing test**

`tests/benchmarks/test_bench_size.py`:

```python
from pathlib import Path

from benchmarks.bench_size import run
from benchmarks.models import make_fitted_model


def test_bench_size_returns_expected_keys(tmp_path):
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    result = run(model, tmp_path)
    assert set(result) == {"joblib_bytes", "safetensors_bytes", "reduction_ratio", "reduction_pct"}
    assert result["joblib_bytes"] > 0
    assert result["safetensors_bytes"] > 0


def test_bench_size_leaves_artifacts_on_disk(tmp_path):
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    run(model, tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "model.safetensors").exists()


def test_bench_size_safetensors_is_smaller(tmp_path):
    """save_for_inference drops gradient/momentum state, so the artifact
    should be notably smaller than joblib.dump of the full model."""
    model = make_fitted_model(n_users=200, n_items=500, no_components=16)
    result = run(model, tmp_path)
    assert result["safetensors_bytes"] < result["joblib_bytes"]
    assert 0.0 < result["reduction_ratio"] < 1.0
    assert result["reduction_pct"] > 0.0
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/benchmarks/test_bench_size.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `benchmarks/bench_size.py`**

```python
"""Axis A: artifact size comparison.

Writes both joblib and safetensors artifacts to disk and returns their byte
counts. Leaves the files on disk so subsequent axes (load, ram) can reuse
them without redumping.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from lightfm import LightFM


def run(model: LightFM, artifact_dir: Path) -> dict:
    """Dump the model two ways under `artifact_dir` and return byte counts."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = artifact_dir / "model.joblib"
    safetensors_path = artifact_dir / "model.safetensors"

    joblib.dump(model, str(joblib_path))
    model.save_for_inference(safetensors_path)

    joblib_bytes = joblib_path.stat().st_size
    safetensors_bytes = safetensors_path.stat().st_size
    ratio = safetensors_bytes / joblib_bytes if joblib_bytes else 0.0

    return {
        "joblib_bytes": joblib_bytes,
        "safetensors_bytes": safetensors_bytes,
        "reduction_ratio": ratio,
        "reduction_pct": 100.0 * (1.0 - ratio),
    }
```

- [ ] **Step 4: Run — should pass**

Run: `uv run pytest tests/benchmarks/test_bench_size.py -v`
Expected: 3 pass.

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: 117 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_size.py tests/benchmarks/test_bench_size.py
git commit -m "Add axis A (artifact size) benchmark"
```

---

### Task 5: `bench_load.py` — Axis B (TDD)

**Files:**
- Create: `benchmarks/bench_load.py`
- Create: `tests/benchmarks/test_bench_load.py`

- [ ] **Step 1: Failing test**

`tests/benchmarks/test_bench_load.py`:

```python
from pathlib import Path

import joblib

from benchmarks.bench_load import run
from benchmarks.bench_size import run as bench_size_run
from benchmarks.models import make_fitted_model


def test_bench_load_returns_three_methods(tmp_path):
    # Prepare artifacts
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    bench_size_run(model, tmp_path)

    result = run(tmp_path, repeats=2)
    assert set(result) == {"joblib", "inference_heap", "inference_mmap"}
    for method in ("joblib", "inference_heap", "inference_mmap"):
        d = result[method]
        assert set(d) == {"min", "median", "max", "repeats"}
        assert d["repeats"] == 2
        assert d["min"] <= d["median"] <= d["max"]
        assert d["min"] >= 0
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/benchmarks/test_bench_load.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `benchmarks/bench_load.py`**

```python
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
```

- [ ] **Step 4: Run — pass**

Run: `uv run pytest tests/benchmarks/test_bench_load.py -v`
Expected: 1 pass.

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: 118 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_load.py tests/benchmarks/test_bench_load.py
git commit -m "Add axis B (load time) benchmark"
```

---

### Task 6: `bench_predict.py` — Axis C (TDD)

**Files:**
- Create: `benchmarks/bench_predict.py`
- Create: `tests/benchmarks/test_bench_predict.py`

- [ ] **Step 1: Failing test**

`tests/benchmarks/test_bench_predict.py`:

```python
from benchmarks.bench_predict import run
from benchmarks.bench_size import run as bench_size_run
from benchmarks.models import make_fitted_model


def test_bench_predict_returns_list_of_per_batch_dicts(tmp_path):
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    bench_size_run(model, tmp_path)
    results = run(tmp_path, repeats=2, batch_sizes=[10, 20])
    assert isinstance(results, list)
    assert len(results) == 2
    for row in results:
        assert set(row) == {"batch_size", "lightfm", "inference", "ratio"}
        assert "per_sec" in row["lightfm"]
        assert "per_sec" in row["inference"]


def test_bench_predict_reports_nonzero_throughput(tmp_path):
    """Throughput numbers are positive — soft sanity rather than strict
    scaling (CI timing noise makes `large >= small` flaky at small sizes)."""
    model = make_fitted_model(n_users=200, n_items=500, no_components=8)
    bench_size_run(model, tmp_path)
    results = run(tmp_path, repeats=2, batch_sizes=[50, 500])
    for row in results:
        assert row["inference"]["per_sec"] > 0
        assert row["lightfm"]["per_sec"] > 0
        assert row["ratio"] > 0
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/benchmarks/test_bench_predict.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `benchmarks/bench_predict.py`**

```python
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
```

- [ ] **Step 4: Run — pass**

Run: `uv run pytest tests/benchmarks/test_bench_predict.py -v`
Expected: 2 pass.

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: 120 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_predict.py tests/benchmarks/test_bench_predict.py
git commit -m "Add axis C (predict throughput) benchmark"
```

---

### Task 7: `bench_ram.py` — Axis D (TDD, careful fork-global pattern)

**Files:**
- Create: `benchmarks/bench_ram.py`
- Create: `tests/benchmarks/test_bench_ram.py`

**Critical:** Variant A relies on module-level globals inherited via fork. Do NOT refactor to pass the model through `Pool.map`'s arg queue — that would pickle the model per worker and measure the wrong thing. See spec Section "bench_ram.py" for the rationale.

- [ ] **Step 1: Failing test**

`tests/benchmarks/test_bench_ram.py`:

```python
import sys

import pytest

from benchmarks.bench_ram import run
from benchmarks.bench_size import run as bench_size_run
from benchmarks.models import make_fitted_model


@pytest.mark.skipif(sys.platform == "win32", reason="fork not available on Windows")
def test_bench_ram_returns_expected_structure(tmp_path):
    model = make_fitted_model(n_users=100, n_items=500, no_components=16)
    bench_size_run(model, tmp_path)

    result = run(tmp_path, n_workers=2)
    assert "platform" in result
    assert result["measurement"] in ("pss", "rss")
    assert result["n_workers"] == 2
    assert result["model_bytes"] > 0
    for variant in ("variant_a", "variant_b"):
        assert variant in result
        assert len(result[variant]["per_worker"]) == 2
        assert result[variant]["total"] == sum(result[variant]["per_worker"])
    assert "sharing_ratio" in result


def test_bench_ram_skips_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    bench_size_run(model, tmp_path)
    result = run(tmp_path, n_workers=2)
    assert result["skipped"] is True
    assert "fork" in result["reason"].lower() or "windows" in result["reason"].lower()
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/benchmarks/test_bench_ram.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `benchmarks/bench_ram.py`**

```python
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
```

- [ ] **Step 4: Run — pass**

Run: `uv run pytest tests/benchmarks/test_bench_ram.py -v`
Expected: 2 pass on macOS/Linux (Windows skip marker).

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: 122 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_ram.py tests/benchmarks/test_bench_ram.py
git commit -m "Add axis D (multi-process RAM) benchmark"
```

---

### Task 8: `__main__.py` — CLI wiring

**Files:**
- Create: `benchmarks/__main__.py`

No new tests for the CLI — the smoke test in Task 9 covers it end-to-end.

- [ ] **Step 1: Implement `benchmarks/__main__.py`**

```python
"""Benchmark suite CLI entry point.

Usage:
    uv run python -m benchmarks --size {tiny,medium,large,custom} [options]

Runs axes in order (size, load, predict, ram). If a later axis needs
artifacts produced by size but --skip-axis=size was passed, size runs
anyway with a one-line notice. Writes a JSON sidecar under the output
directory and prints a human-readable table to stdout.
"""
from __future__ import annotations

import argparse
import json
import platform as platform_mod
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import bench_load, bench_predict, bench_ram, bench_size, results
from benchmarks.models import SIZE_PRESETS, make_fitted_model
from lightfm import __version__ as _lightfm_version

ALL_AXES = ("size", "load", "predict", "ram")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Inference artifact benchmark suite (size / load / predict / RAM).",
    )
    p.add_argument("--size", choices=("tiny", "medium", "large", "custom"), default="tiny")
    p.add_argument("--n-users", type=int, help="Override (required with --size custom)")
    p.add_argument("--n-items", type=int, help="Override")
    p.add_argument("--no-components", type=int, help="Override")
    p.add_argument("--n-workers", type=int, default=4, help="Workers for axis D (default 4)")
    p.add_argument("--repeats-load", type=int, default=5)
    p.add_argument("--repeats-predict", type=int, default=3)
    p.add_argument(
        "--skip-axis",
        default="",
        help=f"Comma-separated axis names to skip. Choices: {','.join(ALL_AXES)}",
    )
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--label", default="", help="Optional label appended to JSON filename")
    return p.parse_args(argv)


def _resolve_config(args: argparse.Namespace) -> dict:
    if args.size == "custom":
        if args.n_users is None or args.n_items is None or args.no_components is None:
            raise SystemExit("--size custom requires --n-users, --n-items, --no-components")
        return {"n_users": args.n_users, "n_items": args.n_items, "no_components": args.no_components}
    cfg = dict(SIZE_PRESETS[args.size])
    if args.n_users is not None:
        cfg["n_users"] = args.n_users
    if args.n_items is not None:
        cfg["n_items"] = args.n_items
    if args.no_components is not None:
        cfg["no_components"] = args.no_components
    return cfg


def _build_metadata(args: argparse.Namespace, cfg: dict) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size": args.size,
        "config": cfg,
        "host": {
            "platform": sys.platform,
            "python": platform_mod.python_version(),
            "cpu_count": (__import__("os")).cpu_count(),
        },
        "lightfm_version": _lightfm_version,
        "n_workers": args.n_workers,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = _resolve_config(args)
    except SystemExit as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    skip = {s.strip() for s in args.skip_axis.split(",") if s.strip()}
    unknown = skip - set(ALL_AXES)
    if unknown:
        print(f"error: unknown --skip-axis values: {sorted(unknown)}", file=sys.stderr)
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / f".artifacts-{int(time.time())}"
    artifact_dir.mkdir()

    print(f"Benchmarking size={args.size} ({cfg['n_users']:,} users × {cfg['n_items']:,} items × {cfg['no_components']} components)")
    print(f"Artifacts: {artifact_dir}")

    metadata = _build_metadata(args, cfg)
    all_results: dict = {}

    needs_artifacts = bool({"load", "predict", "ram"} - skip)
    run_size = "size" not in skip or needs_artifacts

    # --- Axis A: size ---
    if run_size:
        try:
            model = make_fitted_model(**cfg)
            if "size" in skip:
                print("note: running bench_size anyway (later axes need its artifacts)")
            size_result = bench_size.run(model, artifact_dir)
            if "size" not in skip:
                all_results["size"] = size_result
            del model
        except Exception:
            traceback.print_exc()
            all_results["size"] = {"error": "bench_size failed"}

    # --- Axis B: load ---
    if "load" not in skip:
        try:
            all_results["load"] = bench_load.run(artifact_dir, repeats=args.repeats_load)
        except Exception:
            traceback.print_exc()
            all_results["load"] = {"error": "bench_load failed"}

    # --- Axis C: predict ---
    if "predict" not in skip:
        try:
            all_results["predict"] = bench_predict.run(artifact_dir, repeats=args.repeats_predict)
        except Exception:
            traceback.print_exc()
            all_results["predict"] = {"error": "bench_predict failed"}

    # --- Axis D: ram ---
    if "ram" not in skip:
        try:
            all_results["ram"] = bench_ram.run(artifact_dir, n_workers=args.n_workers)
        except Exception:
            traceback.print_exc()
            all_results["ram"] = {"error": "bench_ram failed"}

    # --- Output ---
    results.print_table(all_results, metadata)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = f"-{args.label}" if args.label else ""
    json_path = output_dir / f"{stamp}-{args.size}{suffix}.json"
    results.write_json(all_results, metadata, json_path)
    print(f"\nJSON written to: {json_path}")

    had_error = any(isinstance(v, dict) and v.get("error") for v in all_results.values())
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test via CLI**

Run: `uv run python -m benchmarks --size tiny --n-workers 2 --repeats-load 2 --repeats-predict 2`
Expected: prints table with four sections; writes `benchmarks/results/YYYY-MM-DD-tiny.json`; exit 0.

Clean up the temp artifact directory afterward:

```bash
rm -rf benchmarks/results/.artifacts-* benchmarks/results/*-tiny.json
```

- [ ] **Step 3: Full suite**

Run: `uv run pytest tests/ -q`
Expected: 122 passed, 1 skipped.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/__main__.py
git commit -m "Add benchmarks CLI"
```

---

### Task 9: Smoke test under `tests/benchmarks/`

**Files:**
- Create: `tests/benchmarks/test_smoke.py`

- [ ] **Step 1: Write the test**

```python
"""Smoke test for the benchmark CLI.

Marked `slow` — excluded from default runs via `pytest -m "not slow"`.
Runs the full four-axis pipeline at --size tiny and asserts a valid JSON
appears. Guards the benchmark machinery itself (no numeric assertions).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_benchmarks_cli_smoke(tmp_path):
    output_dir = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmarks",
            "--size", "tiny",
            "--n-workers", "2",
            "--repeats-load", "2",
            "--repeats-predict", "2",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

    # The CLI writes exactly one sidecar JSON matching the size.
    json_files = list(output_dir.glob("*-tiny.json"))
    assert len(json_files) == 1, f"expected one JSON, got {json_files}"
    payload = json.loads(json_files[0].read_text())

    assert payload["metadata"]["size"] == "tiny"
    assert "size" in payload["axes"]
    assert "load" in payload["axes"]
    assert "predict" in payload["axes"]
    assert "ram" in payload["axes"]
    # Basic numeric sanity
    assert payload["axes"]["size"]["joblib_bytes"] > 0
    assert payload["axes"]["size"]["safetensors_bytes"] > 0
    # Axis D emits caveats — spec calls this out explicitly
    assert isinstance(payload["axes"]["ram"].get("caveats"), list)
    assert len(payload["axes"]["ram"]["caveats"]) > 0
```

- [ ] **Step 2: Run (slow marker excluded by default — run explicitly)**

Run: `uv run pytest tests/benchmarks/test_smoke.py -v -m slow`
Expected: 1 pass. Takes a few seconds.

- [ ] **Step 3: Confirm default runs skip it**

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: 122 passed, 1 skipped (unchanged — the smoke test was deselected).

- [ ] **Step 4: Commit**

```bash
git add tests/benchmarks/test_smoke.py
git commit -m "Add benchmark CLI smoke test"
```

---

### Task 10: Run the suite and commit `sample-medium.json`

**Files:**
- Create: `benchmarks/results/sample-medium.json` (committed)

- [ ] **Step 1: Run the full medium benchmark**

Run: `uv run python -m benchmarks --size medium --label sample`
Expected: prints table with real numbers. Takes 1-2 minutes. Produces `benchmarks/results/YYYY-MM-DD-medium-sample.json`.

This allocates ~2GB during model construction and ~670MB on-disk in a temp artifact dir. If the host can't spare the memory, run `--size tiny --label sample` instead and commit that.

- [ ] **Step 2: Rename to the canonical committed filename**

```bash
mv benchmarks/results/*-medium-sample.json benchmarks/results/sample-medium.json
```

(Or `sample-tiny.json` if step 1 used tiny.)

- [ ] **Step 3: Clean the temp artifact directory**

```bash
rm -rf benchmarks/results/.artifacts-*
```

- [ ] **Step 4: Eyeball the JSON for sanity**

- `metadata.size == "medium"` (or `tiny`)
- `axes.size.reduction_pct` > 0 (should be substantial — 40-50% for `medium`)
- `axes.ram.sharing_ratio` > 1.0 on macOS; expected much larger on Linux

- [ ] **Step 5: Commit**

```bash
git add benchmarks/results/sample-medium.json
git commit -m "Add sample-medium.json benchmark results"
```

Alternative commit message if tiny was used: `"Add sample-tiny.json benchmark results (medium infeasible on this host)"`.

---

## Chunk 1 exit criteria

- `uv run pytest tests/ -q` passes (122 passed + 1 skipped), and `-m slow` adds the smoke test (123 passed).
- `uv run python -m benchmarks --size tiny` completes in under 10s and writes a valid JSON.
- `uv run python -m benchmarks --size medium --label sample` produces a sidecar JSON matching the schema.
- `benchmarks/results/sample-medium.json` committed with real numbers from the dev host.
- Branch has 10 new commits (Tasks 1-10, one each).
- Variant A's `_MODEL_A` module-level global pattern is preserved — verify no refactor accidentally passed the model through `Pool.map`'s arg queue.
