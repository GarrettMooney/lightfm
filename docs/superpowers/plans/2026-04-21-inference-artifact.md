# Inference Artifact Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a slim inference artifact format (`safetensors`) and `InferenceLightFM` class to cut d152's production artifact from ~60GB to ~30GB and its runtime RAM from ~8× to ~1× model size.

**Architecture:** New `lightfm/inference/` subpackage with an `_artifact.py` module (single point of on-disk format), a `_predict.py` module (free functions shared by the training and inference classes), an `InferenceLightFM` class backed by memory-mapped safetensors tensors, a `LightFM.save_for_inference(path)` method, and a `convert_joblib_to_inference(src, dst)` one-shot migrator. Training-side code stays untouched except for extracting the predict body into `_predict.py` (behavior-preserving refactor).

**Tech Stack:** Python 3.8+, numpy, scipy.sparse, Cython (existing `predict_lightfm` / `predict_ranks` unchanged), safetensors (new dep, Rust-backed), fsspec (for `gs://` paths in converter), joblib (only for converter input).

**Spec:** `docs/superpowers/specs/2026-04-21-inference-artifact-design.md`

**Branch:** `feature/inference-artifact` (already created)

---

## Notes for the implementer

- **TDD everywhere except the behavior-preserving refactor.** For new behavior (artifact save/load, InferenceLightFM, converter, save_for_inference) write failing tests first. For the `_predict_impl` extraction (Chunk 1 Task 4), rely on the existing `tests/` suite — the contract is "zero regressions," measured by running tests pre- and post-refactor.
- **Commit after every task.** Each task is a green-to-green transition. `pytest tests/` must pass at the end of every task.
- **The `FastLightFM` Cython struct requires 12+ arrays** including gradient/momentum memoryviews. For inference we never read those during predict, but the constructor insists on them. **Solution:** allocate 1-element float32 "dummy" arrays and pass them in. Cheap, explicit, Cython-compatible. Helper lives in `_predict.py`.
- **Cython `predict_lightfm` and `predict_ranks` themselves are untouched.** All changes are in Python.
- **Keep `LightFM.predict` behavior bit-identical.** Every line of input normalization currently in `lightfm.py:830-860` must end up in the shared `_predict_impl` path. The extracted function is called from both `LightFM.predict` and `InferenceLightFM.predict` — missing any coercion there silently breaks `InferenceLightFM`.
- **`preload` / `madvise` are not in v1.** Do not add them "for completeness." Listed in v2 hooks.
- **Run tests with `pytest tests/ -x`** (stop on first failure) during development. Use `pytest tests/ -q` for full runs before commits.

---

## Chunk 1: Artifact format + predict extraction

Foundation chunk. By the end, the `_artifact.save` / `_artifact.load` round-trip works, and `LightFM.predict` has been refactored to delegate to a shared `_predict_impl` helper — with the existing test suite proving zero behavior change.

### Task 1: Add dependency and subpackage skeleton

**Files:**
- Modify: `pyproject.toml` (add `safetensors>=0.4` to `dependencies`)
- Create: `lightfm/inference/__init__.py` (empty stub for now)
- Create: `tests/inference/__init__.py` (empty)

- [ ] **Step 1: Add safetensors to dependencies and register new subpackage for distribution**

Edit `pyproject.toml`:

1. Add `"safetensors>=0.4"` to the `dependencies` list:

   ```toml
   dependencies = [
       "numpy>=1.17.0",
       "scipy>=0.17.0",
       "requests",
       "scikit-learn",
       "safetensors>=0.4",
   ]
   ```

2. Extend `[tool.setuptools] packages` so wheel/sdist builds ship the new subpackage (editable installs would pick it up by filesystem walk, but release builds won't):

   ```toml
   [tool.setuptools]
   packages = ["lightfm", "lightfm.datasets", "lightfm.inference"]
   ```

- [ ] **Step 2: Install the new dep**

Run: `uv pip install -e .`
Expected: installs `safetensors` wheel, no errors.

- [ ] **Step 3: Create subpackage directories and empty `__init__.py` files**

```bash
mkdir -p lightfm/inference tests/inference
touch lightfm/inference/__init__.py tests/inference/__init__.py
```

- [ ] **Step 4: Smoke-test the import**

Run: `uv run python -c "import lightfm.inference; import safetensors.numpy; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Run existing test suite — must still pass**

Run: `uv run pytest tests/ -q`
Expected: all existing tests pass. New dep did not break anything.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml lightfm/inference/__init__.py tests/inference/__init__.py
git commit -m "Add safetensors dep and lightfm.inference subpackage skeleton"
```

---

### Task 2: Implement `_artifact.save` / `_artifact.load` round-trip (TDD)

**Files:**
- Create: `lightfm/inference/_artifact.py`
- Create: `tests/inference/test_artifact.py`

This task gets the happy path working: save 4 float32 arrays + metadata → load them back. Error paths come in Task 3.

- [ ] **Step 1: Write the failing round-trip test**

`tests/inference/test_artifact.py`:

```python
import numpy as np
import pytest

from lightfm.inference import _artifact


def _tiny_arrays():
    rng = np.random.default_rng(42)
    return {
        "item_embeddings": rng.standard_normal((50, 8), dtype=np.float32),
        "user_embeddings": rng.standard_normal((20, 8), dtype=np.float32),
        "item_biases": np.zeros(50, dtype=np.float32),
        "user_biases": np.zeros(20, dtype=np.float32),
    }


def _tiny_metadata():
    return {
        "no_components": 8,
        "loss": "warp",
        "learning_schedule": "adagrad",
    }


def test_save_load_roundtrip(tmp_path):
    arrays = _tiny_arrays()
    metadata = _tiny_metadata()
    path = tmp_path / "model.safetensors"

    _artifact.save(path, arrays=arrays, metadata=metadata)
    loaded_arrays, loaded_metadata = _artifact.load(path, mmap=False)

    for key in arrays:
        np.testing.assert_array_equal(loaded_arrays[key], arrays[key])
    assert loaded_metadata["no_components"] == 8
    assert loaded_metadata["loss"] == "warp"
    assert loaded_metadata["learning_schedule"] == "adagrad"
    assert loaded_metadata["format_version"] == "1"
    assert loaded_metadata["embeddings_dtype"] == "float32"


def test_mmap_returns_views(tmp_path):
    arrays = _tiny_arrays()
    path = tmp_path / "model.safetensors"
    _artifact.save(path, arrays=arrays, metadata=_tiny_metadata())

    loaded_arrays, _ = _artifact.load(path, mmap=True)
    # mmap'd arrays have their base set to a memory-mapped object; heap-loaded
    # arrays have base=None (or a numpy-internal owner).
    assert loaded_arrays["item_embeddings"].base is not None
    np.testing.assert_array_equal(loaded_arrays["item_embeddings"], arrays["item_embeddings"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/inference/test_artifact.py -v`
Expected: FAIL with `ModuleNotFoundError` or similar — `_artifact` doesn't exist yet.

- [ ] **Step 3: Implement `_artifact.py`**

`lightfm/inference/_artifact.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from lightfm.version import __version__ as _lightfm_version

FORMAT_VERSION = "1"
SUPPORTED_VERSIONS = frozenset({"1"})

REQUIRED_TENSORS = (
    "item_embeddings",
    "user_embeddings",
    "item_biases",
    "user_biases",
)


class ArtifactLoadError(Exception):
    """Raised when a safetensors artifact cannot be loaded or validated."""


def save(
    path: str | Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Write the inference artifact to `path`.

    `arrays` must contain all four required tensors (float32, C-contiguous).
    `metadata` supplies no_components, loss, learning_schedule. Other fields
    (format_version, lightfm_version, embeddings_dtype, created_at) are added
    by this function.
    """
    path = Path(path)

    missing = [name for name in REQUIRED_TENSORS if name not in arrays]
    if missing:
        raise ValueError(f"save() missing required tensors: {missing}")

    # Lock in C-contiguous float32 for cache-line-friendly mmap reads.
    contiguous = {
        name: np.ascontiguousarray(arr, dtype=np.float32) for name, arr in arrays.items()
    }

    full_metadata: dict[str, str] = {
        "format_version": FORMAT_VERSION,
        "lightfm_version": _lightfm_version,
        "no_components": str(int(metadata["no_components"])),
        "loss": str(metadata["loss"]),
        "learning_schedule": str(metadata["learning_schedule"]),
        "embeddings_dtype": "float32",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_file(contiguous, str(path), metadata=full_metadata)


def load(
    path: str | Path,
    *,
    mmap: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load the inference artifact from `path`.

    Returns `(arrays, metadata)`. With `mmap=True`, arrays are views onto
    file-backed memory; pages are shared across forks. With `mmap=False`,
    arrays are heap-allocated copies.
    """
    path = Path(path)

    if not path.exists():
        raise ArtifactLoadError(f"artifact not found: {path}")

    if mmap:
        from safetensors import safe_open

        arrays: dict[str, np.ndarray] = {}
        try:
            with safe_open(str(path), framework="numpy") as f:
                metadata = dict(f.metadata() or {})
                keys = set(f.keys())
                _validate_header(metadata, path)
                _validate_tensor_keys(keys, path)
                for name in REQUIRED_TENSORS:
                    arrays[name] = f.get_tensor(name)
                _validate_shapes(arrays, metadata, path)
                _validate_dtypes(arrays, metadata, path)
        except ArtifactLoadError:
            raise
        except Exception as exc:
            raise ArtifactLoadError(f"failed to read {path}: {exc}") from exc
    else:
        try:
            arrays = load_file(str(path))
            # safetensors.numpy.load_file doesn't expose metadata; read it separately.
            from safetensors import safe_open

            with safe_open(str(path), framework="numpy") as f:
                metadata = dict(f.metadata() or {})
        except ArtifactLoadError:
            raise
        except Exception as exc:
            raise ArtifactLoadError(f"failed to read {path}: {exc}") from exc
        _validate_header(metadata, path)
        _validate_tensor_keys(set(arrays.keys()), path)
        _validate_shapes(arrays, metadata, path)
        _validate_dtypes(arrays, metadata, path)

    return arrays, metadata


def _validate_header(metadata: dict[str, str], path: Path) -> None:
    if "format_version" not in metadata:
        raise ArtifactLoadError(
            f"{path}: missing format_version — is this a lightfm artifact?"
        )
    version = metadata["format_version"]
    if version not in SUPPORTED_VERSIONS:
        raise ArtifactLoadError(
            f"{path}: artifact is version {version!r}, this lightfm "
            f"({_lightfm_version}) supports {sorted(SUPPORTED_VERSIONS)}"
        )


def _validate_tensor_keys(keys: set[str], path: Path) -> None:
    missing = [name for name in REQUIRED_TENSORS if name not in keys]
    if missing:
        raise ArtifactLoadError(f"{path}: missing required tensors: {missing}")


def _validate_shapes(
    arrays: dict[str, np.ndarray], metadata: dict[str, str], path: Path
) -> None:
    try:
        no_components = int(metadata["no_components"])
    except (KeyError, ValueError) as exc:
        raise ArtifactLoadError(f"{path}: missing or invalid no_components") from exc

    ie = arrays["item_embeddings"]
    ue = arrays["user_embeddings"]
    ib = arrays["item_biases"]
    ub = arrays["user_biases"]

    if ie.ndim != 2 or ie.shape[1] != no_components:
        raise ArtifactLoadError(
            f"{path}: item_embeddings shape {ie.shape} inconsistent with "
            f"no_components={no_components}"
        )
    if ue.ndim != 2 or ue.shape[1] != no_components:
        raise ArtifactLoadError(
            f"{path}: user_embeddings shape {ue.shape} inconsistent with "
            f"no_components={no_components}"
        )
    if ib.ndim != 1 or ib.shape[0] != ie.shape[0]:
        raise ArtifactLoadError(
            f"{path}: item_biases shape {ib.shape} does not match item_embeddings rows {ie.shape[0]}"
        )
    if ub.ndim != 1 or ub.shape[0] != ue.shape[0]:
        raise ArtifactLoadError(
            f"{path}: user_biases shape {ub.shape} does not match user_embeddings rows {ue.shape[0]}"
        )


def _validate_dtypes(
    arrays: dict[str, np.ndarray], metadata: dict[str, str], path: Path
) -> None:
    declared = metadata.get("embeddings_dtype", "float32")
    if declared != "float32":
        raise ArtifactLoadError(
            f"{path}: embeddings_dtype={declared!r} not supported in v1 (expected 'float32')"
        )
    for name in REQUIRED_TENSORS:
        actual = arrays[name].dtype
        if actual != np.float32:
            raise ArtifactLoadError(
                f"{path}: tensor {name!r} has dtype {actual}, expected float32"
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/inference/test_artifact.py -v`
Expected: both tests pass.

- [ ] **Step 5: Run the full test suite — nothing else should have broken**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lightfm/inference/_artifact.py tests/inference/test_artifact.py
git commit -m "Add _artifact.save / _artifact.load round-trip"
```

---

### Task 3: Implement `_artifact` error paths (TDD)

**Files:**
- Modify: `tests/inference/test_artifact.py` (add error tests)
- Modify: `lightfm/inference/_artifact.py` (if any error branches missed)

The error branches were written in Task 2; this task locks them down with tests. If any branch isn't actually triggered by its test, fix the implementation.

- [ ] **Step 1: Write failing tests for each error case**

Append to `tests/inference/test_artifact.py`:

```python
from pathlib import Path


def test_missing_file(tmp_path):
    with pytest.raises(_artifact.ArtifactLoadError, match="artifact not found"):
        _artifact.load(tmp_path / "does-not-exist.safetensors")


def test_missing_format_version(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    save_file(_tiny_arrays(), str(path), metadata={"loss": "warp"})
    with pytest.raises(_artifact.ArtifactLoadError, match="missing format_version"):
        _artifact.load(path, mmap=False)


def test_unknown_format_version(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    save_file(_tiny_arrays(), str(path), metadata={
        "format_version": "999",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="version '999'"):
        _artifact.load(path, mmap=False)


def test_missing_required_tensor(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    arrays = _tiny_arrays()
    del arrays["item_biases"]
    save_file(arrays, str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="missing required tensors.*item_biases"):
        _artifact.load(path, mmap=False)


def test_shape_inconsistency(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    arrays = _tiny_arrays()
    # item_biases length != item_embeddings rows
    arrays["item_biases"] = np.zeros(7, dtype=np.float32)
    save_file(arrays, str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="item_biases.*does not match"):
        _artifact.load(path, mmap=False)


def test_dtype_mismatch(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    arrays = _tiny_arrays()
    arrays["item_embeddings"] = arrays["item_embeddings"].astype(np.float64)
    save_file(arrays, str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="dtype float64.*expected float32"):
        _artifact.load(path, mmap=False)


def test_unsupported_declared_dtype(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    save_file(_tiny_arrays(), str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float16",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="embeddings_dtype='float16'"):
        _artifact.load(path, mmap=False)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/inference/test_artifact.py -v`
Expected: all pass (the error branches exist from Task 2). If any fail, fix either the test expectation or the matching error message in `_artifact.py`.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/inference/test_artifact.py lightfm/inference/_artifact.py
git commit -m "Cover _artifact error paths with tests"
```

---

### Task 4: Extract `_predict_impl` and `_predict_rank_impl` (behavior-preserving refactor)

**Files:**
- Create: `lightfm/inference/_predict.py`
- Modify: `lightfm/lightfm.py` (lines ~767-883 for `predict`, ~895-1000 for `predict_rank`)

This is a **refactor**, not new behavior. No new tests. The existing test suite is the contract. Every line of input normalization, validation, and feature-matrix construction in `LightFM.predict` / `LightFM.predict_rank` must move into the extracted helpers or be called from them. `LightFM.predict` ends up as a thin delegate.

The `FastLightFM` Cython struct needs 12+ arrays. For training that's all the real arrays. For inference we'll pass dummy 1-element arrays for the 8 gradient/momentum slots. Task 4 introduces a helper `_build_fast_lightfm_for_predict` that accepts either real arrays (training class passes all) or `None` for gradients/momentum (inference-side usage in Chunk 2 passes `None`).

- [ ] **Step 1: Baseline the existing test suite**

Run: `uv run pytest tests/ -q > /tmp/tests_pre_refactor.txt 2>&1; tail -3 /tmp/tests_pre_refactor.txt`
Expected: count of passing tests. Record the number (e.g., "143 passed in 47.32s") — this is the baseline we must match after refactor.

- [ ] **Step 2: Create `_predict.py`**

`lightfm/inference/_predict.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from lightfm._lightfm_fast import (
    CSRMatrix,
    FastLightFM,
    predict_lightfm,
    predict_ranks,
)

CYTHON_DTYPE = np.float32


@dataclass
class _InferenceData:
    """Minimal state needed to score. Populated by LightFM (with full training
    arrays) or InferenceLightFM (with dummy gradients/momentum).
    """
    item_embeddings: np.ndarray
    user_embeddings: np.ndarray
    item_biases: np.ndarray
    user_biases: np.ndarray
    no_components: int
    learning_schedule: str  # "adagrad" or "adadelta"
    # Training-side supplies real arrays; inference supplies None (we fill with dummies).
    item_embedding_gradients: np.ndarray | None = None
    item_embedding_momentum: np.ndarray | None = None
    item_bias_gradients: np.ndarray | None = None
    item_bias_momentum: np.ndarray | None = None
    user_embedding_gradients: np.ndarray | None = None
    user_embedding_momentum: np.ndarray | None = None
    user_bias_gradients: np.ndarray | None = None
    user_bias_momentum: np.ndarray | None = None
    # Training-side hyperparams used by FastLightFM. Unused during predict but
    # required by the struct constructor.
    learning_rate: float = 0.05
    rho: float = 0.95
    epsilon: float = 1e-6
    max_sampled: int = 10


def _dummy_like_2d(shape: tuple[int, int]) -> np.ndarray:
    # A single-row allocation satisfies the [:, ::1] memoryview type without
    # carrying per-feature storage. predict_lightfm never reads these.
    return np.zeros((1, shape[1] if shape[1] > 0 else 1), dtype=np.float32)


def _dummy_like_1d() -> np.ndarray:
    return np.zeros(1, dtype=np.float32)


def _build_fast_lightfm(data: _InferenceData) -> FastLightFM:
    ie_grad = data.item_embedding_gradients if data.item_embedding_gradients is not None else _dummy_like_2d(data.item_embeddings.shape)
    ie_mom = data.item_embedding_momentum if data.item_embedding_momentum is not None else _dummy_like_2d(data.item_embeddings.shape)
    ib_grad = data.item_bias_gradients if data.item_bias_gradients is not None else _dummy_like_1d()
    ib_mom = data.item_bias_momentum if data.item_bias_momentum is not None else _dummy_like_1d()
    ue_grad = data.user_embedding_gradients if data.user_embedding_gradients is not None else _dummy_like_2d(data.user_embeddings.shape)
    ue_mom = data.user_embedding_momentum if data.user_embedding_momentum is not None else _dummy_like_2d(data.user_embeddings.shape)
    ub_grad = data.user_bias_gradients if data.user_bias_gradients is not None else _dummy_like_1d()
    ub_mom = data.user_bias_momentum if data.user_bias_momentum is not None else _dummy_like_1d()

    return FastLightFM(
        data.item_embeddings,
        ie_grad,
        ie_mom,
        data.item_biases,
        ib_grad,
        ib_mom,
        data.user_embeddings,
        ue_grad,
        ue_mom,
        data.user_biases,
        ub_grad,
        ub_mom,
        data.no_components,
        int(data.learning_schedule == "adadelta"),
        data.learning_rate,
        data.rho,
        data.epsilon,
        data.max_sampled,
    )


def _to_cython_dtype(mat: sp.csr_matrix) -> sp.csr_matrix:
    if mat.dtype != CYTHON_DTYPE:
        return mat.astype(CYTHON_DTYPE)
    return mat


def _construct_feature_matrices(
    n_users: int,
    n_items: int,
    user_features: sp.csr_matrix | None,
    item_features: sp.csr_matrix | None,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    if user_features is None:
        user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format="csr")
    else:
        user_features = user_features.tocsr()

    if item_features is None:
        item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format="csr")
    else:
        item_features = item_features.tocsr()

    if n_users > user_features.shape[0]:
        raise Exception(
            "Number of user feature rows does not equal the number of users"
        )
    if n_items > item_features.shape[0]:
        raise Exception(
            "Number of item feature rows does not equal the number of items"
        )

    if not user_embeddings.shape[0] >= user_features.shape[1]:
        raise ValueError(
            "The user feature matrix specifies more features than there are "
            "estimated feature embeddings: {} vs {}.".format(
                user_embeddings.shape[0], user_features.shape[1]
            )
        )
    if not item_embeddings.shape[0] >= item_features.shape[1]:
        raise ValueError(
            "The item feature matrix specifies more features than there are "
            "estimated feature embeddings: {} vs {}.".format(
                item_embeddings.shape[0], item_features.shape[1]
            )
        )

    return _to_cython_dtype(user_features), _to_cython_dtype(item_features)


def _validate_predict_inputs(
    user_ids: int | NDArray[np.int32] | list | tuple,
    item_ids: NDArray[np.int32] | list | tuple,
    num_threads: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    if isinstance(user_ids, int):
        user_ids = np.repeat(np.int32(user_ids), len(item_ids))

    if isinstance(user_ids, (list, tuple)):
        user_ids = np.array(user_ids, dtype=np.int32)

    if isinstance(item_ids, (list, tuple)):
        item_ids = np.array(item_ids, dtype=np.int32)

    if len(user_ids) != len(item_ids):
        raise ValueError(
            f"Expected the number of user IDs ({len(user_ids)}) to equal the number"
            f" of item IDs ({len(item_ids)})"
        )

    if user_ids.dtype != np.int32:
        user_ids = user_ids.astype(np.int32)
    if item_ids.dtype != np.int32:
        item_ids = item_ids.astype(np.int32)

    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    if user_ids.min() < 0 or item_ids.min() < 0:
        raise ValueError(
            "User or item ids cannot be negative. "
            "Check your inputs for negative numbers "
            "or very large numbers that can overflow."
        )

    return user_ids, item_ids


def _predict_impl(
    data: _InferenceData,
    user_ids,
    item_ids,
    user_features: sp.csr_matrix | None = None,
    item_features: sp.csr_matrix | None = None,
    num_threads: int = 1,
) -> NDArray[np.float32]:
    user_ids, item_ids = _validate_predict_inputs(user_ids, item_ids, num_threads)

    n_users = int(user_ids.max()) + 1
    n_items = int(item_ids.max()) + 1

    user_features, item_features = _construct_feature_matrices(
        n_users,
        n_items,
        user_features,
        item_features,
        data.user_embeddings,
        data.item_embeddings,
    )

    lightfm_data = _build_fast_lightfm(data)
    predictions = np.empty(len(user_ids), dtype=np.float32)

    predict_lightfm(
        CSRMatrix(item_features),
        CSRMatrix(user_features),
        user_ids,
        item_ids,
        predictions,
        lightfm_data,
        num_threads,
    )

    return predictions


def _check_test_train_intersections(test_mat, train_mat) -> None:
    if train_mat is not None:
        n_intersections = test_mat.multiply(train_mat).nnz
        if n_intersections:
            raise ValueError(
                "Test interactions matrix and train interactions "
                "matrix share %d interactions. This will cause "
                "incorrect evaluation, check your data split." % n_intersections
            )


def _predict_rank_impl(
    data: _InferenceData,
    test_interactions: sp.csr_matrix,
    train_interactions: sp.csr_matrix | None = None,
    user_features: sp.csr_matrix | None = None,
    item_features: sp.csr_matrix | None = None,
    num_threads: int = 1,
    check_intersections: bool = True,
) -> sp.csr_matrix:
    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    if check_intersections:
        _check_test_train_intersections(test_interactions, train_interactions)

    n_users, n_items = test_interactions.shape

    user_features, item_features = _construct_feature_matrices(
        n_users,
        n_items,
        user_features,
        item_features,
        data.user_embeddings,
        data.item_embeddings,
    )

    if not item_features.shape[1] == data.item_embeddings.shape[0]:
        raise ValueError("Incorrect number of features in item_features")
    if not user_features.shape[1] == data.user_embeddings.shape[0]:
        raise ValueError("Incorrect number of features in user_features")

    test_interactions = _to_cython_dtype(test_interactions.tocsr())

    if train_interactions is None:
        train_interactions = sp.csr_matrix((n_users, n_items), dtype=CYTHON_DTYPE)
    else:
        train_interactions = _to_cython_dtype(train_interactions.tocsr())

    ranks = sp.csr_matrix(
        (
            np.zeros_like(test_interactions.data),
            test_interactions.indices,
            test_interactions.indptr,
        ),
        shape=test_interactions.shape,
    )

    lightfm_data = _build_fast_lightfm(data)

    predict_ranks(
        CSRMatrix(item_features),
        CSRMatrix(user_features),
        CSRMatrix(test_interactions),
        CSRMatrix(train_interactions),
        ranks.data,
        lightfm_data,
        num_threads,
    )

    return ranks
```

- [ ] **Step 3: Refactor `LightFM.predict` to delegate**

In `lightfm/lightfm.py`, replace the body of `predict` (lines ~828-883) with:

```python
        self._check_initialized()
        from lightfm.inference._predict import _InferenceData, _predict_impl
        data = _InferenceData(
            item_embeddings=self.item_embeddings,
            user_embeddings=self.user_embeddings,
            item_biases=self.item_biases,
            user_biases=self.user_biases,
            no_components=self.no_components,
            learning_schedule=self.learning_schedule,
            item_embedding_gradients=self.item_embedding_gradients,
            item_embedding_momentum=self.item_embedding_momentum,
            item_bias_gradients=self.item_bias_gradients,
            item_bias_momentum=self.item_bias_momentum,
            user_embedding_gradients=self.user_embedding_gradients,
            user_embedding_momentum=self.user_embedding_momentum,
            user_bias_gradients=self.user_bias_gradients,
            user_bias_momentum=self.user_bias_momentum,
            learning_rate=self.learning_rate,
            rho=self.rho,
            epsilon=self.epsilon,
            max_sampled=self.max_sampled,
        )
        return _predict_impl(
            data,
            user_ids,
            item_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )
```

Leave the docstring and signature untouched. Keep the opening `self._check_initialized()` call (shown in the replacement snippet above) — that's the guard against unfit predict calls and must remain on the training-side code path. Remove everything after it (input normalization, `_construct_feature_matrices` call, `_get_lightfm_data` call, `predict_lightfm` call) — all moved to `_predict.py`. Do not leave behind any stray `predictions = np.empty(...)` or similar.

- [ ] **Step 4: Refactor `LightFM.predict_rank` to delegate**

Similarly, replace the body of `predict_rank` (lines ~950-1000) with:

```python
        self._check_initialized()
        from lightfm.inference._predict import _InferenceData, _predict_rank_impl
        data = _InferenceData(
            item_embeddings=self.item_embeddings,
            user_embeddings=self.user_embeddings,
            item_biases=self.item_biases,
            user_biases=self.user_biases,
            no_components=self.no_components,
            learning_schedule=self.learning_schedule,
            item_embedding_gradients=self.item_embedding_gradients,
            item_embedding_momentum=self.item_embedding_momentum,
            item_bias_gradients=self.item_bias_gradients,
            item_bias_momentum=self.item_bias_momentum,
            user_embedding_gradients=self.user_embedding_gradients,
            user_embedding_momentum=self.user_embedding_momentum,
            user_bias_gradients=self.user_bias_gradients,
            user_bias_momentum=self.user_bias_momentum,
            learning_rate=self.learning_rate,
            rho=self.rho,
            epsilon=self.epsilon,
            max_sampled=self.max_sampled,
        )
        return _predict_rank_impl(
            data,
            test_interactions,
            train_interactions=train_interactions,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
            check_intersections=check_intersections,
        )
```

- [ ] **Step 5: Leave `_construct_feature_matrices`, `_to_cython_dtype`, `_check_test_train_intersections`, `_get_lightfm_data` in `LightFM`**

These are still used by the `fit` / `fit_partial` path (confirmed by existing test suite). Do not delete them. The refactor only redirects `predict` and `predict_rank`; training paths stay on their existing helpers.

- [ ] **Step 6: Run the full test suite — must match baseline**

Run: `uv run pytest tests/ -q`
Expected: same pass count as the baseline from Step 1. Zero regressions.

If any test fails, **do not** commit. Compare against the baseline, find the behavior difference, fix `_predict.py`. Common causes:
- Missing list/tuple coercion branch in `_validate_predict_inputs`
- Wrong dtype for `FastLightFM` hyperparams
- `_InferenceData` field not passed through

- [ ] **Step 7: Commit**

```bash
git add lightfm/inference/_predict.py lightfm/lightfm.py
git commit -m "Extract _predict_impl / _predict_rank_impl for sharing with InferenceLightFM"
```

---

## Chunk 2: `InferenceLightFM` class + fork-sharing test

By the end of this chunk, an `InferenceLightFM.load(path).predict(...)` call works end-to-end against an artifact written by `_artifact.save`, and the load-bearing fork-sharing test has been written and verified on Linux.

### Task 5: `InferenceLightFM` skeleton, `load`, and properties (TDD)

**Files:**
- Create: `lightfm/inference/model.py`
- Modify: `lightfm/inference/__init__.py` (re-export)
- Create: `tests/inference/test_model.py`

- [ ] **Step 1: Write failing tests for the loader and properties**

`tests/inference/test_model.py`:

```python
import numpy as np
import pytest
import scipy.sparse as sp

from lightfm.inference import _artifact
from lightfm.inference.model import InferenceLightFM


def _save_tiny(tmp_path, n_items=50, n_users=20, k=8, loss="warp"):
    rng = np.random.default_rng(0)
    arrays = {
        "item_embeddings": rng.standard_normal((n_items, k), dtype=np.float32),
        "user_embeddings": rng.standard_normal((n_users, k), dtype=np.float32),
        "item_biases": rng.standard_normal(n_items, dtype=np.float32),
        "user_biases": rng.standard_normal(n_users, dtype=np.float32),
    }
    path = tmp_path / "model.safetensors"
    _artifact.save(path, arrays=arrays, metadata={
        "no_components": k,
        "loss": loss,
        "learning_schedule": "adagrad",
    })
    return path, arrays


def test_load_returns_inference_lightfm(tmp_path):
    path, arrays = _save_tiny(tmp_path)
    model = InferenceLightFM.load(path)
    assert isinstance(model, InferenceLightFM)


def test_properties_match_saved(tmp_path):
    path, arrays = _save_tiny(tmp_path, k=16, loss="bpr")
    model = InferenceLightFM.load(path)
    assert model.no_components == 16
    assert model.loss == "bpr"
    assert model.learning_schedule == "adagrad"
    np.testing.assert_array_equal(model.item_embeddings, arrays["item_embeddings"])
    np.testing.assert_array_equal(model.user_embeddings, arrays["user_embeddings"])
    np.testing.assert_array_equal(model.item_biases, arrays["item_biases"])
    np.testing.assert_array_equal(model.user_biases, arrays["user_biases"])


def test_load_mmap_flag_controls_base(tmp_path):
    path, _ = _save_tiny(tmp_path)
    mmapped = InferenceLightFM.load(path, mmap=True)
    heap = InferenceLightFM.load(path, mmap=False)
    assert mmapped.item_embeddings.base is not None  # file-backed
    # heap-loaded arrays may or may not have base=None depending on safetensors;
    # the load semantics test just verifies this doesn't raise.


def test_no_fit_attribute():
    assert not hasattr(InferenceLightFM, "fit")
    assert not hasattr(InferenceLightFM, "fit_partial")
```

- [ ] **Step 2: Run tests — expect failures**

Run: `uv run pytest tests/inference/test_model.py -v`
Expected: FAIL with `ImportError` — `InferenceLightFM` doesn't exist yet.

- [ ] **Step 3: Implement `InferenceLightFM` skeleton**

`lightfm/inference/model.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from lightfm.inference import _artifact
from lightfm.inference._predict import (
    _InferenceData,
    _predict_impl,
    _predict_rank_impl,
)


class InferenceLightFM:
    """Memory-mapped, predict-only LightFM model.

    Load an artifact written by `LightFM.save_for_inference()` (or produced by
    `lightfm.inference.convert.convert_joblib_to_inference`). Supports
    `predict()` and `predict_rank()` with signatures matching `LightFM`'s.
    Does not support fitting — use `LightFM` for that.
    """

    def __init__(
        self,
        *,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, str],
    ):
        self._arrays = arrays
        self._metadata = metadata
        self._no_components = int(metadata["no_components"])
        self._loss = metadata["loss"]
        self._learning_schedule = metadata["learning_schedule"]

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> "InferenceLightFM":
        arrays, metadata = _artifact.load(path, mmap=mmap)
        return cls(arrays=arrays, metadata=metadata)

    # ---- Introspection ----

    @property
    def no_components(self) -> int:
        return self._no_components

    @property
    def loss(self) -> str:
        return self._loss

    @property
    def learning_schedule(self) -> str:
        return self._learning_schedule

    @property
    def item_embeddings(self) -> np.ndarray:
        return self._arrays["item_embeddings"]

    @property
    def user_embeddings(self) -> np.ndarray:
        return self._arrays["user_embeddings"]

    @property
    def item_biases(self) -> np.ndarray:
        return self._arrays["item_biases"]

    @property
    def user_biases(self) -> np.ndarray:
        return self._arrays["user_biases"]

    # ---- Prediction ----

    def _inference_data(self) -> _InferenceData:
        return _InferenceData(
            item_embeddings=self.item_embeddings,
            user_embeddings=self.user_embeddings,
            item_biases=self.item_biases,
            user_biases=self.user_biases,
            no_components=self._no_components,
            learning_schedule=self._learning_schedule,
            # gradients/momentum stay None → _build_fast_lightfm fills dummies
        )

    def predict(
        self,
        user_ids,
        item_ids,
        item_features: sp.csr_matrix | None = None,
        user_features: sp.csr_matrix | None = None,
        num_threads: int = 1,
    ) -> NDArray[np.float32]:
        return _predict_impl(
            self._inference_data(),
            user_ids,
            item_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )

    def predict_rank(
        self,
        test_interactions: sp.csr_matrix,
        train_interactions: sp.csr_matrix | None = None,
        item_features: sp.csr_matrix | None = None,
        user_features: sp.csr_matrix | None = None,
        num_threads: int = 1,
        check_intersections: bool = True,
    ) -> sp.csr_matrix:
        return _predict_rank_impl(
            self._inference_data(),
            test_interactions,
            train_interactions=train_interactions,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
            check_intersections=check_intersections,
        )
```

Update `lightfm/inference/__init__.py`:

```python
from lightfm.inference._artifact import ArtifactLoadError
from lightfm.inference.model import InferenceLightFM

__all__ = ["InferenceLightFM", "ArtifactLoadError"]
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/inference/test_model.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Full suite sanity check**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lightfm/inference/model.py lightfm/inference/__init__.py tests/inference/test_model.py
git commit -m "Add InferenceLightFM skeleton with load() and properties"
```

---

### Task 6: Round-trip prediction parity (TDD)

**Files:**
- Modify: `tests/inference/test_model.py` (add round-trip test)

Train a small `LightFM` model, save its arrays via `_artifact.save`, load into `InferenceLightFM`, and verify `predict` outputs match bit-for-bit.

- [ ] **Step 1: Write failing parity test**

Append to `tests/inference/test_model.py`:

```python
from lightfm import LightFM


def _train_tiny_model(n_users=30, n_items=50, k=8, loss="warp", schedule="adagrad"):
    rng = np.random.default_rng(7)
    # Random dense interaction matrix, thresholded.
    dense = (rng.random((n_users, n_items)) > 0.8).astype(np.float32)
    interactions = sp.csr_matrix(dense)
    model = LightFM(no_components=k, loss=loss, learning_schedule=schedule, random_state=1)
    model.fit(interactions, epochs=2)
    return model, interactions


def _save_from_trained(model, tmp_path):
    path = tmp_path / "model.safetensors"
    _artifact.save(
        path,
        arrays={
            "item_embeddings": model.item_embeddings,
            "user_embeddings": model.user_embeddings,
            "item_biases": model.item_biases,
            "user_biases": model.user_biases,
        },
        metadata={
            "no_components": model.no_components,
            "loss": model.loss,
            "learning_schedule": model.learning_schedule,
        },
    )
    return path


def test_predict_bit_identical_to_lightfm(tmp_path):
    model, _ = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    rng = np.random.default_rng(99)
    user_ids = rng.integers(0, 30, size=200, dtype=np.int32)
    item_ids = rng.integers(0, 50, size=200, dtype=np.int32)

    expected = model.predict(user_ids, item_ids)
    actual = inf.predict(user_ids, item_ids)

    np.testing.assert_array_equal(actual, expected)


def test_predict_int_user_id_shorthand(tmp_path):
    model, _ = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    # Exercises the isinstance(user_ids, int) branch in _validate_predict_inputs.
    item_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    expected = model.predict(0, item_ids)
    actual = inf.predict(0, item_ids)
    np.testing.assert_array_equal(actual, expected)


def test_predict_list_input_coerced(tmp_path):
    model, _ = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    # Exercises the list/tuple coercion branches.
    expected = model.predict([0, 1, 2], [3, 4, 5])
    actual = inf.predict([0, 1, 2], [3, 4, 5])
    np.testing.assert_array_equal(actual, expected)


def test_predict_rank_bit_identical_to_lightfm(tmp_path):
    model, interactions = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    # Construct a small test interactions matrix disjoint from the training one
    # (predict_rank with check_intersections=True will reject overlapping entries).
    rng = np.random.default_rng(55)
    dense = (rng.random(interactions.shape) > 0.95).astype(np.float32)
    test_interactions = sp.csr_matrix(dense).multiply(interactions.toarray() == 0).tocsr()

    expected = model.predict_rank(test_interactions)
    actual = inf.predict_rank(test_interactions)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
```

- [ ] **Step 2: Run tests — expect pass (Task 4's refactor + Task 5's class already make this work)**

Run: `uv run pytest tests/inference/test_model.py -v`
Expected: all tests pass. If parity fails, the bug is in `_predict_impl` (likely a missed coercion branch); fix it in `_predict.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/inference/test_model.py
git commit -m "Add InferenceLightFM predict parity tests vs LightFM"
```

---

### Task 7: Fork-sharing RSS/PSS test (the load-bearing perf claim)

**Files:**
- Create: `tests/inference/test_fork_sharing.py`

This is the core perf claim. Linux: measure PSS. macOS: coarser RSS sanity check.

- [ ] **Step 1: Add `psutil` to dev dependencies**

Edit `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "flake8", "pre-commit", "psutil"]
```

Install: `uv pip install -e ".[dev]"`

(The test itself uses `pytest.importorskip("psutil")`, so CI jobs that run without `[dev]` extras simply skip the test rather than error.)

- [ ] **Step 2: Write the fork-sharing test**

`tests/inference/test_fork_sharing.py`:

```python
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
            assert rss < 1.5 * total_bytes, (
                f"worker pid {pid} RSS {rss:,} exceeds 1.5× model size "
                f"{total_bytes:,} — mmap sharing regressed"
            )
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/inference/test_fork_sharing.py -v -s`
Expected: passes on Linux and macOS. On macOS the assertion is looser.

**If the Linux assertion fails with total_pss much larger than model size,** mmap is not working. Debug: add prints of individual worker PSS, verify `model.item_embeddings.base is not None` inside the worker, check that the safetensors wheel is the real Rust one (`python -c "import safetensors; print(safetensors.__file__)"`).

- [ ] **Step 4: Full suite sanity check**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/inference/test_fork_sharing.py
git commit -m "Add fork-sharing PSS/RSS test — the load-bearing perf claim"
```

---

## Chunk 3: save_for_inference + converter + remaining tests

Final chunk. Training class gets `save_for_inference()`, the joblib→safetensors converter lands with a CLI, and the full test matrix (parity across losses/schedules, sparse features, error paths, cross-version joblib) is completed.

### Task 8: `LightFM.save_for_inference` (TDD)

**Files:**
- Modify: `lightfm/lightfm.py`
- Modify: `tests/inference/test_model.py` (add save-side round-trip)

- [ ] **Step 1: Write failing test**

Append to `tests/inference/test_model.py`:

```python
def test_save_for_inference_then_load(tmp_path):
    model, _ = _train_tiny_model(loss="logistic")
    path = tmp_path / "saved.safetensors"
    model.save_for_inference(path)

    inf = InferenceLightFM.load(path, mmap=False)
    assert inf.no_components == model.no_components
    assert inf.loss == "logistic"

    rng = np.random.default_rng(42)
    user_ids = rng.integers(0, 30, size=50, dtype=np.int32)
    item_ids = rng.integers(0, 50, size=50, dtype=np.int32)
    np.testing.assert_array_equal(inf.predict(user_ids, item_ids), model.predict(user_ids, item_ids))


def test_save_for_inference_unfitted_raises():
    model = LightFM()
    with pytest.raises(ValueError, match="fit the model"):
        model.save_for_inference("/tmp/should-not-be-written.safetensors")
```

- [ ] **Step 2: Run tests — expect failure**

Run: `uv run pytest tests/inference/test_model.py::test_save_for_inference_then_load -v`
Expected: FAIL — `save_for_inference` doesn't exist.

- [ ] **Step 3: Implement `LightFM.save_for_inference`**

Add to `LightFM` class in `lightfm/lightfm.py`, near `predict_rank` (after the existing `get_user_representations` / `get_item_representations` is fine — place it after `predict_rank`):

```python
    def save_for_inference(self, path: str | Path) -> None:
        """Save a slim inference-only artifact.

        Writes a single safetensors file containing only the arrays needed
        for prediction (`item_embeddings`, `user_embeddings`, `item_biases`,
        `user_biases`), along with minimal metadata. Training-only optimizer
        state (gradients, momentum) is omitted — typically halves artifact
        size vs. `joblib.dump(model)`.

        Load the result with `lightfm.inference.InferenceLightFM.load(path)`.

        Parameters
        ----------
        path : str or Path
            Destination path for the safetensors file.
        """
        self._check_initialized()
        from lightfm.inference import _artifact
        _artifact.save(
            path,
            arrays={
                "item_embeddings": self.item_embeddings,
                "user_embeddings": self.user_embeddings,
                "item_biases": self.item_biases,
                "user_biases": self.user_biases,
            },
            metadata={
                "no_components": self.no_components,
                "loss": self.loss,
                "learning_schedule": self.learning_schedule,
            },
        )
```

Add `from pathlib import Path` to the top of `lightfm/lightfm.py` if not already present.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/inference/test_model.py -v`
Expected: all tests pass, including the new ones.

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lightfm/lightfm.py tests/inference/test_model.py
git commit -m "Add LightFM.save_for_inference()"
```

---

### Task 9: Converter (TDD)

**Files:**
- Create: `lightfm/inference/convert.py`
- Create: `tests/inference/test_convert.py`
- Modify: `lightfm/inference/__init__.py` (re-export `ConversionError`)

Skip the fsspec integration for now — do local paths only. fsspec support comes in Task 10.

- [ ] **Step 1: Write failing test**

`tests/inference/test_convert.py`:

```python
import joblib
import numpy as np
import pytest
import scipy.sparse as sp

from lightfm import LightFM
from lightfm.inference import ConversionError
from lightfm.inference.convert import convert_joblib_to_inference
from lightfm.inference.model import InferenceLightFM


def _train_and_dump(tmp_path, loss="warp"):
    rng = np.random.default_rng(5)
    interactions = sp.csr_matrix((rng.random((30, 50)) > 0.8).astype(np.float32))
    model = LightFM(no_components=8, loss=loss, random_state=2)
    model.fit(interactions, epochs=2)
    src = tmp_path / "lightfm.joblib"
    joblib.dump(model, src)
    return src, model


def test_convert_roundtrip_parity(tmp_path):
    src, trained = _train_and_dump(tmp_path)
    dst = tmp_path / "model.safetensors"

    convert_joblib_to_inference(src, dst)

    inf = InferenceLightFM.load(dst, mmap=False)
    assert inf.no_components == trained.no_components
    assert inf.loss == trained.loss

    rng = np.random.default_rng(321)
    user_ids = rng.integers(0, 30, size=50, dtype=np.int32)
    item_ids = rng.integers(0, 50, size=50, dtype=np.int32)
    np.testing.assert_array_equal(inf.predict(user_ids, item_ids), trained.predict(user_ids, item_ids))


def test_convert_source_missing(tmp_path):
    with pytest.raises(ConversionError):
        convert_joblib_to_inference(tmp_path / "nope.joblib", tmp_path / "out.safetensors")


def test_convert_source_not_lightfm(tmp_path):
    src = tmp_path / "not_lightfm.joblib"
    joblib.dump({"not": "a model"}, src)
    with pytest.raises(ConversionError, match="expected LightFM"):
        convert_joblib_to_inference(src, tmp_path / "out.safetensors")


def test_convert_source_not_fitted(tmp_path):
    src = tmp_path / "unfit.joblib"
    joblib.dump(LightFM(), src)
    with pytest.raises(ConversionError, match="not fitted"):
        convert_joblib_to_inference(src, tmp_path / "out.safetensors")


def test_convert_artifact_is_smaller(tmp_path):
    src, _ = _train_and_dump(tmp_path)
    dst = tmp_path / "model.safetensors"
    convert_joblib_to_inference(src, dst)
    assert dst.stat().st_size < src.stat().st_size
```

- [ ] **Step 2: Run tests — expect failures**

Run: `uv run pytest tests/inference/test_convert.py -v`
Expected: FAIL — `convert` module missing.

- [ ] **Step 3: Implement converter**

`lightfm/inference/convert.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path

import joblib

from lightfm.inference import _artifact

log = logging.getLogger(__name__)


class ConversionError(Exception):
    """Raised when converting a legacy joblib model fails."""


def convert_joblib_to_inference(
    src: str | Path,
    dst: str | Path,
    *,
    storage_options: dict | None = None,
) -> None:
    """Convert a legacy `joblib.dump(LightFM)` artifact to safetensors.

    Loads the entire model into RAM. For very large models (~60GB), run on a
    host with sufficient memory. Streaming conversion is a v2 concern.
    """
    from lightfm import LightFM  # local import avoids circular dep at module load

    src_path = Path(str(src))
    dst_path = Path(str(dst))

    try:
        loaded = joblib.load(str(src_path))
    except FileNotFoundError as exc:
        raise ConversionError(f"source not found: {src_path}") from exc
    except Exception as exc:
        raise ConversionError(f"failed to load {src_path}: {exc}") from exc

    if not isinstance(loaded, LightFM):
        raise ConversionError(
            f"expected LightFM, got {type(loaded).__name__} at {src_path}"
        )

    if loaded.item_embeddings is None:
        raise ConversionError(f"source model is not fitted: {src_path}")

    _artifact.save(
        dst_path,
        arrays={
            "item_embeddings": loaded.item_embeddings,
            "user_embeddings": loaded.user_embeddings,
            "item_biases": loaded.item_biases,
            "user_biases": loaded.user_biases,
        },
        metadata={
            "no_components": loaded.no_components,
            "loss": loaded.loss,
            "learning_schedule": loaded.learning_schedule,
        },
    )

    src_size = src_path.stat().st_size
    dst_size = dst_path.stat().st_size
    pct = 100.0 * (1.0 - dst_size / src_size) if src_size > 0 else 0.0
    log.info(
        "converted %s (%d bytes) → %s (%d bytes), %.1f%% smaller",
        src_path, src_size, dst_path, dst_size, pct,
    )
```

Library callers see the message if they configure logging; silent by default — which is the right shape for a function called from other code. The CLI wrapper (Task 10) enables INFO logging so end users see the message on stderr.

Update `lightfm/inference/__init__.py`:

```python
from lightfm.inference._artifact import ArtifactLoadError
from lightfm.inference.convert import ConversionError
from lightfm.inference.model import InferenceLightFM

__all__ = ["InferenceLightFM", "ArtifactLoadError", "ConversionError"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/inference/test_convert.py -v`
Expected: all pass.

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lightfm/inference/convert.py lightfm/inference/__init__.py tests/inference/test_convert.py
git commit -m "Add convert_joblib_to_inference (local paths)"
```

---

### Task 10: CLI entrypoint + fsspec for `gs://` URIs

**Files:**
- Modify: `lightfm/inference/convert.py` (add `main()` + `if __name__ == "__main__"`; add fsspec support)
- Modify: `tests/inference/test_convert.py` (CLI test)

`python -m <dotted.path>` works for any importable module with an `if __name__ == "__main__"` block — module OR package. Since `convert` already exists as `lightfm/inference/convert.py`, we add the CLI guard directly to it, matching the spec's `python -m lightfm.inference.convert` hint. No separate `__main__.py` needed.

`fsspec` remains a soft dep (imported lazily only when a remote URI is used). Not added to `pyproject.toml`.

- [ ] **Step 1: Write failing CLI test**

Append to `tests/inference/test_convert.py`:

```python
import subprocess
import sys


def test_cli_runs(tmp_path):
    src, _ = _train_and_dump(tmp_path)
    dst = tmp_path / "cli_out.safetensors"
    result = subprocess.run(
        [sys.executable, "-m", "lightfm.inference.convert", str(src), str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert dst.exists()
    # The CLI configures logging → info messages go to stderr.
    assert "smaller" in result.stderr
```

- [ ] **Step 2: Run test — expect failure**

Run: `uv run pytest tests/inference/test_convert.py::test_cli_runs -v`
Expected: FAIL — no CLI guard in `convert.py` yet (running `python -m lightfm.inference.convert` exits silently with code 0 because there's no `if __name__ == "__main__"` block).

- [ ] **Step 3: Replace `lightfm/inference/convert.py` with the full final version (adds fsspec + CLI guard)**

Replace the entire file contents:

```python
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import joblib

from lightfm.inference import _artifact

log = logging.getLogger(__name__)


class ConversionError(Exception):
    """Raised when converting a legacy joblib model fails."""


def _is_remote(path: str | Path) -> bool:
    s = str(path)
    return "://" in s and not s.startswith("file://")


def _scheme(path: str | Path) -> str:
    return str(path).split("://", 1)[0]


@contextmanager
def _open_read(path: str | Path, storage_options: dict | None):
    if _is_remote(path):
        import fsspec
        with fsspec.open(str(path), "rb", **(storage_options or {})) as f:
            yield f
    else:
        with open(str(path), "rb") as f:
            yield f


def _file_size(path: str | Path, storage_options: dict | None) -> int:
    if _is_remote(path):
        import fsspec
        fs = fsspec.filesystem(_scheme(path), **(storage_options or {}))
        return int(fs.size(str(path)))
    return Path(str(path)).stat().st_size


def _write_artifact(
    dst: str | Path,
    *,
    arrays: dict,
    metadata: dict,
    storage_options: dict | None,
) -> None:
    """Write via _artifact.save to `dst`. For remote URIs, stage through a
    local tempfile then upload."""
    if _is_remote(dst):
        import fsspec
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _artifact.save(tmp_path, arrays=arrays, metadata=metadata)
            # Streaming copy — critical at the 60GB scale this project targets;
            # a .read() into memory would OOM the converter host.
            with fsspec.open(str(dst), "wb", **(storage_options or {})) as out, \
                 open(tmp_path, "rb") as src_f:
                shutil.copyfileobj(src_f, out)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        _artifact.save(Path(str(dst)), arrays=arrays, metadata=metadata)


def convert_joblib_to_inference(
    src: str | Path,
    dst: str | Path,
    *,
    storage_options: dict | None = None,
) -> None:
    """Convert a legacy `joblib.dump(LightFM)` artifact to safetensors.

    Loads the entire model into RAM. For very large models (~60GB), run on a
    host with sufficient memory. Streaming conversion is a v2 concern.
    """
    from lightfm import LightFM  # local import avoids circular dep

    try:
        with _open_read(src, storage_options) as f:
            loaded = joblib.load(f)
    except FileNotFoundError as exc:
        raise ConversionError(f"source not found: {src}") from exc
    except Exception as exc:
        raise ConversionError(f"failed to load {src}: {exc}") from exc

    if not isinstance(loaded, LightFM):
        raise ConversionError(
            f"expected LightFM, got {type(loaded).__name__} at {src}"
        )
    if loaded.item_embeddings is None:
        raise ConversionError(f"source model is not fitted: {src}")

    arrays = {
        "item_embeddings": loaded.item_embeddings,
        "user_embeddings": loaded.user_embeddings,
        "item_biases": loaded.item_biases,
        "user_biases": loaded.user_biases,
    }
    metadata = {
        "no_components": loaded.no_components,
        "loss": loaded.loss,
        "learning_schedule": loaded.learning_schedule,
    }
    _write_artifact(dst, arrays=arrays, metadata=metadata, storage_options=storage_options)

    src_size = _file_size(src, storage_options)
    dst_size = _file_size(dst, storage_options)
    pct = 100.0 * (1.0 - dst_size / src_size) if src_size > 0 else 0.0
    log.info(
        "converted %s (%d bytes) → %s (%d bytes), %.1f%% smaller",
        src, src_size, dst, dst_size, pct,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lightfm.inference.convert",
        description="Convert a joblib.dump(LightFM) artifact to safetensors.",
    )
    parser.add_argument("src", help="Source path or URI (local or gs://..., s3://..., etc.)")
    parser.add_argument("dst", help="Destination path or URI")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    try:
        convert_joblib_to_inference(args.src, args.dst)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/inference/test_convert.py -v`
Expected: all pass, including the CLI test. No remote-path tests in CI (would need GCS/S3 credentials — those are a post-merge manual smoke per the spec's rollout section).

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lightfm/inference/convert.py tests/inference/test_convert.py
git commit -m "Add convert CLI guard and fsspec support for remote URIs"
```

---

### Task 10a: Cross-version joblib fixture (deferred, documented only)

**Files:**
- Create: `tests/inference/fixtures/README.md`

**Status: deferred to a follow-up PR.** Rationale:

The spec calls for a test where a `joblib.dump(LightFM)` produced under `numpy < 2.0` round-trips through the converter running under `numpy >= 2.0` — this is the realistic d152 scenario because the existing 60GB blob was dumped in a numpy-1.x training environment. Generating that fixture in CI requires one of:

1. A binary fixture checked into the repo, produced once externally under `numpy<2`. ~kB for a tiny model. Viable but adds a binary artifact to the repo with no automated way to regenerate.
2. A secondary CI job with `numpy<2` installed, which then dumps the fixture, then the primary job converts it. Requires a matrix CI setup we don't have today.
3. A post-merge manual smoke on an actual d152 artifact copy. Real but not automated.

For v1 we rely on (3) as the integration gate (the first real production conversion against `gs://.../lightfm.joblib` is the actual test). For v2, option 1 is the cheapest automation.

- [ ] **Step 1: Document this deferral**

Create `tests/inference/fixtures/README.md`:

```markdown
# Inference test fixtures

## Deferred: cross-numpy-version joblib fixture

The converter (`lightfm.inference.convert.convert_joblib_to_inference`) must
load a `joblib.dump(LightFM)` produced under `numpy < 2.0` when running under
`numpy >= 2.0` — the realistic d152 production scenario.

v1 validates this via a **manual post-merge smoke** against a copy of a real
d152 artifact. See the spec's "Operational Rollout" section.

Follow-up work (tracked separately): check in a small binary fixture
(`legacy_numpy1_lightfm.joblib`, a few KB) produced by training a tiny model
under a Python env with `numpy<2` + `joblib<1.3`. A CI test then asserts
`convert_joblib_to_inference(fixture) → load → predict` works. Regeneration
instructions: `uv run --with "numpy<2,joblib<1.3" python scripts/make_fixture.py`.
```

- [ ] **Step 2: Commit**

```bash
git add tests/inference/fixtures/README.md
git commit -m "Document deferred cross-numpy-version joblib fixture"
```

---

### Task 11: Parity across losses and learning schedules

**Files:**
- Create: `tests/inference/test_loss_variants.py`

- [ ] **Step 1: Write the parametric parity test**

```python
import numpy as np
import pytest
import scipy.sparse as sp

from lightfm import LightFM
from lightfm.inference.model import InferenceLightFM
from lightfm.inference import _artifact


@pytest.mark.parametrize("loss", ["logistic", "bpr", "warp", "warp-kos"])
@pytest.mark.parametrize("schedule", ["adagrad", "adadelta"])
def test_predict_parity_across_loss_and_schedule(tmp_path, loss, schedule):
    rng = np.random.default_rng(11)
    interactions = sp.csr_matrix((rng.random((40, 60)) > 0.8).astype(np.float32))
    model = LightFM(
        no_components=16,
        loss=loss,
        learning_schedule=schedule,
        random_state=3,
    )
    model.fit(interactions, epochs=2)

    path = tmp_path / "m.safetensors"
    _artifact.save(
        path,
        arrays={
            "item_embeddings": model.item_embeddings,
            "user_embeddings": model.user_embeddings,
            "item_biases": model.item_biases,
            "user_biases": model.user_biases,
        },
        metadata={
            "no_components": model.no_components,
            "loss": model.loss,
            "learning_schedule": model.learning_schedule,
        },
    )
    inf = InferenceLightFM.load(path, mmap=False)

    user_ids = rng.integers(0, 40, size=1000, dtype=np.int32)
    item_ids = rng.integers(0, 60, size=1000, dtype=np.int32)
    np.testing.assert_array_equal(inf.predict(user_ids, item_ids), model.predict(user_ids, item_ids))
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/inference/test_loss_variants.py -v`
Expected: 8 parametrized cases, all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/inference/test_loss_variants.py
git commit -m "Add predict parity tests across loss and learning_schedule"
```

---

### Task 12: Sparse features path

**Files:**
- Create: `tests/inference/test_sparse_features.py`

- [ ] **Step 1: Write the sparse-features test**

```python
import numpy as np
import scipy.sparse as sp

from lightfm import LightFM
from lightfm.inference import _artifact
from lightfm.inference.model import InferenceLightFM


def test_predict_with_item_and_user_features(tmp_path):
    rng = np.random.default_rng(13)
    n_users, n_items = 30, 50
    n_user_features, n_item_features = 10, 15

    user_features = sp.csr_matrix((rng.random((n_users, n_user_features)) > 0.5).astype(np.float32))
    item_features = sp.csr_matrix((rng.random((n_items, n_item_features)) > 0.5).astype(np.float32))
    interactions = sp.csr_matrix((rng.random((n_users, n_items)) > 0.8).astype(np.float32))

    model = LightFM(no_components=8, loss="warp", random_state=7)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=2)

    path = tmp_path / "m.safetensors"
    _artifact.save(
        path,
        arrays={
            "item_embeddings": model.item_embeddings,
            "user_embeddings": model.user_embeddings,
            "item_biases": model.item_biases,
            "user_biases": model.user_biases,
        },
        metadata={
            "no_components": model.no_components,
            "loss": model.loss,
            "learning_schedule": model.learning_schedule,
        },
    )
    inf = InferenceLightFM.load(path, mmap=False)

    user_ids = rng.integers(0, n_users, size=100, dtype=np.int32)
    item_ids = rng.integers(0, n_items, size=100, dtype=np.int32)

    expected = model.predict(user_ids, item_ids, user_features=user_features, item_features=item_features)
    actual = inf.predict(user_ids, item_ids, user_features=user_features, item_features=item_features)
    np.testing.assert_array_equal(actual, expected)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/inference/test_sparse_features.py -v`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/inference/test_sparse_features.py
git commit -m "Add sparse features predict parity test"
```

---

### Task 13: Re-export `InferenceLightFM` at top level

**Files:**
- Modify: `lightfm/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

Replace `lightfm/__init__.py`:

```python
from .lightfm import LightFM
from .inference import InferenceLightFM
from .version import __version__

__all__ = ["LightFM", "InferenceLightFM", "datasets", "evaluation", "__version__"]
```

- [ ] **Step 2: Smoke-test**

Run: `uv run python -c "from lightfm import InferenceLightFM; print(InferenceLightFM)"`
Expected: prints the class.

- [ ] **Step 3: Full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add lightfm/__init__.py
git commit -m "Re-export InferenceLightFM at package top level"
```

---

### Task 14: Final verification — full suite + one-off sanity predict

- [ ] **Step 1: Full suite with verbose output for inference tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all tests pass, including the `tests/inference/` tree.

- [ ] **Step 2: Eyeball count**

Compare test count against the pre-refactor baseline from Chunk 1 Task 4 Step 1. Expected: N_baseline + all the newly-added tests (model, artifact, convert, fork_sharing, loss_variants, sparse_features).

- [ ] **Step 3: Manual end-to-end smoke**

```bash
uv run python - <<'EOF'
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM, InferenceLightFM

rng = np.random.default_rng(0)
interactions = sp.csr_matrix((rng.random((100, 200)) > 0.8).astype(np.float32))
model = LightFM(no_components=32, loss="warp", random_state=1)
model.fit(interactions, epochs=3)

model.save_for_inference("/tmp/smoke.safetensors")

inf = InferenceLightFM.load("/tmp/smoke.safetensors")
print("loaded:", inf.no_components, inf.loss)

user_ids = rng.integers(0, 100, size=10, dtype=np.int32)
item_ids = rng.integers(0, 200, size=10, dtype=np.int32)
trained_scores = model.predict(user_ids, item_ids)
inf_scores = inf.predict(user_ids, item_ids)
assert np.array_equal(trained_scores, inf_scores), "parity broken"
print("parity: OK")
print("trained sample:", trained_scores[:3])
print("inference sample:", inf_scores[:3])
EOF
```

Expected: prints `loaded: 32 warp`, `parity: OK`, and two matching score samples.

- [ ] **Step 4: No commit for this task — already fully committed by prior tasks**

---

## Exit criteria

- All tests in `tests/` pass (`uv run pytest tests/ -q`).
- `tests/inference/test_fork_sharing.py` passes on Linux (PSS < 1.5× model) and macOS (RSS check).
- `LightFM.predict` and `LightFM.predict_rank` still produce bit-identical outputs to pre-refactor (verified by the unchanged existing `tests/` suite passing).
- `python -m lightfm.inference <src> <dst>` successfully converts a legacy joblib artifact.
- Branch `feature/inference-artifact` has ~14 commits, one per task (give or take minor fixes).
