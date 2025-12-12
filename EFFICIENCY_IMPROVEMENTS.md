# LightFM Efficiency Improvements

This document outlines potential efficiency improvements for the LightFM codebase, organized by impact and effort.

## Overview

LightFM is a mature hybrid recommendation library with strong Cython optimization and OpenMP parallelization. The codebase was recently updated for Cython 3.0 compatibility but uses older build tooling patterns. The suggestions below aim to improve both runtime performance and developer experience.

---

## High-Impact Improvements

### 1. SIMD Vectorization for Dot Products

**Current State:** The core bottleneck is in embedding dot products (`_lightfm_fast.pyx.template`), which use scalar loops:

```cython
for j in range(no_components):
    result += item_repr[j] * user_repr[j]
```

**Proposed Change:** Use NumPy's optimized BLAS via `np.dot()` or explicit SIMD intrinsics.

**Expected Impact:** 2-4x speedup in compute-bound operations.

**Effort:** Medium

**Notes:**
- Could use `scipy.linalg.blas` for direct BLAS calls
- Alternatively, restructure to batch dot products and use `np.einsum` or matrix multiplication
- Consider `cython.parallel` with SIMD-friendly loop structures

---

### 2. Memory Access Pattern Optimization

**Current State:** Embedding layout is `[n_features, n_components]`. During prediction, we iterate over components in the inner loop, which may not be cache-optimal depending on access patterns.

**Proposed Change:** Profile and potentially transpose to `[n_components, n_features]` for better cache locality during vector operations.

**Expected Impact:** 10-30% improvement in memory-bound operations.

**Effort:** Medium (requires careful benchmarking)

**Notes:**
- Use `perf` or `cachegrind` to measure cache miss rates
- May require changes to both Python and Cython code
- Test on representative workloads before committing

---

### 3. Build System Modernization (pyproject.toml)

**Current State:** Requires manual `python setup.py cythonize` before building. No `pyproject.toml` for PEP 517/518 compliance.

**Proposed Change:**
1. Add `pyproject.toml` with build system specification
2. Auto-cythonize during `pip install`
3. Declare build dependencies properly

**Expected Impact:** Significantly improved developer experience; standard `pip install -e .` workflow.

**Effort:** Low-Medium

**Example `pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "Cython>=3.0", "numpy>=1.21"]
build-backend = "setuptools.build_meta"

[project]
name = "lightfm"
dynamic = ["version"]
description = "LightFM recommendation library"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.17.0",
    "scipy>=0.17.0",
    "scikit-learn",
    "requests",
]

[project.optional-dependencies]
dev = ["pytest", "black", "flake8", "pre-commit"]
docs = ["sphinx", "sphinx_rtd_theme"]
```

---

## Medium-Impact Improvements

### 4. macOS/Windows OpenMP Support

**Current State:** OpenMP is disabled on macOS and Windows (`setup.py:162-164`):

```python
use_openmp = not sys.platform.startswith("darwin") and not sys.platform.startswith("win")
```

**Proposed Changes:**

**Option A: Platform-specific OpenMP libraries**
- macOS: Use `libomp` from Homebrew (`brew install libomp`)
- Windows: Use Intel OpenMP or MSVC OpenMP

**Option B: Alternative parallelization**
- Use Python's `concurrent.futures.ThreadPoolExecutor` as fallback
- Implement thread pool in pure Python for non-OpenMP platforms

**Expected Impact:** Parallel training on macOS/Windows (currently single-threaded).

**Effort:** Medium-High

**Notes:**
- Option A requires documentation for users to install OpenMP
- Option B provides consistent cross-platform experience but less performance

---

### 5. Memory Pool for Cython Allocations

**Current State:** Uses `malloc`/`free` in hot loops:

```cython
cdef flt* temp = <flt*>malloc(size * sizeof(flt))
# ... use temp ...
free(temp)
```

**Proposed Change:** Implement a simple memory pool or arena allocator for temporary allocations in tight loops.

**Expected Impact:** Reduced allocation overhead, especially for many small allocations.

**Effort:** Medium

**Notes:**
- Could use Cython's `cpython.mem` module
- Alternative: Pre-allocate workspace arrays at model construction time
- Consider thread-local pools for OpenMP compatibility

---

### 6. NumPy 2.0 Compatibility

**Current State:** Uses `np.float32` and sparse matrix operations that should be compatible, but full NumPy 2.0 testing needed.

**Proposed Change:**
1. Test against NumPy 2.0+ in CI
2. Fix any deprecation warnings (e.g., `np.object` -> `object`)
3. Update dtype handling if needed

**Expected Impact:** Future-proofing; avoid breakage for users on newer NumPy.

**Effort:** Low

---

## Lower-Priority / Larger Scope

### 7. GPU Acceleration (CuPy/CUDA)

**Current State:** CPU-only implementation.

**Proposed Change:** Add optional GPU backend using CuPy for array operations and custom CUDA kernels for training loops.

**Expected Impact:** 10-100x speedup for large-scale deployments.

**Effort:** High

**Notes:**
- Start with CuPy drop-in replacement for NumPy operations
- Profile to identify GPU-suitable operations
- Consider sparse GPU libraries (cuSPARSE)
- Make GPU optional dependency

---

### 8. Type Hints for Public API

**Current State:** No type hints in Python code (only Cython has explicit types).

**Proposed Change:** Add type hints to `lightfm.py`, `data.py`, and `evaluation.py`.

**Expected Impact:** Better IDE support, earlier bug detection, improved documentation.

**Effort:** Low

**Example:**

```python
from typing import Optional, Union
import numpy as np
from scipy.sparse import csr_matrix

def fit(
    self,
    interactions: csr_matrix,
    user_features: Optional[csr_matrix] = None,
    item_features: Optional[csr_matrix] = None,
    sample_weight: Optional[csr_matrix] = None,
    epochs: int = 1,
    num_threads: int = 1,
    verbose: bool = False,
) -> "LightFM":
    ...
```

---

### 9. Sparse Matrix Backend Options

**Current State:** Locked to SciPy sparse matrices.

**Proposed Change:** Support alternative sparse backends:
- `sparse` (PyData sparse)
- `cupy.sparse` (GPU sparse)
- Custom CSR implementation for specific optimizations

**Expected Impact:** Flexibility for different deployment scenarios; GPU compatibility.

**Effort:** High

---

### 10. Distributed Training

**Current State:** Single-machine only.

**Proposed Change:** Add support for distributed training via:
- Dask for distributed arrays
- Ray for distributed computing
- Horovod for data-parallel training

**Expected Impact:** Scale beyond single server memory/compute limits.

**Effort:** Very High

---

## Quick Reference

| Improvement | Effort | Runtime Impact | DX Impact |
|-------------|--------|----------------|-----------|
| SIMD vectorization | Medium | High | - |
| Memory access patterns | Medium | Medium | - |
| pyproject.toml | Low | - | High |
| macOS/Windows OpenMP | Medium-High | Medium | Medium |
| Memory pool | Medium | Medium | - |
| NumPy 2.0 compatibility | Low | - | Medium |
| GPU acceleration | High | Very High | - |
| Type hints | Low | - | Medium |
| Sparse backends | High | Medium | Low |
| Distributed training | Very High | Very High | - |

---

## Implementation Notes

### Benchmarking

Before implementing performance changes, establish baselines:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run benchmarks (create if needed)
python -m pytest tests/test_movielens.py -v --benchmark
```

Key metrics to track:
- Training time per epoch
- Prediction throughput (predictions/second)
- Memory usage (peak and steady-state)
- Parallelization speedup (1, 2, 4, 8 threads)

### Profiling Tools

- **CPU profiling:** `py-spy`, `cProfile`, `line_profiler`
- **Memory profiling:** `memory_profiler`, `tracemalloc`
- **Cache analysis:** `perf stat`, `cachegrind`
- **Cython annotation:** `cython -a` for HTML annotation

### Compatibility

Any changes should maintain:
- Python 3.8+ support
- NumPy 1.17+ support
- SciPy 0.17+ support
- Backward compatibility with existing saved models

---

## Contributing

When implementing these improvements:

1. Create a focused PR for each improvement
2. Include benchmarks showing before/after performance
3. Update tests to cover new functionality
4. Update documentation as needed
5. Follow existing code style (enforced by pre-commit hooks)
