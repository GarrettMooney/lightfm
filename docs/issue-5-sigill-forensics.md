# Issue #5: SIGILL in `LightFM.predict()` — investigation record

This document captures the forensic investigation into
[issue #5](https://github.com/GarrettMooney/lightfm/issues/5):
intermittent `SIGILL` / exit code 132 in `LightFM.predict()` on Modal
container fleets.

## TL;DR

The crash is **inside the OpenBLAS bundled in NumPy's wheel**
(`libscipy_openblas64_*.so`). Its runtime CPU dispatcher can select an
AVX-512 (Skylake-X) DGEMM kernel — `dgemm_small_kernel_b0_nn_SKYLAKEX`
— on a CPU that does not actually execute AVX-512, yielding `SIGILL`
on the first `vbroadcastsd %xmm0,%zmm25` instruction.

Compile flags in this fork (`-march=native`, `-ffast-math`) are not
involved. The Cython hot path in `_lightfm_fast*.so` is scalar C and
contains no wide-vector instructions; `LIGHTFM_NO_CFLAGS=1` does not
prevent the SIGILL.

## What was tested

Three phases, all on 2026-04-22 / 2026-04-23.

### Phase 1: source inspection

- `setup.py:64-70` passes `-ffast-math` and `-march=native` to gcc on
  Linux. Initial hypothesis: `-march=native` on the Modal *build*
  container bakes AVX-512 into the resulting `_lightfm_fast_openmp.so`,
  which then SIGILLs on a narrower *runtime* container.
- `_lightfm_fast.pyx.template:287-334` — the `predict_lightfm` hot path
  is pure scalar C operating on raw `float *` pointers (`malloc`'d
  buffers). The inner loop is `result += user_repr[i] * item_repr[i]`
  over `no_components` elements (typically 16-128). No numpy ops, no
  BLAS calls, no intrinsics.

### Phase 2: Modal probes

Two Modal scripts were run, both as one-off forensic probes (not
committed to the repo):

1. **objdump probe** — built lightfm from source on Modal under both
   default flags and `LIGHTFM_NO_CFLAGS=1`, then disassembled
   `_lightfm_fast_openmp.so` and counted AVX-512 markers
   (`%zmm*`, `%k0..%k7` mask registers, `vpdp{busd,wssd}` VNNI ops,
   `{1to*}` EVEX broadcast). Result: **zero** AVX-512 markers in
   either build, despite the builder CPU advertising `avx512f,
   avx512vl, avx512cd, avx512bw, avx512dq, avx512_vnni`. The Cython
   C is too irregular for gcc's auto-vectorizer to emit wide SIMD —
   `-march=native` is a no-op for this codebase.

2. **In-process repro** — ran the issue's exact repro 50 times with
   `faulthandler.enable()` on Modal, with `retries=0` per attempt.
   Result: **0/50 SIGILLs**. All 50 runtime containers landed on CPUs
   advertising only `avx, avx2` (no AVX-512). Bug did not reproduce
   in this fleet state.

### Phase 3: GCE controlled host

Created an `n1-standard-2` preemptible VM in
`gcp-dsw-marketing-it-sandbox` / `us-east1-b`. CPU was an Intel Xeon
@ 2.30GHz with `avx, avx2, bmi1, bmi2, fma, sse4_2` — **no AVX-512**.

Ran 30 trials × 3 regimes under `gdb --batch` with `handle SIGILL
stop`:

| Regime | Config | SIGILLs / 30 |
|---|---|---|
| A | default env | 0 |
| **B** | `OPENBLAS_CORETYPE=SKYLAKEX` | **30** |
| C | `LIGHTFM_NO_CFLAGS=1` rebuild | 0 |

Regime B caught the SIGILL deterministically. Backtrace:

```
Thread 1 "python" received signal SIGILL, Illegal instruction.
0x00007ffff64cb84b in dgemm_small_kernel_b0_nn_SKYLAKEX ()
   from .../numpy.libs/libscipy_openblas64_-32a4b2a6.so
#0  dgemm_small_kernel_b0_nn_SKYLAKEX     in libscipy_openblas64_-32a4b2a6.so
#1  scipy_cblas_dgemm64_                  in libscipy_openblas64_-32a4b2a6.so
#2  DOUBLE_matmul_matrixmatrix.isra.0     in _multiarray_umath.cpython-311-x86_64-linux-gnu.so
#3  DOUBLE_matmul                         in _multiarray_umath.cpython-311-x86_64-linux-gnu.so
#4  generic_wrapped_legacy_loop           in _multiarray_umath.cpython-311-x86_64-linux-gnu.so
#5  PyUFunc_GeneralizedFunctionInternal   in _multiarray_umath.cpython-311-x86_64-linux-gnu.so
#6  ufunc_generic_fastcall                in _multiarray_umath.cpython-311-x86_64-linux-gnu.so
... CPython eval loop ...
```

Crashing instruction: `vbroadcastsd %xmm0,%zmm25` (AVX-512). The
`%zmm25` register is 512-bit; on an AVX2-only CPU the decoder rejects
the EVEX-prefixed encoding → SIGILL.

## What this proves

- **Root cause is OpenBLAS dispatch**, specifically NumPy's bundled
  `libscipy_openblas64_*.so` selecting `dgemm_small_kernel_b0_nn_SKYLAKEX`
  on a CPU that does not support AVX-512.
- **Compile flags in this fork are irrelevant.** Regime C kept the
  SIGILL exactly as before. Removing `-march=native` would not change
  the outcome.
- **The original "fit works, predict crashes" asymmetry** is consistent
  with OpenBLAS's small-vs-large kernel dispatch: predict's matmul
  dimensions hit the SKYLAKEX `dgemm_small_kernel` path; fit's hit a
  different (safe) kernel path.
- **The faiss-cpu mitigation** observed in the issue is consistent with
  faiss-cpu shipping its own OpenBLAS that wins the `dlopen` resolution,
  preventing NumPy's bad dispatcher from being loaded.

## Mitigations (deployed in README)

1. `OPENBLAS_CORETYPE=HASWELL` — pins OpenBLAS to AVX2 kernels.
2. `OPENBLAS_NUM_THREADS=1` — sometimes steers around the small-matrix
   path.
3. Install `faiss-cpu` — empirical, changes load order.

## Reproducer

The minimum reproducer needs an AVX2-only x86_64 Linux host (or any
host where `OPENBLAS_CORETYPE=SKYLAKEX` triggers an actual ISA mismatch).

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.12", "lightfm"]
# ///
import os, faulthandler, sys
os.environ["OPENBLAS_CORETYPE"] = "SKYLAKEX"  # force AVX-512 dispatch
faulthandler.enable(file=sys.stderr)

import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix

rng = np.random.default_rng(0)
n_users, n_items = 50, 20
rows = rng.integers(0, n_users, size=200)
cols = rng.integers(0, n_items, size=200)
data = np.ones_like(rows, dtype=np.float32)
interactions = coo_matrix((data, (rows, cols)), shape=(n_users, n_items))

model = LightFM(no_components=16, loss="warp", random_state=0)
model.fit(interactions, epochs=1, num_threads=1)

user_ids = np.repeat(np.arange(5), n_items).astype(np.int32)
item_ids = np.tile(np.arange(n_items), 5).astype(np.int32)
print(model.predict(user_ids, item_ids, num_threads=1)[:3])
```

Run with `gdb --batch --nx -ex 'handle SIGILL stop' -ex run --args
python repro.py` to capture the backtrace.

## What's needed to file upstream

Two reports are valuable; drafts live in
[`issue-5-modal-report.md`](issue-5-modal-report.md) (Modal) and below
(numpy / OpenBLAS).

### NumPy / OpenBLAS upstream draft

Filing target options, in order of fit:

1. **`numpy/numpy`** — they ship the OpenBLAS wheel with this dispatcher.
   They are the right entry point even if the underlying bug is in
   `OpenMathLib/OpenBLAS`, because they control which OpenBLAS build
   their wheel ships and how its dispatcher is configured.
2. **`OpenMathLib/OpenBLAS`** — secondary, for the dispatcher logic
   itself.
3. **`MacPython/openblas-libs`** — if the issue is specific to the
   manylinux build process.

Information to include when filing:

- Title: "scipy_openblas64 dispatcher selects SKYLAKEX kernel on
  CPUs that mis-report AVX-512 support, causing SIGILL"
- The GCE backtrace above (proves the mechanism).
- numpy version: `2.4.4` (sdist resolved by `uv` for `numpy>=1.26`
  on Python 3.11 / debian_slim, observed 2026-04-22).
- OpenBLAS shared object hash: `libscipy_openblas64_-32a4b2a6.so`
  (the suffix `-32a4b2a6` identifies the build).
- Reproducer (above) showing `OPENBLAS_CORETYPE=SKYLAKEX` is enough
  to trigger SIGILL on AVX2-only hosts. Note this is a synthetic
  forced reproduction; we do **not** have a captured backtrace from
  a spontaneous Modal crash. The forced reproduction proves the
  mechanism but does not identify *which* CPU/dispatcher edge case
  triggered the spontaneous crash.
- Open question for upstream: under what CPU-feature-flag combinations
  will the dispatcher select `_SKYLAKEX` kernels? Is there a known
  issue with virtualized CPUID where guest CPUs report a feature bit
  the host hypervisor cannot deliver?
- Workarounds users can apply today (the three from the README).

### Outstanding unknowns

- We could not catch a spontaneous SIGILL on Modal (70 clean trials).
  Modal's fleet may have rotated out the affected hosts after the
  original bug filing; alternatively the bug requires a specific
  CPU-flag-combination we did not happen to land on.
- We have not confirmed *which* `OPENBLAS_CORETYPE` value the dispatcher
  picked spontaneously on the original crashing Modal containers. The
  forced `SKYLAKEX` reproduction proves the kernel-selection-mismatch
  mechanism but not the specific kernel involved.
- We have not confirmed whether `OPENBLAS_CORETYPE=HASWELL` actually
  prevents the spontaneous Modal crash (untested because we cannot
  reproduce the spontaneous form).

If the bug recurs on Modal, the next step is to set
`OPENBLAS_CORETYPE=HASWELL` and confirm the crash rate drops to zero.
