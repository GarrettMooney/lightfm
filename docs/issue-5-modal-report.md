# Draft: Modal support report

Copy this into a Modal support email or Slack thread. The bug itself
lives upstream in NumPy/OpenBLAS, but Modal users are a primary
exposure surface and Modal is best-positioned to (a) document the
issue, (b) confirm whether their fleet exposes CPUID flags the
underlying host cannot fully execute, and (c) answer whether host
class can be pinned for reproducibility.

---

**Subject:** Intermittent `SIGILL` (exit 132) in NumPy-bundled OpenBLAS
on default Modal CPU class — AVX-512 dispatch mismatch

## Summary

On 2026-04-22 we observed `LightFM.predict()` exiting with **signal 4
(SIGILL) / exit code 132** on roughly **50 % of container attempts**
under the default Modal CPU class (`debian_slim`, 2 vCPU, 2 GB RAM).
Modal's 8-attempt retry policy masked most of these, but ~10 % of trials
exhausted all retries and failed permanently.

We have since pinned the root cause to NumPy's bundled OpenBLAS
(`libscipy_openblas64_*.so`), whose runtime CPU dispatcher selects an
**AVX-512 (Skylake-X) DGEMM kernel** on hosts whose advertised CPUID
flags include AVX-512 support that the actual instruction stream
cannot execute. The first `vbroadcastsd %xmm0,%zmm25` instruction
delivers SIGILL.

This is filed with Modal because:

1. Modal's heterogeneous CPU fleet is the trigger condition — bare-metal
   single-CPU deployments would never see this.
2. The intermittency pattern (~50 % per attempt, varying by container
   placement) suggests a specific subset of hosts in your fleet.
3. The crash bypasses Python's exception machinery (kernel-delivered
   SIGILL with no traceback), so users see only "Runner failed with
   exit code: 132" in Modal logs and have no actionable signal.
4. Workarounds we found (`OPENBLAS_CORETYPE=HASWELL`,
   `OPENBLAS_NUM_THREADS=1`) are not Modal-discoverable; documenting
   them on Modal would help affected users.

## Reproducer (the original failing case)

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["modal"]
# ///
import modal

app = modal.App("lightfm-sigill-repro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++", "git")
    .uv_pip_install("numpy>=1.26", "scipy>=1.12")
    .run_commands("pip install git+https://github.com/GarrettMooney/lightfm.git")
)

def _fit_and_predict():
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
    return model.predict(user_ids, item_ids, num_threads=1)[:3].tolist()

@app.function(image=image, cpu=2, memory=2*1024,
              retries=modal.Retries(max_retries=0))
def run(trial: int):
    return {"trial": trial, "scores_head": _fit_and_predict()}

@app.local_entrypoint()
def main():
    for i in range(10):
        try: print(i, run.remote(i))
        except Exception as e: print(i, "CRASH", e)
```

## What we verified out-of-fleet

We could not reproduce on Modal in our investigation window (70 clean
trials across two probe shapes), suggesting fleet rotation. We
reproduced the underlying mechanism on a controlled GCE
`n1-standard-2` (Intel Xeon, AVX2 only, no AVX-512) by forcing the
problematic kernel:

```
OPENBLAS_CORETYPE=SKYLAKEX python repro.py   # 30/30 SIGILL
                          python repro.py    #  0/30 SIGILL
```

Backtrace under `gdb`:

```
Thread 1 received signal SIGILL, Illegal instruction.
0x... in dgemm_small_kernel_b0_nn_SKYLAKEX ()
   from .../numpy.libs/libscipy_openblas64_-32a4b2a6.so

#0  dgemm_small_kernel_b0_nn_SKYLAKEX     in libscipy_openblas64_-32a4b2a6.so
#1  scipy_cblas_dgemm64_                  in libscipy_openblas64_-32a4b2a6.so
#2  DOUBLE_matmul_matrixmatrix.isra.0     in numpy/_core/_multiarray_umath...
#3  DOUBLE_matmul                         in numpy/_core/_multiarray_umath...
... CPython eval loop ...

=> vbroadcastsd %xmm0,%zmm25
```

## Questions for Modal

1. **Fleet composition.** What CPU classes are in the default
   container pool (`debian_slim`, no CPU class specified) for our
   account / region as of 2026-04-22? Did anything change in fleet
   composition between morning and evening on that date?

2. **CPUID exposure.** Do Modal's container CPUs ever advertise
   feature flags (specifically `avx512f`, `avx512vl`, etc.) that the
   underlying physical host cannot fully execute, or is CPU-feature
   exposure always accurate to host capability? OpenBLAS's dispatcher
   trusts CPUID; if guests see flags the host can't deliver, this
   exact failure mode is inevitable for any user of the NumPy wheel.

3. **Pinning host class.** Is there a documented way for users to
   request a homogeneous CPU class (e.g. "AVX2-only" or
   "AVX-512-confirmed") for reproducibility-sensitive workloads?
   Today users cannot detect or steer this from the Modal SDK.

4. **Run history.** Could you correlate the failed
   `lightfm-sigill-repro` runs in our account history (afternoon of
   2026-04-22) with the specific host hardware they landed on? That
   would confirm or refute the "specific host subset" hypothesis
   without our needing to wait for the bug to recur naturally.

5. **Documentation surface.** Would Modal consider adding a brief
   note to your "Containers and images" docs about the OpenBLAS
   dispatcher pitfall and the `OPENBLAS_CORETYPE` workaround? It
   affects every Modal user who imports NumPy.

## Suggested workarounds for affected users (today)

Set on the image, not in user code:

```python
image = modal.Image.debian_slim(...).env({
    "OPENBLAS_CORETYPE": "HASWELL",       # pin to AVX2 baseline
    "OPENBLAS_NUM_THREADS": "1",          # reduce small-matrix dispatch
})
```

Or co-install `faiss-cpu` (without importing it) — its bundled
OpenBLAS wins the `dlopen` resolution and bypasses the dispatcher.

## Affected user / contact

(fill in account / email / Modal workspace before sending)

## References

- Bug report: https://github.com/GarrettMooney/lightfm/issues/5
- Investigation record:
  https://github.com/GarrettMooney/lightfm/blob/master/docs/issue-5-sigill-forensics.md
