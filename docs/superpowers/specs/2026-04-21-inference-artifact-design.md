# Inference Artifact Design

**Date:** 2026-04-21
**Branch:** `feature/inference-artifact`
**Status:** Draft, pending spec review

## Problem

The d152 production recommender (`ea-data-lake-vertex-d152-dsw-product-rec-model`) deploys LightFM models as ~60GB `lightfm.joblib` blobs. Two concrete pain points:

1. **Artifact size.** `joblib.dump(model)` serializes the full `LightFM` instance, including training-only optimizer state (`item_embedding_gradients`, `item_embedding_momentum`, `item_bias_gradients`, `item_bias_momentum`, and user counterparts). For a fitted inference model, roughly half of the serialized bytes are dead weight.

2. **Runtime RAM.** The predictor downloads the blob into a `BytesIO`, `joblib.load`s it into `_model`, then launches `multiprocessing.Pool(processes=8)`. With fork + Python refcount touches, each worker ends up with roughly a full copy of the model. Peak RAM is ~8× the model size, driving oversized VMs and OOM risk.

This spec addresses both axes.

## Goals

- Cut inference artifact size by removing training-only optimizer state.
- Reduce runtime RAM in multi-worker prediction from ~N× model size to ~1×.
- Provide a one-shot migration path so d152 (and similar consumers) can shrink existing artifacts without retraining.
- Leave the existing `joblib.dump(model)` / `joblib.load` training-side flow untouched.
- Leave clean seams for future work: GPU backend, float16/int8 quantization, custom scoring kernels.

## Non-Goals

- float16 / int8 quantization (hook only; implementation deferred).
- GPU or custom CUDA scoring kernel.
- Changes to the d152 pipeline. Library-only scope.
- Streaming or low-memory conversion of existing joblib artifacts. The v1 converter requires enough RAM to `joblib.load` the full model.
- Deprecating `joblib.dump(LightFM)`. Training-side flow is preserved.
- Managing d152's companion pickles (`dict_user_id.pickle`, etc.). Those are pipeline artifacts, not LightFM model state.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Axes | Artifact size (A) + runtime RAM (B) | Stated by user. Latency and ergonomics are secondary wins that fall out. |
| Backward compat | Additive API + one-shot converter | d152 can shrink existing artifact without a retrain cycle. |
| API shape | New `InferenceLightFM` class | Clean separation; `fit()` absent means misuse fails loudly. Owns the mmap load path without polluting `LightFM.__init__`. |
| Memory-sharing mechanism | Memory-mapped tensors via file-backed pages | No load-time 30GB copy (unlike `multiprocessing.shared_memory`). OS page cache handles sharing across forks. No Python preprocessing GIL bottleneck (unlike threads). |
| Wire format | `safetensors` | Same mmap property as raw `.npy`, but single-file, embedded metadata, safe (no pickle), framework-neutral. Modest new dep (`safetensors`, Rust-prebuilt wheels). |
| Predict code sharing | Extract `_predict_impl` free function; both classes call it | Avoids a zombie `LightFM` instance used only for predict; clearer boundaries. |
| float16 | Design the header hook, defer implementation | Keeps v1 scope tight; v2 is a one-branch change in `_artifact.py`. |

## Architecture

Three new components living in `lightfm/inference/`:

1. **`LightFM.save_for_inference(path)`** — new method on the existing training class. Writes a slim artifact containing only inference-relevant arrays. Does not touch the existing `joblib.dump(model)` path.

2. **`InferenceLightFM`** — new class. `InferenceLightFM.load(path)` is the real entry point; returns a ready-to-predict instance backed by memory-mapped tensors. Exposes `predict()` and `predict_rank()` matching `LightFM`'s signatures. No `fit()` / `fit_partial()` — attempts to mutate fail loudly.

3. **`convert_joblib_to_inference(src, dst)`** — one-shot migration utility. Reads an existing `lightfm.joblib`, extracts inference-only arrays, writes them in the new format. Lets d152 shrink its existing artifact without retraining. Runnable as `python -m lightfm.inference.convert <src> <dst>`.

Shared internal module `lightfm/inference/_artifact.py` owns the on-disk format. `save_for_inference`, `convert_joblib_to_inference`, and `InferenceLightFM.load` all call into it — single point of format change.

Shared `_predict_impl` free function (`lightfm/inference/_predict.py`) is the scoring kernel wrapper. Both `LightFM.predict` and `InferenceLightFM.predict` call it with their respective tensor references. No behavior change to the Cython kernel itself.

## Artifact Format

Single file: `<path>/model.safetensors`.

Tensors (all float32, C-contiguous):
- `item_embeddings` — shape `[n_item_features, no_components]`
- `user_embeddings` — shape `[n_user_features, no_components]`
- `item_biases` — shape `[n_item_features]`
- `user_biases` — shape `[n_user_features]`

Header metadata (embedded in safetensors, values serialized as strings per safetensors convention):

```
format_version:    "1"
lightfm_version:   "1.20"                    (captured from lightfm.__version__ at save time)
no_components:     "<int>"
loss:              "logistic" | "bpr" | "warp" | "warp-kos"
learning_schedule: "adagrad" | "adadelta"
embeddings_dtype:  "float32"                 (float16 hook for v2)
created_at:        ISO-8601 UTC timestamp
```

`save_for_inference` calls `np.ascontiguousarray` on each array before writing to lock in cache-friendly layout.

### Why safetensors over `.npy`-per-tensor

Evaluated both. Perf is identical — both are file-backed and share pages across forks. Safetensors wins on:

- Single-file atomic upload / versioning (vs. a directory of blobs).
- Embedded metadata, eliminating a separate `manifest.json` that could desynchronize from the weights.
- Safe (no pickle) — same as `.npy`.
- Framework-neutral: future GPU/torch/JAX consumers read the same file without re-serialization.

Cost: new dependency (`safetensors`, ~5MB, Rust-prebuilt wheels on linux/macOS/windows × x86_64/arm64).

### Explicitly not in v1 format

- Compression (incompatible with mmap's file-backed shared pages).
- float16 / int8 (header field reserved; loader asserts `"float32"`).
- Sharding a single model across multiple files (not needed at 30GB scale).

## `InferenceLightFM` API

```python
class InferenceLightFM:
    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> "InferenceLightFM": ...

    def predict(
        self,
        user_ids: np.ndarray | int,
        item_ids: np.ndarray,
        item_features: csr_matrix | None = None,
        user_features: csr_matrix | None = None,
        num_threads: int = 1,
    ) -> np.ndarray: ...

    def predict_rank(
        self,
        test_interactions: csr_matrix,
        train_interactions: csr_matrix | None = None,
        item_features: csr_matrix | None = None,
        user_features: csr_matrix | None = None,
        num_threads: int = 1,
        check_intersections: bool = True,
    ) -> csr_matrix: ...

    # Introspection
    @property
    def no_components(self) -> int: ...
    @property
    def loss(self) -> str: ...
    @property
    def item_embeddings(self) -> np.ndarray: ...  # read-only view onto mmap'd memory
    @property
    def user_embeddings(self) -> np.ndarray: ...
    @property
    def item_biases(self) -> np.ndarray: ...
    @property
    def user_biases(self) -> np.ndarray: ...
```

### Load semantics

- `mmap=True` (default): arrays are views onto file-backed memory. Pages are shared across forks via the OS page cache. This is the core perf property.
- `mmap=False`: arrays are read into heap memory. Escape hatch for tests and environments where mmap misbehaves.

`preload` / `madvise` is deliberately **not** in v1 (see v2 hooks). User's primary axes are artifact size and runtime RAM; cold-start latency was explicitly deprioritized.

### Validation at load

- `format_version` must be in `SUPPORTED_VERSIONS` (v1 → `{"1"}`). Unknown version is a hard refusal, not a warning.
- Required tensors must be present.
- Declared dtype must match actual tensor dtype.
- Shapes must be self-consistent: `item_biases.shape[0] == item_embeddings.shape[0]`, same for user side, and both embeddings' second dim must equal `no_components`.

### Omitted by design

No `fit`, `fit_partial`, `_reset_state`, `_initialize`. No `*_gradients`, `*_momentum` attributes. No `save` (loaded artifacts are immutable; re-saving would be a code smell).

### Thread safety

Read-only mmap views are safe for concurrent reads. The Cython predict kernel releases the GIL in its inner loop. No locks needed.

## `save_for_inference` on `LightFM`

```python
def save_for_inference(self, path: str | Path) -> None: ...
```

Calls `self._check_initialized()`, then `_artifact.save(path, ...)` with the fitted inference arrays and metadata pulled from `self` (`no_components`, `loss`, `learning_schedule`, `lightfm_version`).

Does not overlap with existing `joblib.dump(model)` flows — separate path, different on-disk format.

## Converter

```python
# lightfm/inference/convert.py
def convert_joblib_to_inference(
    src: str | Path,
    dst: str | Path,
    *,
    storage_options: dict | None = None,
) -> None: ...
```

CLI: `python -m lightfm.inference.convert <src> <dst>`.

Steps:

1. Resolve `src` via `fsspec` (supports `gs://`, local, etc.). Same for `dst`.
2. `joblib.load(src)` → returns a `LightFM` instance. This step requires enough RAM to materialize the full fitted model. Documented as such.
3. Validate: loaded object is a fitted `LightFM`.
4. Extract inference-only arrays (no copy — references only).
5. Call `_artifact.save(dst, ...)` with metadata sourced from the instance.
6. Log size reduction (src → dst bytes, percentage).

### Subtlety: joblib forward-compatibility

Existing `lightfm.joblib` files were dumped with whatever Python/NumPy versions were in d152 at training time. The converter must successfully load them in a newer environment.

**Concrete CI compat matrix (relevant given the recently-merged `feature/numpy2-compat`):**
- Fixture 1: tiny model dumped under `numpy < 2.0`, `joblib < 1.3`. Converter runs under numpy 2.x in CI. This is the realistic d152 scenario — their current blob was produced under numpy 1.x.
- Fixture 2: tiny model dumped under current environment. Baseline round-trip.

Both must load → convert → predict-parity successfully.

### Out of scope for converter

- Converting companion pickles (`dict_user_id.pickle`, `dict_item_id.pickle`, `interaction_matrix.pickle`). Those are d152 pipeline artifacts, not LightFM model state.
- Streaming conversion. If the converter host cannot allocate full-model RAM, that's a v2 concern.

## Error Handling

Three boundaries; everywhere else we trust internal invariants.

### `_artifact.load()`

Exception class: `ArtifactLoadError(Exception)`.

- Missing / unreadable file → include path.
- Header parses but no `format_version` → `"missing format_version — is this a lightfm artifact?"`.
- Unknown `format_version` → `f"artifact is version {v}, this lightfm ({__version__}) supports {SUPPORTED_VERSIONS}"`.
- Required tensors missing → list each missing tensor.
- Dtype mismatch with declared dtype → explicit error at load.
- Shape inconsistency → explicit error at load.

### `InferenceLightFM.predict()`

Reuses the shape/type checks already present in `LightFM.predict`, extracted into a shared `_validate_predict_inputs` helper. Identical error surface as the training class — no behavior change for callers migrating over.

### `convert_joblib_to_inference()`

Exception class: `ConversionError(Exception)`.

- `joblib.load` fails (version skew, corruption) → wrap original exception with path context.
- Loaded object is not a `LightFM` instance → `f"expected LightFM, got {type}"`.
- Loaded `LightFM` is not fitted → `"source model is not fitted"`.
- Write to `dst` fails → propagate with path context.

### Non-error conditions

- Empty `item_ids` to predict → empty array out (matches `LightFM`).
- `mmap=False` → works, just no sharing.
- Missing optional header fields (e.g., `lightfm_version`) → tolerate; `format_version` is the compatibility gate.

## Testing

Ordered by load-bearing importance.

### 1. Forked workers share pages (the core perf claim)

**This is the whole point of the project.** Load an `InferenceLightFM` with a medium-sized model (~500MB of embeddings), fork N=4 workers via `multiprocessing.Pool`, have each call `predict` on a batch.

**Linux (quantitative pass/fail):** use PSS (Proportional Set Size) via `/proc/<pid>/smaps_rollup` — PSS correctly accounts for shared pages (each shared page counts as `size / num_sharers` in each process). Summing PSS across all worker children + parent gives a true memory footprint. RSS double-counts shared pages and cannot distinguish shared-vs-copied; the test must use PSS on Linux.

- Assertion: `sum(PSS) < 1.5 × single_process_loaded_baseline`. The `1.5×` tolerance covers per-worker Python interpreter overhead, non-shared stack/heap, and minor variance. A non-mmap regression would show `sum(PSS) ≈ N × model_size`, well above the threshold.

**macOS (coarse sanity check):** PSS isn't available via equivalent APIs. Fall back to RSS with a looser check — `max(worker RSS) ≈ model_size` (individual workers don't exceed single-process baseline). Not as strong a signal but catches catastrophic regressions.

If the Linux test ever regresses, the project has failed silently — it must live in the core test suite. The macOS check runs as a smoke test only.

### 2. Round-trip correctness via converter

Train a small `LightFM` model → `joblib.dump` → `convert_joblib_to_inference` → `InferenceLightFM.load` → `predict`. Compare against `LightFM.predict` on the original model. Must be bit-identical (same float32 kernel, same arrays).

### 3. Round-trip correctness via `save_for_inference`

Train small → `save_for_inference` → `load` → `predict`. Compare against `LightFM.predict`. Bit-identical.

### 4. Predict parity across loss / schedule combinations

For each `loss ∈ {logistic, bpr, warp, warp-kos}` and `learning_schedule ∈ {adagrad, adadelta}`: train tiny → save → load → assert `InferenceLightFM.predict` matches `LightFM.predict` across 1000 random (user, item) pairs. Catches bugs in the `_predict_impl` extraction.

### 5. Sparse features path

Same round-trip with `user_features` and `item_features` passed to predict. Exercises the different Cython code path.

### 6. `LightFM.predict` behavior preservation under `_predict_impl` extraction

Run the existing `tests/` suite before the `_predict_impl` refactor and again after. Full diff must be empty. The risk is subtle — e.g., the `user_ids` int-vs-array coercion at `lightfm.py:832-836` must land in the shared helper (not stay in `LightFM.predict`), or `InferenceLightFM.predict(0, [...])` silently breaks while the training-class tests still pass.

### 7. Error paths

Each error case from "Error Handling" has an explicit test: corrupt header, missing tensor, wrong dtype, shape mismatch, unknown format version, missing file, unfitted model in converter, non-LightFM object in converter.

### 8. Cross-platform smoke

Load + predict on Linux and macOS in CI. Validates safetensors wheel availability. Windows optional.

### Not in v1 test matrix

- GPU / float16 tests (deferred with those features).
- Large-model (60GB) tests — infeasible in CI; document as a manual smoke test procedure.
- Fuzzing the safetensors header — trust safetensors' own tests.

## Files Added / Modified

### Added under `lightfm/inference/`

- `__init__.py` — re-exports `InferenceLightFM`, `ArtifactLoadError`, `ConversionError`.
- `_artifact.py` — `save()`, `load()`, `SUPPORTED_VERSIONS`. Wraps `safetensors.numpy`.
- `_predict.py` — `_predict_impl(...)`, `_validate_predict_inputs(...)`.
- `model.py` — `InferenceLightFM` class.
- `convert.py` — `convert_joblib_to_inference`, `__main__`.

### Added under `tests/inference/`

- `test_fork_sharing.py` — the load-bearing RSS test.
- `test_roundtrip.py` — save → load → predict parity.
- `test_convert.py` — joblib → safetensors conversion.
- `test_loss_variants.py` — parity across losses / schedules.
- `test_sparse_features.py` — feature-matrix path.
- `test_errors.py` — error-path coverage.

### Modified

- `lightfm/lightfm.py` — add `LightFM.save_for_inference(path)`. `LightFM.predict` delegates to shared `_predict_impl` (behavior-preserving; verified by existing test suite).
- `lightfm/__init__.py` — re-export `InferenceLightFM` for ergonomics.
- `pyproject.toml` — add `safetensors>=0.4` to dependencies.

## v2 Hooks

| Feature | Hook reserved in v1 | v2 implementation cost |
|---|---|---|
| float16 embeddings | `embeddings_dtype` header field (v1 writes `"float32"`, loader asserts it) | Extend `_artifact.load` dtype branch; add `save_for_inference(dtype=)` flag |
| GPU backend | `_artifact.py` is single format boundary; safetensors reads into torch/JAX natively | New class `GpuInferenceLightFM` (or `device=` param); same artifact |
| int8 quantization | `embeddings_dtype` + room for `scales` / `zero_points` tensors in header | Additive; `format_version` bump ensures v1 loaders refuse v2 artifacts cleanly |
| Streaming converter | `convert.py` is a separate module | Additive function alongside existing |
| `preload=True` / `madvise(MADV_WILLNEED)` | `InferenceLightFM.load()` kwargs extensible | Additive kwarg; conditional on `sys.platform` for non-Linux fallback |

## Expected Outcomes

- **Artifact size:** ~60GB → ~30GB (training state removed). Further ~2× reduction possible with v2 float16.
- **Runtime RAM on predictor:** ~8× model size → ~1× model size (file-backed shared pages across fork).
- **Cold-start latency:** faster — single mmap, no 60GB `BytesIO` assembly, no full-object `joblib.load`.

## Operational Rollout (d152)

Out of scope for this branch but documented for the consuming team:

1. Run converter once: `python -m lightfm.inference.convert gs://.../lightfm.joblib gs://.../model.safetensors`.
2. Point predictor at new artifact URI.
3. In `d152_prediction.py`: swap `joblib.load(BytesIO(model_bytes))` for `InferenceLightFM.load(path)`. Delete `download_model_in_chunks` — obsolete with mmap.
4. Keep old `lightfm.joblib` around until new pipeline is verified in prod, then delete.
