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
