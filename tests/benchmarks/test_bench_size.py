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
