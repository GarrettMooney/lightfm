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
