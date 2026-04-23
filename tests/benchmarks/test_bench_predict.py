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
