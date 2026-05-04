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
