import json
from pathlib import Path

import pytest

from benchmarks.results import print_table, write_json


@pytest.fixture
def sample_results():
    return {
        "size": {
            "joblib_bytes": 1_000_000,
            "safetensors_bytes": 500_000,
            "reduction_ratio": 0.5,
            "reduction_pct": 50.0,
        },
        "load": {
            "joblib": {"min": 1.0, "median": 1.1, "max": 1.2, "repeats": 5},
            "inference_heap": {"min": 0.5, "median": 0.55, "max": 0.6, "repeats": 5},
            "inference_mmap": {"min": 0.001, "median": 0.002, "max": 0.003, "repeats": 5},
        },
        "predict": [
            {
                "batch_size": 1_000,
                "lightfm": {"min": 0.01, "median": 0.011, "max": 0.012, "per_sec": 90909},
                "inference": {"min": 0.01, "median": 0.011, "max": 0.012, "per_sec": 90909},
                "ratio": 1.00,
            },
        ],
        "ram": {
            "platform": "darwin",
            "measurement": "rss",
            "n_workers": 4,
            "model_bytes": 500_000,
            "variant_a": {"per_worker": [400_000, 410_000, 390_000, 405_000], "total": 1_605_000},
            "variant_b": {"per_worker": [200_000, 205_000, 195_000, 202_000], "total": 802_000},
            "sharing_ratio": 2.0,
            "caveats": ["test caveat"],
        },
    }


@pytest.fixture
def sample_metadata():
    return {
        "timestamp": "2026-04-23T14:32:17Z",
        "size": "custom",
        "config": {"n_users": 10, "n_items": 20, "no_components": 4},
        "host": {"platform": "darwin", "python": "3.13", "cpu_count": 8},
        "lightfm_version": "1.20",
        "n_workers": 4,
    }


def test_write_json_roundtrip(tmp_path, sample_results, sample_metadata):
    path = tmp_path / "out.json"
    write_json(sample_results, sample_metadata, path)
    loaded = json.loads(path.read_text())
    assert loaded["metadata"]["size"] == "custom"
    assert loaded["axes"]["size"]["reduction_pct"] == 50.0
    assert loaded["axes"]["ram"]["variant_b"]["total"] == 802_000


def test_print_table_does_not_raise(capsys, sample_results, sample_metadata):
    print_table(sample_results, sample_metadata)
    captured = capsys.readouterr()
    assert "Artifact Size" in captured.out
    assert "Load Time" in captured.out
    assert "Predict Throughput" in captured.out
    assert "Multi-Process RAM" in captured.out


def test_write_json_handles_skipped_axes(tmp_path, sample_metadata):
    partial = {"size": {"joblib_bytes": 100, "safetensors_bytes": 50, "reduction_ratio": 0.5, "reduction_pct": 50.0}}
    path = tmp_path / "partial.json"
    write_json(partial, sample_metadata, path)
    loaded = json.loads(path.read_text())
    assert "size" in loaded["axes"]
    assert "load" not in loaded["axes"]


def test_print_table_handles_skipped_axes(capsys, sample_metadata):
    partial = {"size": {"joblib_bytes": 100, "safetensors_bytes": 50, "reduction_ratio": 0.5, "reduction_pct": 50.0}}
    print_table(partial, sample_metadata)
    captured = capsys.readouterr()
    assert "Artifact Size" in captured.out
    assert "Load Time" not in captured.out
