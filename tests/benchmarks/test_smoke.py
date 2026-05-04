"""Smoke test for the benchmark CLI.

Marked `slow` — excluded from default runs via `pytest -m "not slow"`.
Runs the full four-axis pipeline at --size tiny and asserts a valid JSON
appears. Guards the benchmark machinery itself (no numeric assertions).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_benchmarks_cli_smoke(tmp_path):
    output_dir = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmarks",
            "--size", "tiny",
            "--n-workers", "2",
            "--repeats-load", "2",
            "--repeats-predict", "2",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

    # The CLI writes exactly one sidecar JSON matching the size.
    json_files = list(output_dir.glob("*-tiny.json"))
    assert len(json_files) == 1, f"expected one JSON, got {json_files}"
    payload = json.loads(json_files[0].read_text())

    assert payload["metadata"]["size"] == "tiny"
    assert "size" in payload["axes"]
    assert "load" in payload["axes"]
    assert "predict" in payload["axes"]
    assert "ram" in payload["axes"]
    # Basic numeric sanity
    assert payload["axes"]["size"]["joblib_bytes"] > 0
    assert payload["axes"]["size"]["safetensors_bytes"] > 0
    # Axis D emits caveats — spec calls this out explicitly
    assert isinstance(payload["axes"]["ram"].get("caveats"), list)
    assert len(payload["axes"]["ram"]["caveats"]) > 0
