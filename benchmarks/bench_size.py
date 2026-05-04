"""Axis A: artifact size comparison.

Writes both joblib and safetensors artifacts to disk and returns their byte
counts. Leaves the files on disk so subsequent axes (load, ram) can reuse
them without redumping.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from lightfm import LightFM


def run(model: LightFM, artifact_dir: Path) -> dict:
    """Dump the model two ways under `artifact_dir` and return byte counts."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = artifact_dir / "model.joblib"
    safetensors_path = artifact_dir / "model.safetensors"

    joblib.dump(model, str(joblib_path))
    model.save_for_inference(safetensors_path)

    joblib_bytes = joblib_path.stat().st_size
    safetensors_bytes = safetensors_path.stat().st_size
    ratio = safetensors_bytes / joblib_bytes if joblib_bytes else 0.0

    return {
        "joblib_bytes": joblib_bytes,
        "safetensors_bytes": safetensors_bytes,
        "reduction_ratio": ratio,
        "reduction_pct": 100.0 * (1.0 - ratio),
    }
