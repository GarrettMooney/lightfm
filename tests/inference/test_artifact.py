import numpy as np
import pytest

from lightfm.inference import _artifact


def _tiny_arrays():
    rng = np.random.default_rng(42)
    return {
        "item_embeddings": rng.standard_normal((50, 8), dtype=np.float32),
        "user_embeddings": rng.standard_normal((20, 8), dtype=np.float32),
        "item_biases": np.zeros(50, dtype=np.float32),
        "user_biases": np.zeros(20, dtype=np.float32),
    }


def _tiny_metadata():
    return {
        "no_components": 8,
        "loss": "warp",
        "learning_schedule": "adagrad",
    }


def test_save_load_roundtrip(tmp_path):
    arrays = _tiny_arrays()
    metadata = _tiny_metadata()
    path = tmp_path / "model.safetensors"

    _artifact.save(path, arrays=arrays, metadata=metadata)
    loaded_arrays, loaded_metadata = _artifact.load(path, mmap=False)

    for key in arrays:
        np.testing.assert_array_equal(loaded_arrays[key], arrays[key])
    assert loaded_metadata["no_components"] == "8"
    assert loaded_metadata["loss"] == "warp"
    assert loaded_metadata["learning_schedule"] == "adagrad"
    assert loaded_metadata["format_version"] == "1"
    assert loaded_metadata["embeddings_dtype"] == "float32"


def test_mmap_returns_views(tmp_path):
    arrays = _tiny_arrays()
    path = tmp_path / "model.safetensors"
    _artifact.save(path, arrays=arrays, metadata=_tiny_metadata())

    loaded_arrays, _ = _artifact.load(path, mmap=True)
    # mmap'd arrays have their base set to a memory-mapped object; heap-loaded
    # arrays have base=None (or a numpy-internal owner).
    assert loaded_arrays["item_embeddings"].base is not None
    np.testing.assert_array_equal(loaded_arrays["item_embeddings"], arrays["item_embeddings"])
