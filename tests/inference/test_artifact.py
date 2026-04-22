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


def test_missing_file(tmp_path):
    with pytest.raises(_artifact.ArtifactLoadError, match="artifact not found"):
        _artifact.load(tmp_path / "does-not-exist.safetensors")


def test_missing_format_version(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    save_file(_tiny_arrays(), str(path), metadata={"loss": "warp"})
    with pytest.raises(_artifact.ArtifactLoadError, match="missing format_version"):
        _artifact.load(path, mmap=False)


def test_unknown_format_version(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    save_file(_tiny_arrays(), str(path), metadata={
        "format_version": "999",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="version '999'"):
        _artifact.load(path, mmap=False)


def test_missing_required_tensor(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    arrays = _tiny_arrays()
    del arrays["item_biases"]
    save_file(arrays, str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="missing required tensors.*item_biases"):
        _artifact.load(path, mmap=False)


def test_shape_inconsistency(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    arrays = _tiny_arrays()
    # item_biases length != item_embeddings rows
    arrays["item_biases"] = np.zeros(7, dtype=np.float32)
    save_file(arrays, str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="item_biases.*does not match"):
        _artifact.load(path, mmap=False)


def test_dtype_mismatch(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    arrays = _tiny_arrays()
    arrays["item_embeddings"] = arrays["item_embeddings"].astype(np.float64)
    save_file(arrays, str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float32",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="dtype float64.*expected float32"):
        _artifact.load(path, mmap=False)


def test_unsupported_declared_dtype(tmp_path):
    path = tmp_path / "model.safetensors"
    from safetensors.numpy import save_file
    save_file(_tiny_arrays(), str(path), metadata={
        "format_version": "1",
        "no_components": "8",
        "loss": "warp",
        "learning_schedule": "adagrad",
        "embeddings_dtype": "float16",
    })
    with pytest.raises(_artifact.ArtifactLoadError, match="embeddings_dtype='float16'"):
        _artifact.load(path, mmap=False)
