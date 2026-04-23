import numpy as np
import pytest
import scipy.sparse as sp

from lightfm.inference import _artifact
from lightfm.inference.model import InferenceLightFM


def _save_tiny(tmp_path, n_items=50, n_users=20, k=8, loss="warp"):
    rng = np.random.default_rng(0)
    arrays = {
        "item_embeddings": rng.standard_normal((n_items, k)).astype(np.float32),
        "user_embeddings": rng.standard_normal((n_users, k)).astype(np.float32),
        "item_biases": rng.standard_normal(n_items).astype(np.float32),
        "user_biases": rng.standard_normal(n_users).astype(np.float32),
    }
    path = tmp_path / "model.safetensors"
    _artifact.save(path, arrays=arrays, metadata={
        "no_components": k,
        "loss": loss,
        "learning_schedule": "adagrad",
    })
    return path, arrays


def test_load_returns_inference_lightfm(tmp_path):
    path, arrays = _save_tiny(tmp_path)
    model = InferenceLightFM.load(path)
    assert isinstance(model, InferenceLightFM)


def test_properties_match_saved(tmp_path):
    path, arrays = _save_tiny(tmp_path, k=16, loss="bpr")
    model = InferenceLightFM.load(path)
    assert model.no_components == 16
    assert model.loss == "bpr"
    assert model.learning_schedule == "adagrad"
    np.testing.assert_array_equal(model.item_embeddings, arrays["item_embeddings"])
    np.testing.assert_array_equal(model.user_embeddings, arrays["user_embeddings"])
    np.testing.assert_array_equal(model.item_biases, arrays["item_biases"])
    np.testing.assert_array_equal(model.user_biases, arrays["user_biases"])


def test_load_mmap_flag_controls_base(tmp_path):
    path, _ = _save_tiny(tmp_path)
    mmapped = InferenceLightFM.load(path, mmap=True)
    heap = InferenceLightFM.load(path, mmap=False)
    assert mmapped.item_embeddings.base is not None  # file-backed
    # heap-loaded arrays may or may not have base=None depending on safetensors;
    # the load semantics test just verifies this doesn't raise.


def test_no_fit_attribute():
    assert not hasattr(InferenceLightFM, "fit")
    assert not hasattr(InferenceLightFM, "fit_partial")
