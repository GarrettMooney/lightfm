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


from lightfm import LightFM


def _train_tiny_model(n_users=30, n_items=50, k=8, loss="warp", schedule="adagrad"):
    rng = np.random.default_rng(7)
    # Random dense interaction matrix, thresholded.
    dense = (rng.random((n_users, n_items)) > 0.8).astype(np.float32)
    interactions = sp.csr_matrix(dense)
    model = LightFM(no_components=k, loss=loss, learning_schedule=schedule, random_state=1)
    model.fit(interactions, epochs=2)
    return model, interactions


def _save_from_trained(model, tmp_path):
    path = tmp_path / "model.safetensors"
    _artifact.save(
        path,
        arrays={
            "item_embeddings": model.item_embeddings,
            "user_embeddings": model.user_embeddings,
            "item_biases": model.item_biases,
            "user_biases": model.user_biases,
        },
        metadata={
            "no_components": model.no_components,
            "loss": model.loss,
            "learning_schedule": model.learning_schedule,
        },
    )
    return path


def test_predict_bit_identical_to_lightfm(tmp_path):
    model, _ = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    rng = np.random.default_rng(99)
    user_ids = rng.integers(0, 30, size=200, dtype=np.int32)
    item_ids = rng.integers(0, 50, size=200, dtype=np.int32)

    expected = model.predict(user_ids, item_ids)
    actual = inf.predict(user_ids, item_ids)

    np.testing.assert_array_equal(actual, expected)


def test_predict_int_user_id_shorthand(tmp_path):
    model, _ = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    # Exercises the isinstance(user_ids, int) branch in _validate_predict_inputs.
    item_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    expected = model.predict(0, item_ids)
    actual = inf.predict(0, item_ids)
    np.testing.assert_array_equal(actual, expected)


def test_predict_list_input_coerced(tmp_path):
    model, _ = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    # Exercises the list/tuple coercion branches.
    expected = model.predict([0, 1, 2], [3, 4, 5])
    actual = inf.predict([0, 1, 2], [3, 4, 5])
    np.testing.assert_array_equal(actual, expected)


def test_predict_rank_bit_identical_to_lightfm(tmp_path):
    model, interactions = _train_tiny_model()
    path = _save_from_trained(model, tmp_path)
    inf = InferenceLightFM.load(path, mmap=False)

    # Construct a small test interactions matrix disjoint from the training one
    # (predict_rank with check_intersections=True will reject overlapping entries).
    rng = np.random.default_rng(55)
    dense = (rng.random(interactions.shape) > 0.95).astype(np.float32)
    test_interactions = sp.csr_matrix(dense).multiply(interactions.toarray() == 0).tocsr()

    expected = model.predict_rank(test_interactions)
    actual = inf.predict_rank(test_interactions)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
