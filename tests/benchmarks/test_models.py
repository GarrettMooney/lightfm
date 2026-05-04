import numpy as np
import pytest

from benchmarks.models import SIZE_PRESETS, make_fitted_model
from lightfm import LightFM


def test_size_presets_shape():
    """Presets have the expected keys and dimensions."""
    assert set(SIZE_PRESETS) == {"tiny", "medium", "large"}
    for preset in ("tiny", "medium", "large"):
        cfg = SIZE_PRESETS[preset]
        assert "n_users" in cfg and "n_items" in cfg and "no_components" in cfg


def test_make_fitted_model_is_a_lightfm():
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    assert isinstance(model, LightFM)


def test_make_fitted_model_arrays_are_initialized():
    """All 12 arrays that FastLightFM needs are present and float32."""
    model = make_fitted_model(n_users=50, n_items=100, no_components=4)
    for attr in (
        "item_embeddings", "item_embedding_gradients", "item_embedding_momentum",
        "item_biases", "item_bias_gradients", "item_bias_momentum",
        "user_embeddings", "user_embedding_gradients", "user_embedding_momentum",
        "user_biases", "user_bias_gradients", "user_bias_momentum",
    ):
        arr = getattr(model, attr)
        assert arr is not None, attr
        assert arr.dtype == np.float32, attr


def test_make_fitted_model_predict_works():
    """Predict runs without error — the synthetic model is a valid LightFM."""
    model = make_fitted_model(n_users=50, n_items=100, no_components=4, seed=0)
    user_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    item_ids = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    scores = model.predict(user_ids, item_ids)
    assert scores.shape == (5,)
    assert scores.dtype == np.float32


def test_make_fitted_model_deterministic_with_seed():
    m1 = make_fitted_model(n_users=10, n_items=20, no_components=4, seed=42)
    m2 = make_fitted_model(n_users=10, n_items=20, no_components=4, seed=42)
    np.testing.assert_array_equal(m1.item_embeddings, m2.item_embeddings)
    np.testing.assert_array_equal(m1.user_embeddings, m2.user_embeddings)
