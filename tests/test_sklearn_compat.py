"""Tests for scikit-learn compatibility.

LightFM inherits from sklearn.base.BaseEstimator to support sklearn 1.6+
which requires the __sklearn_tags__ method for cross-validation and
other utilities.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone

from lightfm import LightFM


def test_clone():
    """Test that sklearn.base.clone() works with LightFM."""
    model = LightFM(
        no_components=20,
        learning_rate=0.01,
        loss="warp",
        random_state=42,
    )

    cloned = clone(model)

    # Verify parameters are copied
    assert cloned.no_components == 20
    assert cloned.learning_rate == 0.01
    assert cloned.loss == "warp"
    # random_state is converted to RandomState object internally
    assert cloned.random_state is not None

    # Verify it's a different object
    assert cloned is not model


def test_clone_fitted_model():
    """Test that cloning a fitted model produces an unfitted clone."""
    no_users, no_items = 10, 20
    interactions = sp.coo_matrix(
        (np.ones(5), ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])),
        shape=(no_users, no_items),
        dtype=np.float32,
    )

    model = LightFM(no_components=10, random_state=42)
    model.fit(interactions, epochs=1)

    # Model should have embeddings after fitting
    assert model.item_embeddings is not None

    cloned = clone(model)

    # Cloned model should not have embeddings (unfitted)
    assert cloned.item_embeddings is None
    assert cloned.no_components == 10
    # random_state is converted to RandomState object internally
    assert cloned.random_state is not None


def test_get_params():
    """Test that get_params returns all constructor parameters."""
    model = LightFM(
        no_components=30,
        k=10,
        n=15,
        learning_schedule="adadelta",
        loss="bpr",
        learning_rate=0.05,
        rho=0.9,
        epsilon=1e-5,
        item_alpha=1e-4,
        user_alpha=1e-4,
        max_sampled=20,
        random_state=123,
    )

    params = model.get_params()

    assert params["no_components"] == 30
    assert params["k"] == 10
    assert params["n"] == 15
    assert params["learning_schedule"] == "adadelta"
    assert params["loss"] == "bpr"
    assert params["learning_rate"] == 0.05
    assert params["rho"] == 0.9
    assert params["epsilon"] == 1e-5
    assert params["item_alpha"] == 1e-4
    assert params["user_alpha"] == 1e-4
    assert params["max_sampled"] == 20
    # random_state is converted to RandomState object internally
    assert params["random_state"] is not None


def test_set_params():
    """Test that set_params correctly updates parameters."""
    model = LightFM()

    model.set_params(no_components=50, loss="warp-kos", learning_rate=0.1)

    assert model.no_components == 50
    assert model.loss == "warp-kos"
    assert model.learning_rate == 0.1


def test_sklearn_tags():
    """Test that __sklearn_tags__ is available (required by sklearn 1.6+)."""
    model = LightFM()

    # BaseEstimator provides __sklearn_tags__ in sklearn 1.6+
    assert hasattr(model, "__sklearn_tags__")

    # Should be callable and return tags
    tags = model.__sklearn_tags__()
    assert tags is not None
