import numpy as np
import scipy.sparse as sp

from lightfm import LightFM
from lightfm.inference import _artifact
from lightfm.inference.model import InferenceLightFM


def test_predict_with_item_and_user_features(tmp_path):
    rng = np.random.default_rng(13)
    n_users, n_items = 30, 50
    n_user_features, n_item_features = 10, 15

    user_features = sp.csr_matrix((rng.random((n_users, n_user_features)) > 0.5).astype(np.float32))
    item_features = sp.csr_matrix((rng.random((n_items, n_item_features)) > 0.5).astype(np.float32))
    interactions = sp.csr_matrix((rng.random((n_users, n_items)) > 0.8).astype(np.float32))

    model = LightFM(no_components=8, loss="warp", random_state=7)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=2)

    path = tmp_path / "m.safetensors"
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
    inf = InferenceLightFM.load(path, mmap=False)

    user_ids = rng.integers(0, n_users, size=100, dtype=np.int32)
    item_ids = rng.integers(0, n_items, size=100, dtype=np.int32)

    expected = model.predict(user_ids, item_ids, user_features=user_features, item_features=item_features)
    actual = inf.predict(user_ids, item_ids, user_features=user_features, item_features=item_features)
    np.testing.assert_array_equal(actual, expected)
