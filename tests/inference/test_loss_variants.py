import numpy as np
import pytest
import scipy.sparse as sp

from lightfm import LightFM
from lightfm.inference.model import InferenceLightFM
from lightfm.inference import _artifact


@pytest.mark.parametrize("loss", ["logistic", "bpr", "warp", "warp-kos"])
@pytest.mark.parametrize("schedule", ["adagrad", "adadelta"])
def test_predict_parity_across_loss_and_schedule(tmp_path, loss, schedule):
    rng = np.random.default_rng(11)
    interactions = sp.csr_matrix((rng.random((40, 60)) > 0.8).astype(np.float32))
    model = LightFM(
        no_components=16,
        loss=loss,
        learning_schedule=schedule,
        random_state=3,
    )
    model.fit(interactions, epochs=2)

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

    user_ids = rng.integers(0, 40, size=1000, dtype=np.int32)
    item_ids = rng.integers(0, 60, size=1000, dtype=np.int32)
    np.testing.assert_array_equal(inf.predict(user_ids, item_ids), model.predict(user_ids, item_ids))
