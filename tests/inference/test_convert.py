import joblib
import numpy as np
import pytest
import scipy.sparse as sp

from lightfm import LightFM
from lightfm.inference import ConversionError
from lightfm.inference.convert import convert_joblib_to_inference
from lightfm.inference.model import InferenceLightFM


def _train_and_dump(tmp_path, loss="warp"):
    rng = np.random.default_rng(5)
    interactions = sp.csr_matrix((rng.random((30, 50)) > 0.8).astype(np.float32))
    model = LightFM(no_components=8, loss=loss, random_state=2)
    model.fit(interactions, epochs=2)
    src = tmp_path / "lightfm.joblib"
    joblib.dump(model, src)
    return src, model


def test_convert_roundtrip_parity(tmp_path):
    src, trained = _train_and_dump(tmp_path)
    dst = tmp_path / "model.safetensors"

    convert_joblib_to_inference(src, dst)

    inf = InferenceLightFM.load(dst, mmap=False)
    assert inf.no_components == trained.no_components
    assert inf.loss == trained.loss

    rng = np.random.default_rng(321)
    user_ids = rng.integers(0, 30, size=50, dtype=np.int32)
    item_ids = rng.integers(0, 50, size=50, dtype=np.int32)
    np.testing.assert_array_equal(inf.predict(user_ids, item_ids), trained.predict(user_ids, item_ids))


def test_convert_source_missing(tmp_path):
    with pytest.raises(ConversionError):
        convert_joblib_to_inference(tmp_path / "nope.joblib", tmp_path / "out.safetensors")


def test_convert_source_not_lightfm(tmp_path):
    src = tmp_path / "not_lightfm.joblib"
    joblib.dump({"not": "a model"}, src)
    with pytest.raises(ConversionError, match="expected LightFM"):
        convert_joblib_to_inference(src, tmp_path / "out.safetensors")


def test_convert_source_not_fitted(tmp_path):
    src = tmp_path / "unfit.joblib"
    joblib.dump(LightFM(), src)
    with pytest.raises(ConversionError, match="not fitted"):
        convert_joblib_to_inference(src, tmp_path / "out.safetensors")


def test_convert_artifact_is_smaller(tmp_path):
    src, _ = _train_and_dump(tmp_path)
    dst = tmp_path / "model.safetensors"
    convert_joblib_to_inference(src, dst)
    assert dst.stat().st_size < src.stat().st_size
