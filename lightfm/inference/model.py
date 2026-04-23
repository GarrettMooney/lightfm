from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from lightfm.inference import _artifact
from lightfm.inference._predict import (
    _InferenceData,
    _predict_impl,
    _predict_rank_impl,
)


class InferenceLightFM:
    """Memory-mapped, predict-only LightFM model.

    Load an artifact written by `LightFM.save_for_inference()` (or produced by
    `lightfm.inference.convert.convert_joblib_to_inference`). Supports
    `predict()` and `predict_rank()` with signatures matching `LightFM`'s.
    Does not support fitting — use `LightFM` for that.
    """

    def __init__(
        self,
        *,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, str],
    ):
        self._arrays = arrays
        self._metadata = metadata
        self._no_components = int(metadata["no_components"])
        self._loss = metadata["loss"]
        self._learning_schedule = metadata["learning_schedule"]

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> "InferenceLightFM":
        arrays, metadata = _artifact.load(path, mmap=mmap)
        return cls(arrays=arrays, metadata=metadata)

    # ---- Introspection ----

    @property
    def no_components(self) -> int:
        return self._no_components

    @property
    def loss(self) -> str:
        return self._loss

    @property
    def learning_schedule(self) -> str:
        return self._learning_schedule

    @property
    def item_embeddings(self) -> np.ndarray:
        return self._arrays["item_embeddings"]

    @property
    def user_embeddings(self) -> np.ndarray:
        return self._arrays["user_embeddings"]

    @property
    def item_biases(self) -> np.ndarray:
        return self._arrays["item_biases"]

    @property
    def user_biases(self) -> np.ndarray:
        return self._arrays["user_biases"]

    # ---- Prediction ----

    def _inference_data(self) -> _InferenceData:
        return _InferenceData(
            item_embeddings=self.item_embeddings,
            user_embeddings=self.user_embeddings,
            item_biases=self.item_biases,
            user_biases=self.user_biases,
            no_components=self._no_components,
            learning_schedule=self._learning_schedule,
            # gradients/momentum stay None → _build_fast_lightfm fills dummies
        )

    def predict(
        self,
        user_ids,
        item_ids,
        item_features: sp.csr_matrix | None = None,
        user_features: sp.csr_matrix | None = None,
        num_threads: int = 1,
    ) -> NDArray[np.float32]:
        return _predict_impl(
            self._inference_data(),
            user_ids,
            item_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )

    def predict_rank(
        self,
        test_interactions: sp.csr_matrix,
        train_interactions: sp.csr_matrix | None = None,
        item_features: sp.csr_matrix | None = None,
        user_features: sp.csr_matrix | None = None,
        num_threads: int = 1,
        check_intersections: bool = True,
    ) -> sp.csr_matrix:
        return _predict_rank_impl(
            self._inference_data(),
            test_interactions,
            train_interactions=train_interactions,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
            check_intersections=check_intersections,
        )
