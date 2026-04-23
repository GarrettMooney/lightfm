from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from lightfm._lightfm_fast import (
    CSRMatrix,
    FastLightFM,
    predict_lightfm,
    predict_ranks,
)

CYTHON_DTYPE = np.float32


@dataclass
class _InferenceData:
    """Minimal state needed to score. Populated by LightFM (with full training
    arrays) or InferenceLightFM (with dummy gradients/momentum).
    """
    item_embeddings: np.ndarray
    user_embeddings: np.ndarray
    item_biases: np.ndarray
    user_biases: np.ndarray
    no_components: int
    learning_schedule: str  # "adagrad" or "adadelta"
    # Training-side supplies real arrays; inference supplies None (we fill with dummies).
    item_embedding_gradients: np.ndarray | None = None
    item_embedding_momentum: np.ndarray | None = None
    item_bias_gradients: np.ndarray | None = None
    item_bias_momentum: np.ndarray | None = None
    user_embedding_gradients: np.ndarray | None = None
    user_embedding_momentum: np.ndarray | None = None
    user_bias_gradients: np.ndarray | None = None
    user_bias_momentum: np.ndarray | None = None
    # Training-side hyperparams used by FastLightFM. Unused during predict but
    # required by the struct constructor.
    learning_rate: float = 0.05
    rho: float = 0.95
    epsilon: float = 1e-6
    max_sampled: int = 10


def _dummy_like_2d(shape: tuple[int, int]) -> np.ndarray:
    # A single-row allocation satisfies the [:, ::1] memoryview type without
    # carrying per-feature storage. predict_lightfm never reads these.
    return np.zeros((1, shape[1] if shape[1] > 0 else 1), dtype=np.float32)


def _dummy_like_1d() -> np.ndarray:
    return np.zeros(1, dtype=np.float32)


def _build_fast_lightfm(data: _InferenceData) -> FastLightFM:
    ie_grad = data.item_embedding_gradients if data.item_embedding_gradients is not None else _dummy_like_2d(data.item_embeddings.shape)
    ie_mom = data.item_embedding_momentum if data.item_embedding_momentum is not None else _dummy_like_2d(data.item_embeddings.shape)
    ib_grad = data.item_bias_gradients if data.item_bias_gradients is not None else _dummy_like_1d()
    ib_mom = data.item_bias_momentum if data.item_bias_momentum is not None else _dummy_like_1d()
    ue_grad = data.user_embedding_gradients if data.user_embedding_gradients is not None else _dummy_like_2d(data.user_embeddings.shape)
    ue_mom = data.user_embedding_momentum if data.user_embedding_momentum is not None else _dummy_like_2d(data.user_embeddings.shape)
    ub_grad = data.user_bias_gradients if data.user_bias_gradients is not None else _dummy_like_1d()
    ub_mom = data.user_bias_momentum if data.user_bias_momentum is not None else _dummy_like_1d()

    return FastLightFM(
        data.item_embeddings,
        ie_grad,
        ie_mom,
        data.item_biases,
        ib_grad,
        ib_mom,
        data.user_embeddings,
        ue_grad,
        ue_mom,
        data.user_biases,
        ub_grad,
        ub_mom,
        data.no_components,
        int(data.learning_schedule == "adadelta"),
        data.learning_rate,
        data.rho,
        data.epsilon,
        data.max_sampled,
    )


def _to_cython_dtype(mat: sp.csr_matrix) -> sp.csr_matrix:
    if mat.dtype != CYTHON_DTYPE:
        return mat.astype(CYTHON_DTYPE)
    return mat


def _construct_feature_matrices(
    n_users: int,
    n_items: int,
    user_features: sp.csr_matrix | None,
    item_features: sp.csr_matrix | None,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    if user_features is None:
        user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format="csr")
    else:
        user_features = user_features.tocsr()

    if item_features is None:
        item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format="csr")
    else:
        item_features = item_features.tocsr()

    if n_users > user_features.shape[0]:
        raise Exception(
            "Number of user feature rows does not equal the number of users"
        )
    if n_items > item_features.shape[0]:
        raise Exception(
            "Number of item feature rows does not equal the number of items"
        )

    if not user_embeddings.shape[0] >= user_features.shape[1]:
        raise ValueError(
            "The user feature matrix specifies more features than there are "
            "estimated feature embeddings: {} vs {}.".format(
                user_embeddings.shape[0], user_features.shape[1]
            )
        )
    if not item_embeddings.shape[0] >= item_features.shape[1]:
        raise ValueError(
            "The item feature matrix specifies more features than there are "
            "estimated feature embeddings: {} vs {}.".format(
                item_embeddings.shape[0], item_features.shape[1]
            )
        )

    return _to_cython_dtype(user_features), _to_cython_dtype(item_features)


def _validate_predict_inputs(
    user_ids: int | NDArray[np.int32] | list | tuple,
    item_ids: NDArray[np.int32] | list | tuple,
    num_threads: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    if isinstance(user_ids, int):
        user_ids = np.repeat(np.int32(user_ids), len(item_ids))

    if isinstance(user_ids, (list, tuple)):
        user_ids = np.array(user_ids, dtype=np.int32)

    if isinstance(item_ids, (list, tuple)):
        item_ids = np.array(item_ids, dtype=np.int32)

    if len(user_ids) != len(item_ids):
        raise ValueError(
            f"Expected the number of user IDs ({len(user_ids)}) to equal the number"
            f" of item IDs ({len(item_ids)})"
        )

    if user_ids.dtype != np.int32:
        user_ids = user_ids.astype(np.int32)
    if item_ids.dtype != np.int32:
        item_ids = item_ids.astype(np.int32)

    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    if user_ids.min() < 0 or item_ids.min() < 0:
        raise ValueError(
            "User or item ids cannot be negative. "
            "Check your inputs for negative numbers "
            "or very large numbers that can overflow."
        )

    return user_ids, item_ids


def _predict_impl(
    data: _InferenceData,
    user_ids,
    item_ids,
    user_features: sp.csr_matrix | None = None,
    item_features: sp.csr_matrix | None = None,
    num_threads: int = 1,
) -> NDArray[np.float32]:
    user_ids, item_ids = _validate_predict_inputs(user_ids, item_ids, num_threads)

    n_users = int(user_ids.max()) + 1
    n_items = int(item_ids.max()) + 1

    user_features, item_features = _construct_feature_matrices(
        n_users,
        n_items,
        user_features,
        item_features,
        data.user_embeddings,
        data.item_embeddings,
    )

    lightfm_data = _build_fast_lightfm(data)
    predictions = np.empty(len(user_ids), dtype=np.float32)

    predict_lightfm(
        CSRMatrix(item_features),
        CSRMatrix(user_features),
        user_ids,
        item_ids,
        predictions,
        lightfm_data,
        num_threads,
    )

    return predictions


def _check_test_train_intersections(test_mat, train_mat) -> None:
    if train_mat is not None:
        n_intersections = test_mat.multiply(train_mat).nnz
        if n_intersections:
            raise ValueError(
                "Test interactions matrix and train interactions "
                "matrix share %d interactions. This will cause "
                "incorrect evaluation, check your data split." % n_intersections
            )


def _predict_rank_impl(
    data: _InferenceData,
    test_interactions: sp.csr_matrix,
    train_interactions: sp.csr_matrix | None = None,
    user_features: sp.csr_matrix | None = None,
    item_features: sp.csr_matrix | None = None,
    num_threads: int = 1,
    check_intersections: bool = True,
) -> sp.csr_matrix:
    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    if check_intersections:
        _check_test_train_intersections(test_interactions, train_interactions)

    n_users, n_items = test_interactions.shape

    user_features, item_features = _construct_feature_matrices(
        n_users,
        n_items,
        user_features,
        item_features,
        data.user_embeddings,
        data.item_embeddings,
    )

    if not item_features.shape[1] == data.item_embeddings.shape[0]:
        raise ValueError("Incorrect number of features in item_features")
    if not user_features.shape[1] == data.user_embeddings.shape[0]:
        raise ValueError("Incorrect number of features in user_features")

    test_interactions = _to_cython_dtype(test_interactions.tocsr())

    if train_interactions is None:
        train_interactions = sp.csr_matrix((n_users, n_items), dtype=CYTHON_DTYPE)
    else:
        train_interactions = _to_cython_dtype(train_interactions.tocsr())

    ranks = sp.csr_matrix(
        (
            np.zeros_like(test_interactions.data),
            test_interactions.indices,
            test_interactions.indptr,
        ),
        shape=test_interactions.shape,
    )

    lightfm_data = _build_fast_lightfm(data)

    predict_ranks(
        CSRMatrix(item_features),
        CSRMatrix(user_features),
        CSRMatrix(test_interactions),
        CSRMatrix(train_interactions),
        ranks.data,
        lightfm_data,
        num_threads,
    )

    return ranks
