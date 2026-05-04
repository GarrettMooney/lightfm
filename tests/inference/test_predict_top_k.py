"""Correctness tests for predict_top_k.

Anchor: identical (or tie-equivalent) top-k vs the existing
model.predict(repeat,tile) + argpartition path on a trained model.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from lightfm import LightFM


def _train_tiny(n_users=40, n_items=80, k=16, loss="warp", with_features=False):
    rng = np.random.default_rng(7)
    dense = (rng.random((n_users, n_items)) > 0.85).astype(np.float32)
    interactions = sp.csr_matrix(dense)
    model = LightFM(no_components=k, loss=loss, random_state=1)
    if with_features:
        # 4 categorical-ish item attrs, 3 user attrs, identity-stacked.
        item_feats = sp.hstack([
            sp.identity(n_items, dtype=np.float32, format="csr"),
            sp.csr_matrix(rng.integers(0, 2, size=(n_items, 4)).astype(np.float32)),
        ]).tocsr()
        user_feats = sp.hstack([
            sp.identity(n_users, dtype=np.float32, format="csr"),
            sp.csr_matrix(rng.integers(0, 2, size=(n_users, 3)).astype(np.float32)),
        ]).tocsr()
        model.fit(interactions, item_features=item_feats, user_features=user_feats, epochs=3)
        return model, interactions, item_feats, user_feats
    model.fit(interactions, epochs=3)
    return model, interactions, None, None


def _reference_top_k(model, user_ids, k, n_items, item_ids=None,
                     user_features=None, item_features=None):
    """The current d152-style top-k path: predict all pairs then argpartition."""
    if item_ids is None:
        items = np.arange(n_items, dtype=np.int32)
    else:
        items = np.asarray(item_ids, dtype=np.int32)
    n_batch = len(user_ids)
    n_cand = len(items)
    user_grid = np.repeat(user_ids, n_cand).astype(np.int32)
    item_grid = np.tile(items, n_batch).astype(np.int32)
    scores_flat = model.predict(
        user_grid, item_grid,
        user_features=user_features, item_features=item_features,
    )
    scores = scores_flat.reshape(n_batch, n_cand)
    k_eff = min(k, n_cand)
    topk_unsorted = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
    topk_scores = np.take_along_axis(scores, topk_unsorted, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    topk_pos = np.take_along_axis(topk_unsorted, order, axis=1)
    if item_ids is not None:
        return items[topk_pos], np.take_along_axis(topk_scores, order, axis=1)
    return topk_pos.astype(np.int32), np.take_along_axis(topk_scores, order, axis=1)


def _assert_top_k_equivalent(ref_idx, ref_scores, got_idx, got_scores, atol=1e-4):
    """Top-k can differ in ordering on tied scores; compare as sets per-row plus
    sorted-score sequences (which are stable under tie permutations)."""
    np.testing.assert_allclose(
        np.sort(got_scores, axis=1),
        np.sort(ref_scores, axis=1),
        atol=atol,
    )
    for ref_row, got_row in zip(ref_idx, got_idx):
        assert set(ref_row.tolist()) == set(got_row.tolist()), (
            f"top-k index sets differ: ref={sorted(ref_row.tolist())} "
            f"got={sorted(got_row.tolist())}"
        )


def test_no_features_matches_reference_full_catalog():
    model, _, _, _ = _train_tiny()
    user_ids = np.array([0, 5, 12, 30], dtype=np.int32)
    n_items = model.item_embeddings.shape[0]

    ref_idx, ref_scores = _reference_top_k(model, user_ids, k=10, n_items=n_items)
    got_idx, got_scores = model.predict_top_k(user_ids, k=10, n_items=n_items)

    _assert_top_k_equivalent(ref_idx, ref_scores, got_idx, got_scores)


def test_no_features_matches_reference_item_subset():
    model, _, _, _ = _train_tiny()
    user_ids = np.array([1, 7, 22], dtype=np.int32)
    item_ids = np.array([3, 9, 11, 14, 25, 40, 60, 70], dtype=np.int32)

    ref_idx, ref_scores = _reference_top_k(
        model, user_ids, k=4, n_items=model.item_embeddings.shape[0], item_ids=item_ids
    )
    got_idx, got_scores = model.predict_top_k(user_ids, k=4, item_ids=item_ids)

    _assert_top_k_equivalent(ref_idx, ref_scores, got_idx, got_scores)


def test_with_features_matches_reference_full_catalog():
    model, _, item_feats, user_feats = _train_tiny(with_features=True)
    n_items = item_feats.shape[0]
    user_ids = np.array([0, 4, 9, 15, 28], dtype=np.int32)

    ref_idx, ref_scores = _reference_top_k(
        model, user_ids, k=8, n_items=n_items,
        user_features=user_feats, item_features=item_feats,
    )
    got_idx, got_scores = model.predict_top_k(
        user_ids, k=8, n_items=n_items,
        user_features=user_feats, item_features=item_feats,
    )

    _assert_top_k_equivalent(ref_idx, ref_scores, got_idx, got_scores)


def test_with_features_item_subset():
    model, _, item_feats, user_feats = _train_tiny(with_features=True)
    user_ids = np.array([2, 11, 33], dtype=np.int32)
    item_ids = np.array([0, 5, 10, 15, 20, 35, 50, 75], dtype=np.int32)

    ref_idx, ref_scores = _reference_top_k(
        model, user_ids, k=3, n_items=item_feats.shape[0], item_ids=item_ids,
        user_features=user_feats, item_features=item_feats,
    )
    got_idx, got_scores = model.predict_top_k(
        user_ids, k=3, item_ids=item_ids,
        user_features=user_feats, item_features=item_feats,
    )

    _assert_top_k_equivalent(ref_idx, ref_scores, got_idx, got_scores)


def test_k_greater_than_candidates_clamps():
    model, _, _, _ = _train_tiny(n_items=20)
    user_ids = np.array([0, 1], dtype=np.int32)
    got_idx, got_scores = model.predict_top_k(user_ids, k=999, n_items=20)
    assert got_idx.shape == (2, 20)
    assert got_scores.shape == (2, 20)


def test_invalid_args():
    model, _, _, _ = _train_tiny()
    with pytest.raises(ValueError, match="k must be"):
        model.predict_top_k([0], k=0)
    with pytest.raises(ValueError, match="not both"):
        model.predict_top_k([0], item_ids=[1, 2], n_items=5)


def test_inference_lightfm_predict_top_k_matches_lightfm(tmp_path):
    """The mmap'd inference model must produce identical top-k as LightFM."""
    from lightfm.inference import _artifact
    from lightfm.inference.model import InferenceLightFM

    model, _, _, _ = _train_tiny()
    path = tmp_path / "m.safetensors"
    _artifact.save(path, arrays={
        "item_embeddings": model.item_embeddings,
        "user_embeddings": model.user_embeddings,
        "item_biases": model.item_biases,
        "user_biases": model.user_biases,
    }, metadata={
        "no_components": model.no_components,
        "loss": model.loss,
        "learning_schedule": model.learning_schedule,
    })
    inf = InferenceLightFM.load(path, mmap=False)
    user_ids = np.array([0, 7, 14], dtype=np.int32)
    a_idx, a_sc = model.predict_top_k(user_ids, k=10, n_items=model.item_embeddings.shape[0])
    b_idx, b_sc = inf.predict_top_k(user_ids, k=10, n_items=model.item_embeddings.shape[0])
    np.testing.assert_array_equal(a_idx, b_idx)
    np.testing.assert_allclose(a_sc, b_sc, atol=1e-5)
