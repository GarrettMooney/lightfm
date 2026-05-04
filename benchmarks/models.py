"""Synthetic LightFM factory for benchmarking.

Builds a LightFM instance whose 12 arrays (embeddings, gradients, momentum,
biases) are allocated and filled with pseudo-random float32 values — as if
`fit()` had run for some number of epochs. Skips actual training cost so
medium/large-scale benchmarks are feasible.
"""
from __future__ import annotations

import numpy as np

from lightfm import LightFM

SIZE_PRESETS: dict[str, dict[str, int]] = {
    "tiny":   {"n_users": 1_000,     "n_items": 5_000,      "no_components": 32},
    "medium": {"n_users": 300_000,   "n_items": 1_000_000,  "no_components": 128},
    "large":  {"n_users": 5_000_000, "n_items": 20_000_000, "no_components": 128},
}


def make_fitted_model(
    *,
    n_users: int,
    n_items: int,
    no_components: int,
    loss: str = "warp",
    learning_schedule: str = "adagrad",
    seed: int = 0,
) -> LightFM:
    """Return a LightFM whose arrays are shaped and filled as if `fit()` ran.

    Uses `LightFM._initialize` (the same private method `fit` calls internally)
    to allocate the 12 arrays, then overwrites the embeddings with fresh
    `standard_normal` draws so the model is distinct per seed. Gradients and
    momentum keep `_initialize`'s default values (including the adagrad `+= 1`
    convention). Does not call `fit`.
    """
    model = LightFM(
        no_components=no_components,
        loss=loss,
        learning_schedule=learning_schedule,
        random_state=seed,
    )
    # _initialize(no_components, no_item_features, no_user_features);
    # with identity feature matrices (the default), features == entities.
    model._initialize(no_components, n_items, n_users)

    rng = np.random.default_rng(seed)
    model.item_embeddings = rng.standard_normal(
        (n_items, no_components), dtype=np.float32
    )
    model.user_embeddings = rng.standard_normal(
        (n_users, no_components), dtype=np.float32
    )
    return model
