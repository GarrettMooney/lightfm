from __future__ import annotations

import logging
from pathlib import Path

import joblib

from lightfm.inference import _artifact

log = logging.getLogger(__name__)


class ConversionError(Exception):
    """Raised when converting a legacy joblib model fails."""


def convert_joblib_to_inference(
    src: str | Path,
    dst: str | Path,
    *,
    storage_options: dict | None = None,
) -> None:
    """Convert a legacy `joblib.dump(LightFM)` artifact to safetensors.

    Loads the entire model into RAM. For very large models (~60GB), run on a
    host with sufficient memory. Streaming conversion is a v2 concern.
    """
    from lightfm import LightFM  # local import avoids circular dep at module load

    src_path = Path(str(src))
    dst_path = Path(str(dst))

    try:
        loaded = joblib.load(str(src_path))
    except FileNotFoundError as exc:
        raise ConversionError(f"source not found: {src_path}") from exc
    except Exception as exc:
        raise ConversionError(f"failed to load {src_path}: {exc}") from exc

    if not isinstance(loaded, LightFM):
        raise ConversionError(
            f"expected LightFM, got {type(loaded).__name__} at {src_path}"
        )

    if loaded.item_embeddings is None:
        raise ConversionError(f"source model is not fitted: {src_path}")

    _artifact.save(
        dst_path,
        arrays={
            "item_embeddings": loaded.item_embeddings,
            "user_embeddings": loaded.user_embeddings,
            "item_biases": loaded.item_biases,
            "user_biases": loaded.user_biases,
        },
        metadata={
            "no_components": loaded.no_components,
            "loss": loaded.loss,
            "learning_schedule": loaded.learning_schedule,
        },
    )

    src_size = src_path.stat().st_size
    dst_size = dst_path.stat().st_size
    pct = 100.0 * (1.0 - dst_size / src_size) if src_size > 0 else 0.0
    log.info(
        "converted %s (%d bytes) → %s (%d bytes), %.1f%% smaller",
        src_path, src_size, dst_path, dst_size, pct,
    )
