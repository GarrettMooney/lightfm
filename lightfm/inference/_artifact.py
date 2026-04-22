from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from lightfm.version import __version__ as _lightfm_version

FORMAT_VERSION = "1"
SUPPORTED_VERSIONS = frozenset({"1"})

REQUIRED_TENSORS = (
    "item_embeddings",
    "user_embeddings",
    "item_biases",
    "user_biases",
)


class ArtifactLoadError(Exception):
    """Raised when a safetensors artifact cannot be loaded or validated."""


def save(
    path: str | Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Write the inference artifact to `path`.

    `arrays` must contain all four required tensors (float32, C-contiguous).
    `metadata` supplies no_components, loss, learning_schedule. Other fields
    (format_version, lightfm_version, embeddings_dtype, created_at) are added
    by this function.
    """
    path = Path(path)

    missing = [name for name in REQUIRED_TENSORS if name not in arrays]
    if missing:
        raise ValueError(f"save() missing required tensors: {missing}")

    # Lock in C-contiguous float32 for cache-line-friendly mmap reads.
    contiguous = {
        name: np.ascontiguousarray(arr, dtype=np.float32) for name, arr in arrays.items()
    }

    full_metadata: dict[str, str] = {
        "format_version": FORMAT_VERSION,
        "lightfm_version": _lightfm_version,
        "no_components": str(int(metadata["no_components"])),
        "loss": str(metadata["loss"]),
        "learning_schedule": str(metadata["learning_schedule"]),
        "embeddings_dtype": "float32",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_file(contiguous, str(path), metadata=full_metadata)


def load(
    path: str | Path,
    *,
    mmap: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load the inference artifact from `path`.

    Returns `(arrays, metadata)`. With `mmap=True`, arrays are views onto
    file-backed memory; pages are shared across forks. With `mmap=False`,
    arrays are heap-allocated copies.
    """
    path = Path(path)

    if not path.exists():
        raise ArtifactLoadError(f"artifact not found: {path}")

    if mmap:
        from safetensors import safe_open

        arrays: dict[str, np.ndarray] = {}
        try:
            with safe_open(str(path), framework="numpy") as f:
                metadata = dict(f.metadata() or {})
                keys = set(f.keys())
                _validate_header(metadata, path)
                _validate_tensor_keys(keys, path)
                for name in REQUIRED_TENSORS:
                    arrays[name] = f.get_tensor(name)
                _validate_shapes(arrays, metadata, path)
                _validate_dtypes(arrays, metadata, path)
        except ArtifactLoadError:
            raise
        except Exception as exc:
            raise ArtifactLoadError(f"failed to read {path}: {exc}") from exc
    else:
        try:
            arrays = load_file(str(path))
            # safetensors.numpy.load_file doesn't expose metadata; read it separately.
            from safetensors import safe_open

            with safe_open(str(path), framework="numpy") as f:
                metadata = dict(f.metadata() or {})
        except ArtifactLoadError:
            raise
        except Exception as exc:
            raise ArtifactLoadError(f"failed to read {path}: {exc}") from exc
        _validate_header(metadata, path)
        _validate_tensor_keys(set(arrays.keys()), path)
        _validate_shapes(arrays, metadata, path)
        _validate_dtypes(arrays, metadata, path)

    return arrays, metadata


def _validate_header(metadata: dict[str, str], path: Path) -> None:
    if "format_version" not in metadata:
        raise ArtifactLoadError(
            f"{path}: missing format_version — is this a lightfm artifact?"
        )
    version = metadata["format_version"]
    if version not in SUPPORTED_VERSIONS:
        raise ArtifactLoadError(
            f"{path}: artifact is version {version!r}, this lightfm "
            f"({_lightfm_version}) supports {sorted(SUPPORTED_VERSIONS)}"
        )


def _validate_tensor_keys(keys: set[str], path: Path) -> None:
    missing = [name for name in REQUIRED_TENSORS if name not in keys]
    if missing:
        raise ArtifactLoadError(f"{path}: missing required tensors: {missing}")


def _validate_shapes(
    arrays: dict[str, np.ndarray], metadata: dict[str, str], path: Path
) -> None:
    try:
        no_components = int(metadata["no_components"])
    except (KeyError, ValueError) as exc:
        raise ArtifactLoadError(f"{path}: missing or invalid no_components") from exc

    ie = arrays["item_embeddings"]
    ue = arrays["user_embeddings"]
    ib = arrays["item_biases"]
    ub = arrays["user_biases"]

    if ie.ndim != 2 or ie.shape[1] != no_components:
        raise ArtifactLoadError(
            f"{path}: item_embeddings shape {ie.shape} inconsistent with "
            f"no_components={no_components}"
        )
    if ue.ndim != 2 or ue.shape[1] != no_components:
        raise ArtifactLoadError(
            f"{path}: user_embeddings shape {ue.shape} inconsistent with "
            f"no_components={no_components}"
        )
    if ib.ndim != 1 or ib.shape[0] != ie.shape[0]:
        raise ArtifactLoadError(
            f"{path}: item_biases shape {ib.shape} does not match item_embeddings rows {ie.shape[0]}"
        )
    if ub.ndim != 1 or ub.shape[0] != ue.shape[0]:
        raise ArtifactLoadError(
            f"{path}: user_biases shape {ub.shape} does not match user_embeddings rows {ue.shape[0]}"
        )


def _validate_dtypes(
    arrays: dict[str, np.ndarray], metadata: dict[str, str], path: Path
) -> None:
    declared = metadata.get("embeddings_dtype", "float32")
    if declared != "float32":
        raise ArtifactLoadError(
            f"{path}: embeddings_dtype={declared!r} not supported in v1 (expected 'float32')"
        )
    for name in REQUIRED_TENSORS:
        actual = arrays[name].dtype
        if actual != np.float32:
            raise ArtifactLoadError(
                f"{path}: tensor {name!r} has dtype {actual}, expected float32"
            )
