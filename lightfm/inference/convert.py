from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import joblib

from lightfm.inference import _artifact

log = logging.getLogger(__name__)


class ConversionError(Exception):
    """Raised when converting a legacy joblib model fails."""


def _is_remote(path: str | Path) -> bool:
    s = str(path)
    return "://" in s and not s.startswith("file://")


def _scheme(path: str | Path) -> str:
    return str(path).split("://", 1)[0]


@contextmanager
def _open_read(path: str | Path, storage_options: dict | None):
    if _is_remote(path):
        import fsspec
        with fsspec.open(str(path), "rb", **(storage_options or {})) as f:
            yield f
    else:
        with open(str(path), "rb") as f:
            yield f


def _file_size(path: str | Path, storage_options: dict | None) -> int:
    if _is_remote(path):
        import fsspec
        fs = fsspec.filesystem(_scheme(path), **(storage_options or {}))
        return int(fs.size(str(path)))
    return Path(str(path)).stat().st_size


def _write_artifact(
    dst: str | Path,
    *,
    arrays: dict,
    metadata: dict,
    storage_options: dict | None,
) -> None:
    """Write via _artifact.save to `dst`. For remote URIs, stage through a
    local tempfile then upload."""
    if _is_remote(dst):
        import fsspec
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _artifact.save(tmp_path, arrays=arrays, metadata=metadata)
            # Streaming copy — critical at the 60GB scale this project targets;
            # a .read() into memory would OOM the converter host.
            with fsspec.open(str(dst), "wb", **(storage_options or {})) as out, \
                 open(tmp_path, "rb") as src_f:
                shutil.copyfileobj(src_f, out)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        _artifact.save(Path(str(dst)), arrays=arrays, metadata=metadata)


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
    from lightfm import LightFM  # local import avoids circular dep

    try:
        with _open_read(src, storage_options) as f:
            loaded = joblib.load(f)
    except FileNotFoundError as exc:
        raise ConversionError(f"source not found: {src}") from exc
    except Exception as exc:
        raise ConversionError(f"failed to load {src}: {exc}") from exc

    if not isinstance(loaded, LightFM):
        raise ConversionError(
            f"expected LightFM, got {type(loaded).__name__} at {src}"
        )
    if loaded.item_embeddings is None:
        raise ConversionError(f"source model is not fitted: {src}")

    arrays = {
        "item_embeddings": loaded.item_embeddings,
        "user_embeddings": loaded.user_embeddings,
        "item_biases": loaded.item_biases,
        "user_biases": loaded.user_biases,
    }
    metadata = {
        "no_components": loaded.no_components,
        "loss": loaded.loss,
        "learning_schedule": loaded.learning_schedule,
    }
    _write_artifact(dst, arrays=arrays, metadata=metadata, storage_options=storage_options)

    src_size = _file_size(src, storage_options)
    dst_size = _file_size(dst, storage_options)
    pct = 100.0 * (1.0 - dst_size / src_size) if src_size > 0 else 0.0
    log.info(
        "converted %s (%d bytes) → %s (%d bytes), %.1f%% smaller",
        src, src_size, dst, dst_size, pct,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lightfm.inference.convert",
        description="Convert a joblib.dump(LightFM) artifact to safetensors.",
    )
    parser.add_argument("src", help="Source path or URI (local or gs://..., s3://..., etc.)")
    parser.add_argument("dst", help="Destination path or URI")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    try:
        convert_joblib_to_inference(args.src, args.dst)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
