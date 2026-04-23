"""Benchmark suite CLI entry point.

Usage:
    uv run python -m benchmarks --size {tiny,medium,large,custom} [options]

Runs axes in order (size, load, predict, ram). If a later axis needs
artifacts produced by size but --skip-axis=size was passed, size runs
anyway with a one-line notice. Writes a JSON sidecar under the output
directory and prints a human-readable table to stdout.
"""
from __future__ import annotations

import argparse
import json
import platform as platform_mod
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import bench_load, bench_predict, bench_ram, bench_size, results
from benchmarks.models import SIZE_PRESETS, make_fitted_model
from lightfm import __version__ as _lightfm_version

ALL_AXES = ("size", "load", "predict", "ram")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Inference artifact benchmark suite (size / load / predict / RAM).",
    )
    p.add_argument("--size", choices=("tiny", "medium", "large", "custom"), default="tiny")
    p.add_argument("--n-users", type=int, help="Override (required with --size custom)")
    p.add_argument("--n-items", type=int, help="Override")
    p.add_argument("--no-components", type=int, help="Override")
    p.add_argument("--n-workers", type=int, default=4, help="Workers for axis D (default 4)")
    p.add_argument("--repeats-load", type=int, default=5)
    p.add_argument("--repeats-predict", type=int, default=3)
    p.add_argument(
        "--skip-axis",
        default="",
        help=f"Comma-separated axis names to skip. Choices: {','.join(ALL_AXES)}",
    )
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--label", default="", help="Optional label appended to JSON filename")
    return p.parse_args(argv)


def _resolve_config(args: argparse.Namespace) -> dict:
    if args.size == "custom":
        if args.n_users is None or args.n_items is None or args.no_components is None:
            raise SystemExit("--size custom requires --n-users, --n-items, --no-components")
        return {"n_users": args.n_users, "n_items": args.n_items, "no_components": args.no_components}
    cfg = dict(SIZE_PRESETS[args.size])
    if args.n_users is not None:
        cfg["n_users"] = args.n_users
    if args.n_items is not None:
        cfg["n_items"] = args.n_items
    if args.no_components is not None:
        cfg["no_components"] = args.no_components
    return cfg


def _build_metadata(args: argparse.Namespace, cfg: dict) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size": args.size,
        "config": cfg,
        "host": {
            "platform": sys.platform,
            "python": platform_mod.python_version(),
            "cpu_count": (__import__("os")).cpu_count(),
        },
        "lightfm_version": _lightfm_version,
        "n_workers": args.n_workers,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = _resolve_config(args)
    except SystemExit as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    skip = {s.strip() for s in args.skip_axis.split(",") if s.strip()}
    unknown = skip - set(ALL_AXES)
    if unknown:
        print(f"error: unknown --skip-axis values: {sorted(unknown)}", file=sys.stderr)
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / f".artifacts-{int(time.time())}"
    artifact_dir.mkdir()

    print(f"Benchmarking size={args.size} ({cfg['n_users']:,} users × {cfg['n_items']:,} items × {cfg['no_components']} components)")
    print(f"Artifacts: {artifact_dir}")

    metadata = _build_metadata(args, cfg)
    all_results: dict = {}

    needs_artifacts = bool({"load", "predict", "ram"} - skip)
    run_size = "size" not in skip or needs_artifacts

    # --- Axis A: size ---
    if run_size:
        try:
            model = make_fitted_model(**cfg)
            if "size" in skip:
                print("note: running bench_size anyway (later axes need its artifacts)")
            size_result = bench_size.run(model, artifact_dir)
            if "size" not in skip:
                all_results["size"] = size_result
            del model
        except Exception:
            traceback.print_exc()
            all_results["size"] = {"error": "bench_size failed"}

    # --- Axis B: load ---
    if "load" not in skip:
        try:
            all_results["load"] = bench_load.run(artifact_dir, repeats=args.repeats_load)
        except Exception:
            traceback.print_exc()
            all_results["load"] = {"error": "bench_load failed"}

    # --- Axis C: predict ---
    if "predict" not in skip:
        try:
            all_results["predict"] = bench_predict.run(artifact_dir, repeats=args.repeats_predict)
        except Exception:
            traceback.print_exc()
            all_results["predict"] = {"error": "bench_predict failed"}

    # --- Axis D: ram ---
    if "ram" not in skip:
        try:
            all_results["ram"] = bench_ram.run(artifact_dir, n_workers=args.n_workers)
        except Exception:
            traceback.print_exc()
            all_results["ram"] = {"error": "bench_ram failed"}

    # --- Output ---
    results.print_table(all_results, metadata)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = f"-{args.label}" if args.label else ""
    json_path = output_dir / f"{stamp}-{args.size}{suffix}.json"
    results.write_json(all_results, metadata, json_path)
    print(f"\nJSON written to: {json_path}")

    had_error = any(isinstance(v, dict) and v.get("error") for v in all_results.values())
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
