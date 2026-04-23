"""Shared output formatting for benchmark results.

Two entry points:
- `print_table(results, metadata)` — prints a human-readable table per axis.
- `write_json(results, metadata, path)` — writes the full JSON sidecar.

`results` is a dict with axis keys ("size", "load", "predict", "ram"); any
subset is allowed (axes that were skipped or errored are simply absent).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(results: dict[str, Any], metadata: dict[str, Any], path: Path) -> None:
    """Write the full JSON payload. Missing axes are simply omitted."""
    payload = {"metadata": metadata, "axes": results}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def print_table(results: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Print a human-readable table to stdout, one section per axis present."""
    size_label = metadata.get("size", "?")
    cfg = metadata.get("config", {})
    cfg_str = (
        f"{cfg.get('n_users', '?'):,} users × {cfg.get('n_items', '?'):,} items × "
        f"{cfg.get('no_components', '?')} components"
    )

    if "size" in results:
        _print_size(results["size"], size_label, cfg_str)
    if "load" in results:
        _print_load(results["load"])
    if "predict" in results:
        _print_predict(results["predict"])
    if "ram" in results:
        _print_ram(results["ram"])


def _print_size(data: dict, size_label: str, cfg_str: str) -> None:
    print(f"\n=== Artifact Size ({size_label}: {cfg_str}) ===")
    print(f"{'Method':<30} {'Bytes':>15}   Ratio")
    j = data["joblib_bytes"]
    s = data["safetensors_bytes"]
    print(f"{'joblib.dump(LightFM)':<30} {j:>15,}   1.00x (baseline)")
    print(f"{'save_for_inference':<30} {s:>15,}   {data['reduction_ratio']:.2f}x ({data['reduction_pct']:.1f}% smaller)")


def _print_load(data: dict) -> None:
    print("\n=== Load Time (wall-clock seconds) ===")
    print(f"{'Method':<30} {'Min':>10} {'Median':>10} {'Max':>10}")
    for method, label in (
        ("joblib", "joblib.load"),
        ("inference_heap", "InferenceLightFM mmap=False"),
        ("inference_mmap", "InferenceLightFM mmap=True"),
    ):
        if method in data:
            d = data[method]
            print(f"{label:<30} {d['min']:>10.4f} {d['median']:>10.4f} {d['max']:>10.4f}")


def _print_predict(data: list[dict]) -> None:
    print("\n=== Predict Throughput (predictions/sec, median of 3 runs) ===")
    print(f"{'Batch Size':>12} {'LightFM':>18} {'InferenceLightFM':>18} {'Ratio':>8}")
    for row in data:
        print(
            f"{row['batch_size']:>12,} "
            f"{row['lightfm']['per_sec']:>18,} "
            f"{row['inference']['per_sec']:>18,} "
            f"{row['ratio']:>7.2f}x"
        )


def _print_ram(data: dict) -> None:
    if data.get("skipped"):
        print(f"\n=== Multi-Process RAM ===")
        print(f"SKIPPED: {data.get('reason', 'unknown')}")
        return
    measurement = data["measurement"].upper()
    n = data["n_workers"]
    model_mb = data["model_bytes"] / 1e6
    print(f"\n=== Multi-Process RAM (N={n} workers, {data['platform']} {measurement}) ===")
    print(f"Model size: {model_mb:,.1f} MB")
    a = data["variant_a"]["total"]
    b = data["variant_b"]["total"]
    print(f"Variant A (joblib.load + Pool):  total {measurement} = {a/1e6:,.1f} MB ({a/data['model_bytes']:.2f}x model)")
    print(f"Variant B (mmap + Pool):         total {measurement} = {b/1e6:,.1f} MB ({b/data['model_bytes']:.2f}x model)")
    print(f"Sharing ratio (A/B):             {data['sharing_ratio']:.2f}x")
    if data.get("caveats"):
        print("Caveats:")
        for c in data["caveats"]:
            print(f"  - {c}")
