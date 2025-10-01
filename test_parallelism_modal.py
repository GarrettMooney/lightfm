#!/usr/bin/env python
"""
Modal script to test LightFM parallelism with OpenMP support.
Since macOS doesn't support OpenMP well, we run this in a Linux container.
"""

import modal
from pathlib import Path

app = modal.App("lightfm-parallelism-test")

# Get the local lightfm directory
local_lightfm_path = Path(__file__).parent

# Create an image with all necessary build dependencies and install LightFM
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "gcc", "g++", "libomp-dev", "libopenblas-dev", "liblapack-dev")
    .pip_install(
        "numpy",
        "scipy",
        "requests",
        "cython",
        "lightfm @ git+https://github.com/garrettmooney/lightfm.git@master",
    )
)


@app.function(
    image=image,
    timeout=600,
)
def test_parallelism():
    import time
    import numpy as np
    from scipy.sparse import coo_matrix
    from lightfm import LightFM

    print("=" * 70)
    print("LightFM Parallelism Test")
    print("=" * 70 + "\n")

    # Check OpenMP
    print("=" * 60)
    print("Checking OpenMP Availability")
    print("=" * 60)

    try:
        from lightfm import _lightfm_fast_openmp

        print("✓ OpenMP version is available and loaded")
        has_openmp = True
    except ImportError:
        try:
            from lightfm import _lightfm_fast_no_openmp

            print("⚠ Using non-OpenMP version (single-threaded only)")
            has_openmp = False
        except ImportError:
            print("✗ No LightFM fast module found")
            return 1

    # Generate test data
    print("\n" + "=" * 60)
    print("Generating Test Data")
    print("=" * 60)

    np.random.seed(42)
    num_users, num_items, num_interactions = 2000, 1000, 20000
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    data = np.ones(num_interactions)
    interactions = coo_matrix(
        (data, (user_ids, item_ids)), shape=(num_users, num_items)
    ).tocsr()
    print(
        f"Created {num_users}x{num_items} interaction matrix with {num_interactions} interactions"
    )

    # Test training parallelism
    print("\n" + "=" * 60)
    print("Testing Training Parallelism")
    print("=" * 60)

    thread_counts = [1, 2, 4]
    train_times = {}

    for num_threads in thread_counts:
        model = LightFM(no_components=30, loss="warp", random_state=42)
        start = time.time()
        model.fit(interactions, epochs=5, num_threads=num_threads, verbose=False)
        elapsed = time.time() - start
        train_times[num_threads] = elapsed
        print(f"Threads: {num_threads:2d} | Time: {elapsed:.3f}s")

    speedup_2 = train_times[1] / train_times[2]
    speedup_4 = train_times[1] / train_times[4]
    print(f"\nSpeedup (2 threads): {speedup_2:.2f}x")
    print(f"Speedup (4 threads): {speedup_4:.2f}x")

    if speedup_2 > 1.2:
        print("✓ Training parallelism appears to be working")
    else:
        print("⚠ Warning: Limited speedup with 2 threads")

    # Test prediction parallelism
    print("\n" + "=" * 60)
    print("Testing Prediction Parallelism")
    print("=" * 60)

    # Train a model for predictions
    model = LightFM(no_components=30, loss="warp", random_state=42)
    model.fit(interactions, epochs=3, num_threads=4, verbose=False)

    # Create prediction data
    num_test_users = min(500, num_users)
    user_ids_pred = np.repeat(np.arange(num_test_users), num_items)
    item_ids_pred = np.tile(np.arange(num_items), num_test_users)

    pred_times = {}
    for num_threads in thread_counts:
        start = time.time()
        predictions = model.predict(
            user_ids_pred, item_ids_pred, num_threads=num_threads
        )
        elapsed = time.time() - start
        pred_times[num_threads] = elapsed
        print(
            f"Threads: {num_threads:2d} | Time: {elapsed:.3f}s | Predictions: {len(predictions):,}"
        )

    speedup_2 = pred_times[1] / pred_times[2]
    speedup_4 = pred_times[1] / pred_times[4]
    print(f"\nSpeedup (2 threads): {speedup_2:.2f}x")
    print(f"Speedup (4 threads): {speedup_4:.2f}x")

    if speedup_2 > 1.2:
        print("✓ Prediction parallelism appears to be working")
    else:
        print("⚠ Warning: Limited speedup with 2 threads")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"OpenMP available: {has_openmp}")
    print(f"Training speedup (4 threads): {train_times[1] / train_times[4]:.2f}x")
    print(f"Prediction speedup (4 threads): {pred_times[1] / pred_times[4]:.2f}x")
    print()

    return 0


@app.local_entrypoint()
def main():
    print("Starting LightFM parallelism test on Modal...\n")
    return_code = test_parallelism.remote()

    if return_code == 0:
        print("\n✓ Test completed successfully")
    else:
        print(f"\n✗ Test failed with return code {return_code}")

    return return_code
