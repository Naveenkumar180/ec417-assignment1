"""
EC-417 Assignment 1
Problem 1: Pairwise Euclidean Distances

This program implements pairwise Euclidean distances using:
1. Nested Python loops
2. NumPy broadcasting
3. Vectorized norm-expansion formula

It verifies correctness and benchmarks the implementations.
"""

import time
import numpy as np


def pairwise_distances_loops(X):
    """
    Compute all pairwise Euclidean distances using nested Python loops.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Input matrix where each row represents one data point.

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Pairwise Euclidean distance matrix.
    """
    N = X.shape[0]
    D = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            diff = X[i] - X[j]
            D[i, j] = np.sqrt(np.sum(diff ** 2))

    return D


def pairwise_distances_broadcast(X):
    """
    Compute all pairwise Euclidean distances using NumPy broadcasting.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Input matrix where each row represents one data point.

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Pairwise Euclidean distance matrix.
    """
    differences = X[:, None, :] - X[None, :, :]
    D_squared = np.sum(differences ** 2, axis=2)

    # Protect against tiny negative values from floating-point rounding.
    D_squared = np.maximum(D_squared, 0.0)

    return np.sqrt(D_squared)


def pairwise_distances_norm_expansion(X):
    """
    Compute all pairwise Euclidean distances using norm expansion.

    The identity used is:
        ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi^T xj

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Input matrix where each row represents one data point.

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Pairwise Euclidean distance matrix.
    """
    squared_norms = np.sum(X ** 2, axis=1)

    D_squared = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (X @ X.T)
    )

    # Protect against tiny negative values from floating-point rounding.
    D_squared = np.maximum(D_squared, 0.0)

    return np.sqrt(D_squared)


def benchmark(function, X, repeats=3):
    """
    Measure the best execution time over multiple runs.

    Parameters
    ----------
    function : callable
        Distance function to benchmark.

    X : np.ndarray, shape (N, d)
        Input dataset.

    repeats : int
        Number of timing repetitions.

    Returns
    -------
    best_time : float
        Minimum execution time in seconds.
    """
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        function(X)
        end = time.perf_counter()
        times.append(end - start)

    return min(times)


def main():
    """
    Run correctness checks and benchmark all implementations.
    """

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    SEED = 42
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------
    # Part 1: Correctness verification
    # ---------------------------------------------------------
    N_small = 100
    d_small = 5

    X_small = rng.normal(size=(N_small, d_small))

    D_loop = pairwise_distances_loops(X_small)
    D_broadcast = pairwise_distances_broadcast(X_small)
    D_norm = pairwise_distances_norm_expansion(X_small)

    print("=" * 60)
    print("PROBLEM 1: PAIRWISE EUCLIDEAN DISTANCES")
    print("=" * 60)

    print("\nCorrectness checks:")
    print(
        "Loop vs Broadcasting:",
        np.allclose(D_loop, D_broadcast, rtol=1e-10, atol=1e-10)
    )

    print(
        "Loop vs Norm Expansion:",
        np.allclose(D_loop, D_norm, rtol=1e-10, atol=1e-10)
    )

    print(
        "Broadcasting vs Norm Expansion:",
        np.allclose(D_broadcast, D_norm, rtol=1e-10, atol=1e-10)
    )

    # ---------------------------------------------------------
    # Basic properties of the distance matrix
    # ---------------------------------------------------------
    print("\nDistance matrix properties:")

    print("Shape:", D_loop.shape)

    print(
        "Maximum diagonal value:",
        np.max(np.abs(np.diag(D_loop)))
    )

    print(
        "Maximum symmetry error:",
        np.max(np.abs(D_loop - D_loop.T))
    )

    print("All distances non-negative:", np.all(D_loop >= 0))

    # ---------------------------------------------------------
    # Benchmark dataset required by the assignment
    # N = 2000, d = 20
    # ---------------------------------------------------------
    N = 2000
    d = 20

    X = rng.normal(size=(N, d))

    print("\n" + "=" * 60)
    print(f"BENCHMARK: N = {N}, d = {d}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Benchmark nested loops
    # ---------------------------------------------------------
    print("\nTiming nested loops...")

    loop_time = benchmark(
        pairwise_distances_loops,
        X,
        repeats=3
    )

    print(f"Nested loops: {loop_time:.6f} seconds")

    # ---------------------------------------------------------
    # Benchmark broadcasting
    # ---------------------------------------------------------
    print("\nTiming broadcasting...")

    broadcast_time = benchmark(
        pairwise_distances_broadcast,
        X,
        repeats=3
    )

    print(f"Broadcasting: {broadcast_time:.6f} seconds")

    # ---------------------------------------------------------
    # Benchmark norm expansion
    # ---------------------------------------------------------
    print("\nTiming norm expansion...")

    norm_time = benchmark(
        pairwise_distances_norm_expansion,
        X,
        repeats=3
    )

    print(f"Norm expansion: {norm_time:.6f} seconds")

    # ---------------------------------------------------------
    # Speedups
    # ---------------------------------------------------------
    print("\nSpeedups relative to nested loops:")

    print(
        f"Broadcasting speedup: "
        f"{loop_time / broadcast_time:.2f}x"
    )

    print(
        f"Norm expansion speedup: "
        f"{loop_time / norm_time:.2f}x"
    )

    print("\nSpeedup between vectorized methods:")

    print(
        f"Norm expansion vs Broadcasting: "
        f"{broadcast_time / norm_time:.2f}x"
    )

    # ---------------------------------------------------------
    # Large-dataset correctness verification
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("LARGE-DATASET CORRECTNESS CHECK")
    print("=" * 60)

    D_broadcast = pairwise_distances_broadcast(X)
    D_norm = pairwise_distances_norm_expansion(X)

    max_difference = np.max(np.abs(D_broadcast - D_norm))

    print(
        "Maximum absolute difference:",
        max_difference
    )

    print(
        "Maximum absolute difference between "
        "broadcasting and norm expansion:",
        max_difference
        )

    # ---------------------------------------------------------
    # Memory discussion
    # ---------------------------------------------------------
    difference_memory_bytes = N * N * d * 8
    difference_memory_mib = difference_memory_bytes / (1024 ** 2)

    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS")
    print("=" * 60)

    print(
        f"Broadcasting difference array size: "
        f"{difference_memory_mib:.2f} MiB"
    )

    print(
        "The broadcasting implementation creates an "
        "N x N x d intermediate array."
    )

    print(
        "Therefore, its memory usage scales as O(N^2 d)."
    )

    print(
        "The norm-expansion implementation avoids this "
        "N x N x d intermediate and uses O(N^2) memory "
        "for the distance matrix."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()