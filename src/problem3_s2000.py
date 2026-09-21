import numpy as np

SIGMA = 0.3
GRID = np.linspace(0, 1, 100)

def true_function(x):
    return np.sin(2 * np.pi * x)

def generate_dataset(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = true_function(x) + rng.normal(0, SIGMA, n)
    return x, y

def run(degree, S, seed):
    rng = np.random.default_rng(seed)
    predictions = []

    for _ in range(S):
        x, y = generate_dataset(
            15,
            int(rng.integers(0, 2**32 - 1))
        )
        predictions.append(
            np.polyval(
                np.polyfit(x, y, degree),
                GRID
            )
        )

    predictions = np.array(predictions)

    mean_prediction = predictions.mean(axis=0)

    bias2 = (
        mean_prediction - true_function(GRID)
    ) ** 2

    variance = (
        (predictions - mean_prediction) ** 2
    ).mean(axis=0)

    fresh_noise = rng.normal(
        0,
        SIGMA,
        size=(S, len(GRID))
    )

    fresh_y = (
        true_function(GRID)[None, :]
        + fresh_noise
    )

    simulated_total = (
        (fresh_y - predictions) ** 2
    ).mean(axis=0)

    return (
        bias2.mean(),
        variance.mean(),
        simulated_total.mean()
    )

print("=" * 60)
print("PROBLEM 3 — S = 2000 VERIFICATION")
print("=" * 60)

for degree in [1, 3, 9]:

    bias2, variance, total = run(
        degree,
        2000,
        2026 + degree
    )

    decomposition = (
        SIGMA**2
        + bias2
        + variance
    )

    discrepancy = abs(
        decomposition - total
    )

    print(f"\nM = {degree}")
    print(f"Bias²                 = {bias2:.6f}")
    print(f"Variance              = {variance:.6f}")
    print(f"Sum                   = {decomposition:.6f}")
    print(f"Simulated Total       = {total:.6f}")
    print(f"Discrepancy            = {discrepancy:.6f}")

print("\nThe discrepancy is caused by Monte Carlo sampling error.")
print("Increasing S from 200 to 2000 gives a more accurate")
print("approximation of the theoretical bias-variance identity.")
