import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SIGMA = 0.3
GRID = np.linspace(0, 1, 200)


def true_function(x):
    return np.sin(2 * np.pi * x)


def generate_dataset(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = true_function(x) + rng.normal(0, SIGMA, n)
    return x, y


def fit_predict(x, y, degree, x_eval):
    coefficients = np.polyfit(x, y, degree)
    return np.polyval(coefficients, x_eval)


def simulate(degrees, S, seed):
    rng = np.random.default_rng(seed)
    results = {}

    for degree in degrees:
        predictions = []

        for _ in range(S):
            dataset_seed = int(rng.integers(0, 2**32 - 1))
            x, y = generate_dataset(15, dataset_seed)
            predictions.append(
                fit_predict(x, y, degree, GRID)
            )

        predictions = np.array(predictions)

        mean_prediction = predictions.mean(axis=0)

        bias_squared = (
            mean_prediction - true_function(GRID)
        ) ** 2

        variance = (
            (predictions - mean_prediction) ** 2
        ).mean(axis=0)

        results[degree] = {
            "predictions": predictions,
            "mean": mean_prediction,
            "bias2": bias_squared,
            "variance": variance,
        }

    return results


def main():

    Path("figures").mkdir(exist_ok=True)

    degrees = [1, 3, 9]

    results = simulate(
        degrees=degrees,
        S=200,
        seed=42
    )

    print("\n" + "=" * 70)
    print("PROBLEM 3 — BIAS-VARIANCE SIMULATION")
    print("=" * 70)

    for degree in degrees:

        r = results[degree]

        bias2 = r["bias2"].mean()
        variance = r["variance"].mean()

        theoretical_test_error = (
            SIGMA ** 2
            + bias2
            + variance
        )

        print(f"\nDegree M = {degree}")
        print(f"Noise variance       : {SIGMA**2:.6f}")
        print(f"Bias^2               : {bias2:.6f}")
        print(f"Variance             : {variance:.6f}")
        print(
            f"Bias² + Variance + Noise : "
            f"{theoretical_test_error:.6f}"
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            GRID,
            true_function(GRID),
            linewidth=2,
            label="True function"
        )

        plt.plot(
            GRID,
            r["mean"],
            linewidth=2,
            linestyle="--",
            label="Average prediction"
        )

        for prediction in r["predictions"][:20]:
            plt.plot(
                GRID,
                prediction,
                alpha=0.15
            )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            f"Bias-Variance Simulation — Degree {degree}"
        )
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"figures/problem3_fits_M{degree}.png",
            dpi=150
        )

        plt.close()

    # Full degree sweep
    all_degrees = range(13)

    full_results = simulate(
        degrees=all_degrees,
        S=200,
        seed=123
    )

    bias_values = []
    variance_values = []

    for degree in all_degrees:
        bias_values.append(
            full_results[degree]["bias2"].mean()
        )

        variance_values.append(
            full_results[degree]["variance"].mean()
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(all_degrees),
        bias_values,
        marker="o",
        label="Bias²"
    )

    plt.plot(
        list(all_degrees),
        variance_values,
        marker="o",
        label="Variance"
    )

    plt.xlabel("Polynomial degree")
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "figures/problem3_bias_variance_vs_degree.png",
        dpi=150
    )

    plt.close()

    print("\nPROBLEM 3 COMPLETE")


if __name__ == "__main__":
    main()