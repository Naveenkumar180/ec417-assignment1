"""
EC-417 Assignment 1
Problem 2: Polynomial Regression, Underfitting and Overfitting

This program:
1. Generates training and independent test data.
2. Fits polynomial models of selected degrees.
3. Visualizes underfitting and overfitting.
4. Computes training and test RMSE for degrees 0-12.
5. Repeats the experiment for n=100.
6. Prints the observations needed for the written answers.
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

SEED = 42

NOISE_STD = 0.3

SELECTED_DEGREES = [0, 1, 3, 9]
ALL_DEGREES = range(13)

FIGURE_DIR = "figures"


# ============================================================
# True data-generating function
# ============================================================

def true_function(x):
    """
    True regression function:

        f*(x) = sin(2*pi*x)
    """
    return np.sin(2 * np.pi * x)


# ============================================================
# Data generation
# ============================================================

def generate_data(n, rng):
    """
    Generate independent samples from:

        x ~ Uniform(0, 1)
        y = f*(x) + epsilon

    where

        epsilon ~ N(0, 0.3^2)
    """
    x = rng.uniform(0, 1, size=n)
    noise = rng.normal(0, NOISE_STD, size=n)

    y = true_function(x) + noise

    return x, y


# ============================================================
# Polynomial fitting
# ============================================================

def fit_polynomial(x_train, y_train, degree):
    """
    Fit a polynomial after scaling x from [0, 1] to [-1, 1].

    Scaling improves numerical stability for high-degree
    polynomial regression.
    """
    x_scaled = 2.0 * x_train - 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning)

        coefficients = np.polyfit(
            x_scaled,
            y_train,
            degree
        )

    return coefficients


def predict_polynomial(coefficients, x):
    """
    Evaluate the fitted polynomial after applying the same
    x-scaling used during training.
    """
    x_scaled = 2.0 * x - 1.0

    return np.polyval(
        coefficients,
        x_scaled
    )


# ============================================================
# RMSE
# ============================================================

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ============================================================
# Experiment for one training-set size
# ============================================================

def evaluate_degrees(n, rng, x_test, y_test):
    """
    Fit polynomial degrees 0-12 and calculate training/test RMSE.

    Returns:
        degrees
        training_rmse
        test_rmse
    """

    x_train, y_train = generate_data(n, rng)

    degrees = list(ALL_DEGREES)

    training_rmse = []
    test_rmse = []

    for degree in degrees:

        coefficients = fit_polynomial(
            x_train,
            y_train,
            degree
        )

        train_predictions = predict_polynomial(
            coefficients,
            x_train
        )

        test_predictions = predict_polynomial(
            coefficients,
            x_test
        )

        training_rmse.append(
            rmse(y_train, train_predictions)
        )

        test_rmse.append(
            rmse(y_test, test_predictions)
        )

    return (
        x_train,
        y_train,
        degrees,
        np.array(training_rmse),
        np.array(test_rmse)
    )


# ============================================================
# Part (b): Plot selected polynomial fits
# ============================================================

def plot_selected_fits(
    x_train,
    y_train,
    degrees,
    filename
):
    """
    Create one 2x2 figure for polynomial degrees:
    0, 1, 3, 9.
    """

    x_plot = np.linspace(0, 1, 1000)
    y_true = true_function(x_plot)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )

    axes = axes.ravel()

    for ax, degree in zip(axes, degrees):

        coefficients = fit_polynomial(
            x_train,
            y_train,
            degree
        )

        y_pred = predict_polynomial(
            coefficients,
            x_plot
        )

        ax.scatter(
            x_train,
            y_train,
            label="Training data"
        )

        ax.plot(
            x_plot,
            y_true,
            label="True function"
        )

        ax.plot(
            x_plot,
            y_pred,
            label=f"Polynomial degree {degree}"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_title(
            f"Polynomial Degree M = {degree}"
        )

        ax.legend()

        ax.grid(alpha=0.3)

    fig.suptitle(
        "Polynomial Regression: Underfitting and Overfitting",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Part (c) and (d): RMSE curves
# ============================================================

def plot_rmse_curves(
    degrees,
    training_rmse,
    test_rmse,
    n,
    filename
):
    """
    Plot training and test RMSE against polynomial degree.
    """

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        degrees,
        training_rmse,
        marker="o",
        label="Training RMSE"
    )

    ax.plot(
        degrees,
        test_rmse,
        marker="o",
        label="Test RMSE"
    )

    ax.set_xlabel("Polynomial Degree M")
    ax.set_ylabel("RMSE")

    ax.set_title(
        f"Training and Test RMSE vs Polynomial Degree (n = {n})"
    )

    ax.set_xticks(degrees)

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Print experiment results
# ============================================================

def print_results(
    n,
    degrees,
    training_rmse,
    test_rmse
):
    """
    Print RMSE table and identify important degrees.
    """

    print("\n" + "=" * 70)
    print(f"RESULTS FOR TRAINING SIZE n = {n}")
    print("=" * 70)

    print(
        f"{'Degree':>8}"
        f"{'Train RMSE':>18}"
        f"{'Test RMSE':>18}"
        f"{'Gap':>18}"
    )

    print("-" * 70)

    gaps = test_rmse - training_rmse

    for degree, train_error, test_error, gap in zip(
        degrees,
        training_rmse,
        test_rmse,
        gaps
    ):

        print(
            f"{degree:>8}"
            f"{train_error:>18.6f}"
            f"{test_error:>18.6f}"
            f"{gap:>18.6f}"
        )

    # Minimum training RMSE
    min_train_index = np.argmin(training_rmse)
    min_train_degree = degrees[min_train_index]

    # Minimum test RMSE
    min_test_index = np.argmin(test_rmse)
    min_test_degree = degrees[min_test_index]

    # Largest generalization gap
    largest_gap_index = np.argmax(gaps)
    largest_gap_degree = degrees[largest_gap_index]

    print("\nImportant observations:")

    print(
        f"Degree with minimum training RMSE: "
        f"M = {min_train_degree}"
    )

    print(
        f"Degree with minimum test RMSE: "
        f"M = {min_test_degree}"
    )

    print(
        "Does the degree with minimum test RMSE also "
        "minimize training RMSE?",
        min_test_degree == min_train_degree
    )

    print(
        f"Degree with largest generalization gap: "
        f"M = {largest_gap_degree}"
    )

    print(
        f"Largest generalization gap: "
        f"{gaps[largest_gap_index]:.6f}"
    )

    return (
        min_train_degree,
        min_test_degree,
        largest_gap_degree
    )


# ============================================================
# Main program
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("EC-417 ASSIGNMENT 1 - PROBLEM 2")
    print("Polynomial Regression, Underfitting and Overfitting")
    print("=" * 70)

    print(f"\nRandom seed: {SEED}")
    print(f"Noise standard deviation: {NOISE_STD}")

    # --------------------------------------------------------
    # Part (a): Generate independent test set
    # --------------------------------------------------------

    n_test = 1000

    x_test, y_test = generate_data(
        n_test,
        rng
    )

    print("\n" + "=" * 70)
    print("PART (a): DATA GENERATION")
    print("=" * 70)

    print(
        f"Training sample size for first experiment: n = 15"
    )

    print(
        f"Independent test sample size: n = {n_test}"
    )

    print(
        "x distribution: Uniform(0, 1)"
    )

    print(
        "Noise distribution: Normal(0, 0.3^2)"
    )

    # --------------------------------------------------------
    # Part (b): n = 15, selected polynomial degrees
    # --------------------------------------------------------

    (
        x_train_15,
        y_train_15,
        _,
        _,
        _
    ) = evaluate_degrees(
        n=15,
        rng=rng,
        x_test=x_test,
        y_test=y_test
    )

    plot_selected_fits(
        x_train_15,
        y_train_15,
        SELECTED_DEGREES,
        os.path.join(
            FIGURE_DIR,
            "problem2_polynomial_fits.png"
        )
    )

    print("\nPart (b):")
    print(
        "Saved 2x2 polynomial fit figure to:"
    )
    print(
        "figures/problem2_polynomial_fits.png"
    )

    # --------------------------------------------------------
    # Part (c): n = 15, degrees 0-12
    # --------------------------------------------------------

    (
        x_train_15,
        y_train_15,
        degrees_15,
        training_rmse_15,
        test_rmse_15
    ) = evaluate_degrees(
        n=15,
        rng=np.random.default_rng(SEED + 1),
        x_test=x_test,
        y_test=y_test
    )

    print("\n" + "=" * 70)
    print("PART (c): n = 15")
    print("=" * 70)

    print_results(
        n=15,
        degrees=degrees_15,
        training_rmse=training_rmse_15,
        test_rmse=test_rmse_15
    )

    plot_rmse_curves(
        degrees_15,
        training_rmse_15,
        test_rmse_15,
        n=15,
        filename=os.path.join(
            FIGURE_DIR,
            "problem2_rmse_n15.png"
        )
    )

    print(
        "\nSaved RMSE plot to:"
    )

    print(
        "figures/problem2_rmse_n15.png"
    )

    # --------------------------------------------------------
    # Part (d): n = 100
    # --------------------------------------------------------

    (
        x_train_100,
        y_train_100,
        degrees_100,
        training_rmse_100,
        test_rmse_100
    ) = evaluate_degrees(
        n=100,
        rng=np.random.default_rng(SEED + 2),
        x_test=x_test,
        y_test=y_test
    )

    print("\n" + "=" * 70)
    print("PART (d): n = 100")
    print("=" * 70)

    print_results(
        n=100,
        degrees=degrees_100,
        training_rmse=training_rmse_100,
        test_rmse=test_rmse_100
    )

    plot_rmse_curves(
        degrees_100,
        training_rmse_100,
        test_rmse_100,
        n=100,
        filename=os.path.join(
            FIGURE_DIR,
            "problem2_rmse_n100.png"
        )
    )

    print(
        "\nSaved RMSE plot to:"
    )

    print(
        "figures/problem2_rmse_n100.png"
    )

    # --------------------------------------------------------
    # Part (e): Bias, variance, or noise?
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PART (e): BIAS-VARIANCE INTERPRETATION")
    print("=" * 70)

    print(
        """
When the training-set size increases from n = 15 to n = 100,
the main quantity expected to change is VARIANCE.

A larger training set generally makes the fitted model less
sensitive to the particular random training samples. Therefore,
the variance of the estimator decreases.

The underlying noise level is fixed at sigma = 0.3, so the
irreducible noise does not change.

The bias is primarily determined by model complexity and the
relationship between the model class and the true function.
Increasing n does not fundamentally change the model class.
"""
    )

    # --------------------------------------------------------
    # Part (f): Why can degree 9 overfit?
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PART (f): WHY DEGREE 9 CAN OVERFIT")
    print("=" * 70)

    print(
        """
A degree-9 polynomial has enough flexibility to closely follow
the 15 noisy training observations. This can produce a very small
training RMSE.

However, part of the fitted curve may be responding to random
noise rather than the underlying function sin(2*pi*x). As a
result, the polynomial can perform substantially worse on
unseen test data.

Therefore, a very small training RMSE alone does not imply that
the model is a good predictive model. Generalization performance
on independent test data is also important.
"""
    )

    print("\n" + "=" * 70)
    print("PROBLEM 2 COMPLETE")
    print("=" * 70)

    print(
        "\nGenerated figures are stored in the 'figures/' directory."
    )


if __name__ == "__main__":
    main()