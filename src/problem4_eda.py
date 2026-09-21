import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():

    Path("figures").mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("PROBLEM 4 — CALIFORNIA HOUSING EDA")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    data = fetch_california_housing(as_frame=True)

    X = data.data.copy()
    y = data.target.copy()

    df = X.copy()
    df["MedHouseVal"] = y

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSummary statistics:")
    print(df.describe().T)

    # ------------------------------------------------------------
    # Target distribution
    # ------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        y,
        bins=50
    )

    plt.xlabel("Median House Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Median House Value")

    plt.tight_layout()

    plt.savefig(
        "figures/problem4_target_hist.png",
        dpi=150
    )

    plt.close()

    print("\nMaximum target value:", y.max())

    print(
        "\nThe target has a hard upper limit at 5.0. "
        "This indicates top-coding/right-censoring."
    )

    # ------------------------------------------------------------
    # Predictor distributions
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(12, 14)
    )

    for ax, column in zip(
        axes.ravel(),
        X.columns
    ):

        ax.hist(
            X[column],
            bins=50
        )

        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "figures/problem4_predictor_histograms.png",
        dpi=150
    )

    plt.close()

    print("\nPredictor ranges:")

    for column in X.columns:

        print(
            f"{column:12s} "
            f"min={X[column].min():.4f} "
            f"max={X[column].max():.4f} "
            f"skew={X[column].skew():.3f}"
        )

    # ------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------

    correlation = df.corr()

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=60,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title(
        "California Housing Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/problem4_correlation_heatmap.png",
        dpi=150
    )

    plt.close()

    # Target correlations

    target_corr = (
        correlation["MedHouseVal"]
        .drop("MedHouseVal")
        .abs()
        .sort_values(ascending=False)
    )

    print(
        "\nStrongest target association:"
    )

    print(
        target_corr.head()
    )

    # ------------------------------------------------------------
    # Geographic analysis
    # ------------------------------------------------------------

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        df["Longitude"],
        df["Latitude"],
        c=y,
        s=3
    )

    plt.colorbar(
        scatter,
        label="Median House Value"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title(
        "California Housing Geographic Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/problem4_geographic_scatter.png",
        dpi=150
    )

    plt.close()

    print(
        "\nThe geographic plot shows strong spatial structure."
    )

    print(
        "High-value regions are concentrated around major "
        "coastal metropolitan areas, especially the Bay Area "
        "and Southern California."
    )

    # ------------------------------------------------------------
    # Data leakage demonstration
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Correct preprocessing
    scaler_train = StandardScaler()

    X_train_scaled = scaler_train.fit_transform(
        X_train
    )

    X_test_scaled_correct = scaler_train.transform(
        X_test
    )

    # Incorrect preprocessing using full data
    scaler_all = StandardScaler()

    scaler_all.fit(X)

    X_test_scaled_leaky = scaler_all.transform(
        X_test
    )

    correct_means = X_test_scaled_correct.mean(
        axis=0
    )

    leaky_means = X_test_scaled_leaky.mean(
        axis=0
    )

    comparison = pd.DataFrame(
        {
            "correct_train_fitted": correct_means,
            "incorrect_full_data": leaky_means,
            "difference": (
                leaky_means - correct_means
            )
        },
        index=X.columns
    )

    print(
        "\nData leakage comparison:"
    )

    print(comparison)

    comparison.to_csv(
        "figures/problem4_leakage_comparison.csv"
    )

    print(
        "\nCorrect procedure: fit preprocessing "
        "only on training data."
    )

    print(
        "Using the entire dataset allows information "
        "from the test set to influence preprocessing "
        "and constitutes data leakage."
    )

    print("\nPROBLEM 4 COMPLETE")


if __name__ == "__main__":
    main()