from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    SRC_DIR
    / "results"
    / "plots"
    / "results"
    / "domain_factor_f1_results.csv"
)

OUTPUT_DIR = (
    SRC_DIR
    / "results"
    / "plots"
    / "results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "feature_f1_distribution_all_domains.png"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "feature_f1_distribution_summary.csv"
)


FEATURE_ORDER = [
    "All features",
    "RSSI-only",
    "Baseline + TX/RX",
    "Baseline",
    "TX/RX-only",
]


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}\n\n"
            "Run plot_domain_factor_performance.py first."
        )

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "feature",
        "domain",
        "f1",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns:\n"
            + ", ".join(sorted(missing_columns))
        )

    dataframe = dataframe[
        dataframe["feature"].isin(FEATURE_ORDER)
    ].copy()

    if dataframe.empty:
        raise RuntimeError(
            "No valid feature results were found."
        )

    return dataframe


def save_summary(
    dataframe: pd.DataFrame,
) -> None:
    summary = (
        dataframe
        .groupby(
            "feature",
            observed=True,
        )["f1"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )

    summary["feature"] = pd.Categorical(
        summary["feature"],
        categories=FEATURE_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("feature")
        .reset_index(drop=True)
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nSummary statistics:")
    print(summary.to_string(index=False))


def plot_distribution(
    dataframe: pd.DataFrame,
) -> None:
    values_by_feature = [
        dataframe.loc[
            dataframe["feature"] == feature,
            "f1",
        ].to_numpy()
        for feature in FEATURE_ORDER
    ]

    positions = np.arange(
        1,
        len(FEATURE_ORDER) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 5.8)
    )

    ax.boxplot(
        values_by_feature,
        positions=positions,
        widths=0.52,
        showfliers=False,
        medianprops={
            "linewidth": 2,
        },
    )

    random_generator = np.random.default_rng(42)

    for position, values in zip(
        positions,
        values_by_feature,
    ):
        jitter = random_generator.normal(
            loc=0,
            scale=0.055,
            size=len(values),
        )

        ax.scatter(
            np.full(
                len(values),
                position,
            ) + jitter,
            values,
            s=24,
            alpha=0.35,
        )

        median = np.median(values)

        
        ax.text(
            position,
            1.035,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_xticks(positions)

    ax.set_xticklabels(
        FEATURE_ORDER,
        rotation=12,
        ha="right",
    )

    ax.set_ylabel("F1-score")
    ax.set_xlabel("Feature configuration")

    
    ax.set_ylim(
        -0.03,
        1.075,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    
    fig.subplots_adjust(
        top=0.92,
        bottom=0.22,
        left=0.09,
        right=0.98,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nGenerated:\n{OUTPUT_PATH}")

def main() -> None:
    dataframe = load_data()

    print(
        f"Loaded {len(dataframe)} results."
    )

    print(
        dataframe.groupby("feature").size()
    )

    save_summary(dataframe)

    plot_distribution(dataframe)


if __name__ == "__main__":
    main()