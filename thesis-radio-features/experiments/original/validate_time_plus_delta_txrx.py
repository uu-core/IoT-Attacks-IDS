from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import validate_txrx_time_confound as base


# ============================================================
# Output paths
# ============================================================

OUTPUT_DIR = (
    base.SRC_DIR
    / "results"
    / "time_plus_delta_validation"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "time_plus_delta_all_runs.csv"
)

DOMAIN_OUTPUT = (
    OUTPUT_DIR
    / "time_plus_delta_by_domain.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "time_plus_delta_summary.csv"
)

ATTACK_OUTPUT = (
    OUTPUT_DIR
    / "time_plus_delta_by_attack.csv"
)

INCREMENT_DOMAIN_OUTPUT = (
    OUTPUT_DIR
    / "incremental_value_by_domain.csv"
)

INCREMENT_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "incremental_value_summary.csv"
)

PLOT_OUTPUT = (
    OUTPUT_DIR
    / "time_plus_delta_f1.png"
)

INCREMENT_PLOT_OUTPUT = (
    OUTPUT_DIR
    / "incremental_f1_gain.png"
)


# Previous result files used for paired comparisons
ABSOLUTE_TIME_DOMAIN_FILE = (
    base.SRC_DIR
    / "results"
    / "temporal_validation"
    / "temporal_validation_by_domain.csv"
)

RELATIVE_TIME_DOMAIN_FILE = (
    base.SRC_DIR
    / "results"
    / "relative_time_validation"
    / "relative_time_by_domain.csv"
)


# ============================================================
# Conditions
# ============================================================

CONDITIONS = [
    "absolute_time_plus_differenced_txrx",
    "relative_time_plus_differenced_txrx",
]

TXRX_COLUMNS = [
    "tx",
    "tx.1",
    "rx",
    "rx.1",
]


# ============================================================
# Feature construction
# ============================================================

def load_combined_dataframe(
    path: Path,
    condition: str,
) -> pd.DataFrame:
    """
    Creates one of two combined feature sets:

    1. absolute time + differenced TX/RX
    2. relative time + differenced TX/RX

    Differencing is applied to the already aggregated TX/RX
    columns in the CSV file. This is a diagnostic analysis.
    """

    dataframe = pd.read_csv(
        path,
        index_col=0,
    )

    required_columns = set(
        TXRX_COLUMNS + ["label"]
    )

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing columns in:\n{path}\n"
            f"{sorted(missing_columns)}"
        )

    labels = (
        pd.to_numeric(
            dataframe["label"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    differenced_txrx = (
        dataframe[TXRX_COLUMNS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .diff()
        .fillna(0.0)
    )

    differenced_txrx.columns = [
        "delta_tx_mean",
        "delta_tx_std",
        "delta_rx_mean",
        "delta_rx_std",
    ]

    if condition == (
        "absolute_time_plus_differenced_txrx"
    ):
        time_feature = pd.DataFrame(
            {
                "absolute_time":
                    np.arange(
                        len(dataframe),
                        dtype=float,
                    )
            },
            index=dataframe.index,
        )

    elif condition == (
        "relative_time_plus_differenced_txrx"
    ):
        denominator = max(
            len(dataframe) - 1,
            1,
        )

        time_feature = pd.DataFrame(
            {
                "relative_time":
                    np.arange(
                        len(dataframe),
                        dtype=float,
                    ) / denominator
            },
            index=dataframe.index,
        )

    else:
        raise ValueError(
            f"Unsupported condition: {condition}"
        )

    output = pd.concat(
        [
            time_feature,
            differenced_txrx,
        ],
        axis=1,
    )

    output["label"] = labels.to_numpy()

    return output


# Replace the loader used inside the imported validation code
base.load_run_dataframe = (
    load_combined_dataframe
)


# ============================================================
# Run validation
# ============================================================

def run_validation(
    domains: list[dict[str, object]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    total_jobs = (
        len(domains)
        * len(base.FILE_SPLIT_SEEDS)
        * len(CONDITIONS)
    )

    job_number = 0

    for domain in domains:
        domain_name = str(
            domain["canonical_domain"]
        )

        csv_files = list(
            domain["csv_files"]
        )

        for split_seed in base.FILE_SPLIT_SEEDS:
            (
                training_paths,
                test_paths,
            ) = base.create_file_split(
                csv_files,
                split_seed,
            )

            for condition in CONDITIONS:
                job_number += 1

                print(
                    f"[{job_number}/{total_jobs}] "
                    f"{domain_name} | "
                    f"seed={split_seed} | "
                    f"{condition}"
                )

                (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                ) = base.construct_datasets(
                    training_paths,
                    test_paths,
                    condition,
                )

                metrics = (
                    base.train_and_evaluate(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        split_seed,
                    )
                )

                records.append(
                    {
                        "domain":
                            domain_name,
                        "attack":
                            domain["attack"],
                        "nodes":
                            domain["node"],
                        "variant":
                            domain["variant"],
                        "condition":
                            condition,
                        "seed":
                            split_seed,
                        "train_sequences":
                            len(y_train),
                        "test_sequences":
                            len(y_test),
                        **metrics,
                    }
                )

                # Save after every model
                pd.DataFrame(
                    records
                ).to_csv(
                    DETAIL_OUTPUT,
                    index=False,
                )

    return pd.DataFrame(records)


# ============================================================
# Aggregate combined-condition results
# ============================================================

def aggregate_results(
    detailed: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    by_domain = (
        detailed
        .groupby(
            [
                "domain",
                "attack",
                "nodes",
                "variant",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_test_loss=(
                "test_loss",
                "mean",
            ),
        )
    )

    by_domain.to_csv(
        DOMAIN_OUTPUT,
        index=False,
    )

    summary = (
        by_domain
        .groupby(
            "condition",
            as_index=False,
        )
        .agg(
            number_of_domains=(
                "domain",
                "count",
            ),
            mean_f1=(
                "mean_f1",
                "mean",
            ),
            median_f1=(
                "mean_f1",
                "median",
            ),
            std_f1=(
                "mean_f1",
                "std",
            ),
            minimum_f1=(
                "mean_f1",
                "min",
            ),
            maximum_f1=(
                "mean_f1",
                "max",
            ),
            mean_auc=(
                "mean_auc",
                "mean",
            ),
            median_auc=(
                "mean_auc",
                "median",
            ),
            std_auc=(
                "mean_auc",
                "std",
            ),
            mean_accuracy=(
                "mean_accuracy",
                "mean",
            ),
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    by_attack = (
        by_domain
        .groupby(
            [
                "attack",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            number_of_domains=(
                "domain",
                "count",
            ),
            mean_f1=(
                "mean_f1",
                "mean",
            ),
            std_f1=(
                "mean_f1",
                "std",
            ),
            median_f1=(
                "mean_f1",
                "median",
            ),
            mean_auc=(
                "mean_auc",
                "mean",
            ),
            std_auc=(
                "mean_auc",
                "std",
            ),
        )
    )

    by_attack.to_csv(
        ATTACK_OUTPUT,
        index=False,
    )

    return (
        by_domain,
        summary,
        by_attack,
    )


# ============================================================
# Load previous time-only results
# ============================================================

def load_absolute_time_only() -> pd.DataFrame:
    if not ABSOLUTE_TIME_DOMAIN_FILE.exists():
        raise FileNotFoundError(
            "Absolute-time result file not found:\n"
            f"{ABSOLUTE_TIME_DOMAIN_FILE}"
        )

    dataframe = pd.read_csv(
        ABSOLUTE_TIME_DOMAIN_FILE
    )

    dataframe = dataframe.loc[
        dataframe["condition"] == "time_only"
    ].copy()

    if dataframe.empty:
        raise RuntimeError(
            "No time_only rows found in:\n"
            f"{ABSOLUTE_TIME_DOMAIN_FILE}"
        )

    dataframe["condition"] = (
        "absolute_time_only"
    )

    return dataframe


def load_relative_time_only() -> pd.DataFrame:
    if not RELATIVE_TIME_DOMAIN_FILE.exists():
        raise FileNotFoundError(
            "Relative-time result file not found:\n"
            f"{RELATIVE_TIME_DOMAIN_FILE}"
        )

    dataframe = pd.read_csv(
        RELATIVE_TIME_DOMAIN_FILE
    )

    if "condition" in dataframe.columns:
        dataframe = dataframe.loc[
            dataframe["condition"]
            == "relative_time"
        ].copy()

    if dataframe.empty:
        raise RuntimeError(
            "No relative-time rows found in:\n"
            f"{RELATIVE_TIME_DOMAIN_FILE}"
        )

    dataframe["condition"] = (
        "relative_time_only"
    )

    return dataframe


# ============================================================
# Incremental-value analysis
# ============================================================

def safe_wilcoxon(
    differences: pd.Series,
) -> tuple[float, float]:
    values = (
        differences
        .dropna()
        .to_numpy()
    )

    if len(values) == 0:
        return np.nan, np.nan

    if np.allclose(values, 0):
        return 0.0, 1.0

    try:
        statistic, p_value = wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return (
            float(statistic),
            float(p_value),
        )

    except ValueError:
        return np.nan, np.nan


def compare_pair(
    time_only: pd.DataFrame,
    combined: pd.DataFrame,
    comparison_name: str,
) -> pd.DataFrame:
    time_columns = [
        "domain",
        "mean_f1",
        "mean_auc",
        "mean_accuracy",
    ]

    combined_columns = [
        "domain",
        "mean_f1",
        "mean_auc",
        "mean_accuracy",
    ]

    merged = time_only[
        time_columns
    ].merge(
        combined[
            combined_columns
        ],
        on="domain",
        suffixes=(
            "_time_only",
            "_time_plus_delta",
        ),
        how="inner",
    )

    merged["comparison"] = (
        comparison_name
    )

    merged["delta_f1"] = (
        merged[
            "mean_f1_time_plus_delta"
        ]
        - merged[
            "mean_f1_time_only"
        ]
    )

    merged["delta_auc"] = (
        merged[
            "mean_auc_time_plus_delta"
        ]
        - merged[
            "mean_auc_time_only"
        ]
    )

    merged["delta_accuracy"] = (
        merged[
            "mean_accuracy_time_plus_delta"
        ]
        - merged[
            "mean_accuracy_time_only"
        ]
    )

    return merged


def create_incremental_analysis(
    combined_by_domain: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    absolute_time = (
        load_absolute_time_only()
    )

    relative_time = (
        load_relative_time_only()
    )

    absolute_combined = (
        combined_by_domain.loc[
            combined_by_domain[
                "condition"
            ]
            == (
                "absolute_time_plus_"
                "differenced_txrx"
            )
        ].copy()
    )

    relative_combined = (
        combined_by_domain.loc[
            combined_by_domain[
                "condition"
            ]
            == (
                "relative_time_plus_"
                "differenced_txrx"
            )
        ].copy()
    )

    absolute_comparison = compare_pair(
        absolute_time,
        absolute_combined,
        "absolute_time",
    )

    relative_comparison = compare_pair(
        relative_time,
        relative_combined,
        "relative_time",
    )

    incremental_by_domain = pd.concat(
        [
            absolute_comparison,
            relative_comparison,
        ],
        ignore_index=True,
    )

    incremental_by_domain.to_csv(
        INCREMENT_DOMAIN_OUTPUT,
        index=False,
    )

    summary_records = []

    for comparison_name, group in (
        incremental_by_domain
        .groupby("comparison")
    ):
        f1_stat, f1_p = safe_wilcoxon(
            group["delta_f1"]
        )

        auc_stat, auc_p = safe_wilcoxon(
            group["delta_auc"]
        )

        summary_records.append(
            {
                "comparison":
                    comparison_name,
                "number_of_domains":
                    len(group),
                "mean_delta_f1":
                    group[
                        "delta_f1"
                    ].mean(),
                "median_delta_f1":
                    group[
                        "delta_f1"
                    ].median(),
                "std_delta_f1":
                    group[
                        "delta_f1"
                    ].std(),
                "domains_f1_improved":
                    int(
                        (
                            group["delta_f1"]
                            > 0
                        ).sum()
                    ),
                "domains_f1_declined":
                    int(
                        (
                            group["delta_f1"]
                            < 0
                        ).sum()
                    ),
                "mean_delta_auc":
                    group[
                        "delta_auc"
                    ].mean(),
                "median_delta_auc":
                    group[
                        "delta_auc"
                    ].median(),
                "std_delta_auc":
                    group[
                        "delta_auc"
                    ].std(),
                "domains_auc_improved":
                    int(
                        (
                            group["delta_auc"]
                            > 0
                        ).sum()
                    ),
                "domains_auc_declined":
                    int(
                        (
                            group["delta_auc"]
                            < 0
                        ).sum()
                    ),
                "wilcoxon_f1_statistic":
                    f1_stat,
                "wilcoxon_f1_p_value":
                    f1_p,
                "wilcoxon_auc_statistic":
                    auc_stat,
                "wilcoxon_auc_p_value":
                    auc_p,
            }
        )

    incremental_summary = pd.DataFrame(
        summary_records
    )

    incremental_summary.to_csv(
        INCREMENT_SUMMARY_OUTPUT,
        index=False,
    )

    return (
        incremental_by_domain,
        incremental_summary,
    )


# ============================================================
# Plots
# ============================================================

def create_combined_f1_plot(
    combined_by_domain: pd.DataFrame,
) -> None:
    condition_order = CONDITIONS

    values = [
        combined_by_domain.loc[
            combined_by_domain[
                "condition"
            ] == condition,
            "mean_f1",
        ].to_numpy()
        for condition in condition_order
    ]

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.boxplot(
        values,
        tick_labels=[
            "Absolute time\n+ delta TX/RX",
            "Relative time\n+ delta TX/RX",
        ],
        showmeans=True,
    )

    random_generator = (
        np.random.default_rng(42)
    )

    for position, condition_values in enumerate(
        values,
        start=1,
    ):
        jitter = (
            random_generator.normal(
                0,
                0.045,
                size=len(condition_values),
            )
        )

        axis.scatter(
            np.full(
                len(condition_values),
                position,
            ) + jitter,
            condition_values,
            alpha=0.5,
            s=22,
        )

    axis.set_ylabel(
        "Mean F1-score across three seeds"
    )

    axis.set_xlabel(
        "Combined feature condition"
    )

    axis.set_ylim(
        -0.03,
        1.05,
    )

    axis.spines[
        ["top", "right"]
    ].set_visible(False)

    figure.tight_layout()

    figure.savefig(
        PLOT_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_increment_plot(
    incremental_by_domain: pd.DataFrame,
) -> None:
    comparison_order = [
        "absolute_time",
        "relative_time",
    ]

    values = [
        incremental_by_domain.loc[
            incremental_by_domain[
                "comparison"
            ] == comparison,
            "delta_f1",
        ].to_numpy()
        for comparison in comparison_order
    ]

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.boxplot(
        values,
        tick_labels=[
            "Delta TX/RX added\nto absolute time",
            "Delta TX/RX added\nto relative time",
        ],
        showmeans=True,
    )

    random_generator = (
        np.random.default_rng(42)
    )

    for position, comparison_values in enumerate(
        values,
        start=1,
    ):
        jitter = (
            random_generator.normal(
                0,
                0.045,
                size=len(comparison_values),
            )
        )

        axis.scatter(
            np.full(
                len(comparison_values),
                position,
            ) + jitter,
            comparison_values,
            alpha=0.5,
            s=22,
        )

    axis.axhline(
        0.0,
        linewidth=1,
    )

    axis.set_ylabel(
        "Change in mean F1-score"
    )

    axis.set_xlabel(
        "Paired comparison"
    )

    axis.spines[
        ["top", "right"]
    ].set_visible(False)

    figure.tight_layout()

    figure.savefig(
        INCREMENT_PLOT_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        f"Device: {base.DEVICE}"
    )

    print(
        f"Data root: {base.DATA_ROOT}"
    )

    print(
        "Conditions:"
    )

    for condition in CONDITIONS:
        print(
            f"  - {condition}"
        )

    domains = (
        base.discover_domains()
    )

    detailed_results = (
        run_validation(
            domains
        )
    )

    detailed_results.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    (
        by_domain,
        summary,
        by_attack,
    ) = aggregate_results(
        detailed_results
    )

    (
        incremental_by_domain,
        incremental_summary,
    ) = create_incremental_analysis(
        by_domain
    )

    create_combined_f1_plot(
        by_domain
    )

    create_increment_plot(
        incremental_by_domain
    )

    print(
        "\nCombined-condition summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nResults by attack:"
    )

    print(
        by_attack.to_string(
            index=False
        )
    )

    print(
        "\nIncremental-value summary:"
    )

    print(
        incremental_summary.to_string(
            index=False
        )
    )

    print(
        "\nGenerated files:"
    )

    for path in [
        DETAIL_OUTPUT,
        DOMAIN_OUTPUT,
        SUMMARY_OUTPUT,
        ATTACK_OUTPUT,
        INCREMENT_DOMAIN_OUTPUT,
        INCREMENT_SUMMARY_OUTPUT,
        PLOT_OUTPUT,
        INCREMENT_PLOT_OUTPUT,
    ]:
        print(path)


if __name__ == "__main__":
    main()