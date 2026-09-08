from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import validate_txrx_time_confound as base


# ============================================================
# Output paths
# ============================================================

OUTPUT_DIR = (
    base.SRC_DIR
    / "results"
    / "relative_time_validation"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "relative_time_all_runs.csv"
)

DOMAIN_OUTPUT = (
    OUTPUT_DIR
    / "relative_time_by_domain.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "relative_time_summary.csv"
)

ATTACK_OUTPUT = (
    OUTPUT_DIR
    / "relative_time_by_attack.csv"
)

PLOT_OUTPUT = (
    OUTPUT_DIR
    / "relative_time_f1.png"
)

COMPARISON_OUTPUT = (
    OUTPUT_DIR
    / "absolute_vs_relative_time_summary.csv"
)

PREVIOUS_SUMMARY = (
    base.SRC_DIR
    / "results"
    / "temporal_validation"
    / "temporal_validation_summary.csv"
)


# ============================================================
# Relative-time feature construction
# ============================================================

def load_relative_time_dataframe(
    path: Path,
    condition: str,
) -> pd.DataFrame:
    """
    Loads one CSV and constructs a single relative-time feature.

    relative_time = 0.0 at the beginning of the run
    relative_time = 1.0 at the end of the run
    """
    if condition != "relative_time":
        raise ValueError(
            f"Unsupported condition: {condition}"
        )

    dataframe = pd.read_csv(
        path,
        index_col=0,
    )

    if "label" not in dataframe.columns:
        raise KeyError(
            f"'label' column not found in:\n{path}"
        )

    labels = (
        pd.to_numeric(
            dataframe["label"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    denominator = max(
        len(dataframe) - 1,
        1,
    )

    relative_time = (
        np.arange(
            len(dataframe),
            dtype=float,
        )
        / denominator
    )

    output = pd.DataFrame(
        {
            "relative_time": relative_time,
            "label": labels.to_numpy(),
        },
        index=dataframe.index,
    )

    return output


# Replace only the data-loading function used internally
# by the existing validation script.
base.load_run_dataframe = (
    load_relative_time_dataframe
)


# ============================================================
# Run relative-time validation
# ============================================================

def run_validation(
    domains: list[dict[str, object]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    total_jobs = (
        len(domains)
        * len(base.FILE_SPLIT_SEEDS)
    )

    job_number = 0

    for domain in domains:
        domain_name = str(
            domain["canonical_domain"]
        )

        csv_files = list(
            domain["csv_files"]
        )

        for seed in base.FILE_SPLIT_SEEDS:
            job_number += 1

            print(
                f"[{job_number}/{total_jobs}] "
                f"{domain_name} | "
                f"seed={seed} | "
                f"relative_time"
            )

            (
                training_paths,
                test_paths,
            ) = base.create_file_split(
                csv_files,
                seed,
            )

            (
                X_train,
                y_train,
                X_test,
                y_test,
            ) = base.construct_datasets(
                training_paths,
                test_paths,
                "relative_time",
            )

            metrics = (
                base.train_and_evaluate(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    seed,
                )
            )

            records.append(
                {
                    "domain": domain_name,
                    "attack": domain["attack"],
                    "nodes": domain["node"],
                    "variant":
                        domain["variant"],
                    "condition":
                        "relative_time",
                    "seed": seed,
                    "train_sequences":
                        len(y_train),
                    "test_sequences":
                        len(y_test),
                    **metrics,
                }
            )

            # Save continuously so progress is not lost
            pd.DataFrame(
                records
            ).to_csv(
                DETAIL_OUTPUT,
                index=False,
            )

    return pd.DataFrame(records)


# ============================================================
# Aggregate results
# ============================================================

def create_outputs(
    detailed: pd.DataFrame,
) -> None:
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
            mean_f1=(
                "f1",
                "mean",
            ),
            std_f1=(
                "f1",
                "std",
            ),
            mean_auc=(
                "auc",
                "mean",
            ),
            std_auc=(
                "auc",
                "std",
            ),
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

    summary = pd.DataFrame(
        [
            {
                "condition":
                    "relative_time",
                "number_of_domains":
                    len(by_domain),
                "mean_f1":
                    by_domain[
                        "mean_f1"
                    ].mean(),
                "median_f1":
                    by_domain[
                        "mean_f1"
                    ].median(),
                "std_f1":
                    by_domain[
                        "mean_f1"
                    ].std(),
                "minimum_f1":
                    by_domain[
                        "mean_f1"
                    ].min(),
                "maximum_f1":
                    by_domain[
                        "mean_f1"
                    ].max(),
                "mean_auc":
                    by_domain[
                        "mean_auc"
                    ].mean(),
                "median_auc":
                    by_domain[
                        "mean_auc"
                    ].median(),
                "std_auc":
                    by_domain[
                        "mean_auc"
                    ].std(),
                "mean_accuracy":
                    by_domain[
                        "mean_accuracy"
                    ].mean(),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    by_attack = (
        by_domain
        .groupby(
            "attack",
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

    create_f1_plot(
        by_domain
    )

    create_absolute_relative_comparison(
        summary
    )

    print("\nRelative-time summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nRelative-time results by attack:")
    print(
        by_attack.to_string(
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
        PLOT_OUTPUT,
    ]:
        print(path)

    if COMPARISON_OUTPUT.exists():
        print(COMPARISON_OUTPUT)


# ============================================================
# Plot
# ============================================================

def create_f1_plot(
    by_domain: pd.DataFrame,
) -> None:
    values = (
        by_domain["mean_f1"]
        .to_numpy()
    )

    figure, axis = plt.subplots(
        figsize=(6, 6)
    )

    axis.boxplot(
        [values],
        tick_labels=[
            "Relative\ntime",
        ],
        showmeans=True,
    )

    random_generator = (
        np.random.default_rng(42)
    )

    jitter = (
        random_generator.normal(
            0,
            0.045,
            size=len(values),
        )
    )

    axis.scatter(
        np.ones(
            len(values)
        ) + jitter,
        values,
        alpha=0.5,
        s=24,
    )

    axis.set_ylabel(
        "Mean F1-score across three seeds"
    )

    axis.set_xlabel(
        "Feature condition"
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

    plt.close(
        figure
    )


# ============================================================
# Compare with previous absolute-time result
# ============================================================

def create_absolute_relative_comparison(
    relative_summary: pd.DataFrame,
) -> None:
    if not PREVIOUS_SUMMARY.exists():
        print(
            "\nPrevious temporal summary not found:"
        )
        print(PREVIOUS_SUMMARY)
        return

    previous = pd.read_csv(
        PREVIOUS_SUMMARY
    )

    if "condition" not in previous.columns:
        print(
            "\nPrevious summary has no "
            "'condition' column."
        )
        return

    absolute = previous.loc[
        previous["condition"]
        == "time_only"
    ].copy()

    if absolute.empty:
        print(
            "\nNo 'time_only' row found in:"
        )
        print(PREVIOUS_SUMMARY)
        return

    available_columns = [
        column
        for column in [
            "condition",
            "mean_f1",
            "median_f1",
            "std_f1",
            "minimum_f1",
            "maximum_f1",
            "mean_auc",
        ]
        if column in absolute.columns
    ]

    absolute = absolute[
        available_columns
    ].copy()

    absolute["condition"] = (
        "absolute_time"
    )

    relative_columns = [
        column
        for column in available_columns
        if column
        in relative_summary.columns
    ]

    relative = relative_summary[
        relative_columns
    ].copy()

    comparison = pd.concat(
        [
            absolute[
                relative_columns
            ],
            relative,
        ],
        ignore_index=True,
    )

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    print(
        "\nAbsolute-time versus "
        "relative-time comparison:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )


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
        "Condition: relative_time"
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

    create_outputs(
        detailed_results
    )


if __name__ == "__main__":
    main()