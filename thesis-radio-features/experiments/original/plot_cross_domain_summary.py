from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
CROSS_TEST_DIR = SRC_DIR / "results" / "cross_test"
OUTPUT_DIR = SRC_DIR / "results" / "plots" / "cross_domain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = {
    2: "RSSI-only",
    5: "TX/RX-only",
}

FEATURE_ORDER = [
    "TX/RX-only",
    "RSSI-only",
]


ALLOWED_ATTACKS = {
    "blackhole": "Blackhole",
    "dis_flooding": "DIS Flooding",
    "local_repair": "Local Repair",
    "worst_parent": "Worst Parent",
}

VARIANT_LABELS = {
    "base": "Base",
    "oo": "On-off",
    "gc": "Gradual change",
}

ATTACK_ORDER = [
    "Blackhole",
    "DIS Flooding",
    "Local Repair",
    "Worst Parent",
]

NODE_ORDER = [
    5,
    10,
    15,
    20,
]

VARIANT_ORDER = [
    "Base",
    "On-off",
    "Gradual change",
]


# ============================================================
# Domain parsing
# ============================================================

def parse_domain(domain_name: str) -> tuple[str, int, str]:
    """
    Examples:
        dis_flooding_5_base
        local_repair_20_gc
    """
    try:
        attack_key, node_text, variant_key = domain_name.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse domain name: {domain_name}\n"
            "Expected format: attack_nodecount_variant"
        ) from exc

    if attack_key not in ALLOWED_ATTACKS:
        raise ValueError(
            f"Unknown or excluded attack domain: {domain_name}"
        )

    if variant_key not in VARIANT_LABELS:
        raise ValueError(
            f"Unknown variant in domain: {domain_name}"
        )

    try:
        node_count = int(node_text)
    except ValueError as exc:
        raise ValueError(
            f"Node count is not an integer: {domain_name}"
        ) from exc

    return (
        ALLOWED_ATTACKS[attack_key],
        node_count,
        VARIANT_LABELS[variant_key],
    )


def is_included_domain(domain_name: str) -> bool:
    """
    Filters out failing_node and unrelated files before parsing.
    """
    try:
        attack_key, node_text, variant_key = domain_name.rsplit("_", 2)
    except ValueError:
        return False

    return (
        attack_key in ALLOWED_ATTACKS
        and node_text.isdigit()
        and variant_key in VARIANT_LABELS
    )


# ============================================================
# JSON F1 extraction
# ============================================================

def find_f1_recursively(value: Any) -> float | None:
    """
    Supports common formats such as:

        {"f1": 0.85}
        {"f1_score": 0.85}
        {"metrics": {"f1": 0.85}}
        {"results": {"f1_score": 0.85}}
    """
    accepted_keys = {
        "f1",
        "f1score",
        "f1_score",
        "f1-score",
    }

    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = (
                str(key)
                .strip()
                .lower()
                .replace(" ", "_")
            )

            if (
                normalized_key in accepted_keys
                and isinstance(item, (int, float))
            ):
                return float(item)

        for item in value.values():
            found = find_f1_recursively(item)

            if found is not None:
                return found

    elif isinstance(value, list):
        for item in value:
            found = find_f1_recursively(item)

            if found is not None:
                return found

    return None


def extract_f1(
    data: dict[str, Any],
    json_path: Path,
) -> float:
    f1 = find_f1_recursively(data)

    if f1 is None:
        raise KeyError(
            f"No f1 or f1_score value found in: {json_path}"
        )

    if not 0.0 <= f1 <= 1.0:
        raise ValueError(
            f"F1 must be between 0 and 1. "
            f"Found {f1} in {json_path}"
        )

    return f1


# ============================================================
# Load cross-domain results
# ============================================================

def load_cross_domain_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    skipped_train_domains = 0
    skipped_test_files = 0

    for exp_number, feature_name in EXPERIMENTS.items():
        experiment_dir = (
            CROSS_TEST_DIR / f"exp{exp_number}"
        )

        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"Experiment directory not found:\n"
                f"{experiment_dir}"
            )

        
        for train_dir in sorted(experiment_dir.iterdir()):
            if not train_dir.is_dir():
                continue

            train_domain = train_dir.name

            
            if not is_included_domain(train_domain):
                skipped_train_domains += 1
                continue

            (
                train_attack,
                train_nodes,
                train_variant,
            ) = parse_domain(train_domain)

            
            
            for json_path in sorted(
                train_dir.glob("vs_*.json")
            ):
                test_domain = (
                    json_path
                    .stem
                    .removeprefix("vs_")
                )

                
                if not is_included_domain(test_domain):
                    skipped_test_files += 1
                    continue

                
                
                if train_domain == test_domain:
                    continue

                (
                    test_attack,
                    test_nodes,
                    test_variant,
                ) = parse_domain(test_domain)

                try:
                    with json_path.open(
                        "r",
                        encoding="utf-8",
                    ) as file:
                        result = json.load(file)

                    f1 = extract_f1(
                        result,
                        json_path,
                    )

                except (
                    OSError,
                    json.JSONDecodeError,
                    KeyError,
                    ValueError,
                ) as exc:
                    raise RuntimeError(
                        f"Failed to read result file:\n"
                        f"{json_path}\n\n"
                        f"{exc}"
                    ) from exc

                rows.append(
                    {
                        "experiment": exp_number,
                        "feature": feature_name,
                        "train_domain": train_domain,
                        "test_domain": test_domain,
                        "train_attack": train_attack,
                        "train_nodes": train_nodes,
                        "train_variant": train_variant,
                        "test_attack": test_attack,
                        "test_nodes": test_nodes,
                        "test_variant": test_variant,
                        "f1": f1,
                    }
                )

    if not rows:
        raise RuntimeError(
            "No valid cross-domain results were loaded.\n"
            f"Checked directory:\n{CROSS_TEST_DIR}"
        )

    dataframe = pd.DataFrame(rows)

    output_csv = (
        OUTPUT_DIR
        / "cross_domain_off_diagonal_results.csv"
    )

    dataframe.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"Loaded {len(dataframe)} "
        "cross-domain results."
    )

    print(
        f"Skipped {skipped_train_domains} "
        "excluded training-domain folders."
    )

    print(
        f"Skipped {skipped_test_files} "
        "excluded test JSON files."
    )

    print(
        f"Combined data saved to:\n{output_csv}"
    )

    return dataframe


# ============================================================
# Overall statistics
# ============================================================

def save_overall_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
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
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )

    below_half = (
        dataframe
        .assign(
            below_05=dataframe["f1"] < 0.5
        )
        .groupby(
            "feature",
            observed=True,
        )["below_05"]
        .agg(
            ["sum", "mean"]
        )
        .reset_index()
        .rename(
            columns={
                "sum": "count_f1_below_0.5",
                "mean": "proportion_f1_below_0.5",
            }
        )
    )

    zero_count = (
        dataframe
        .assign(
            is_zero=np.isclose(
                dataframe["f1"],
                0.0,
            )
        )
        .groupby(
            "feature",
            observed=True,
        )["is_zero"]
        .sum()
        .reset_index(
            name="count_f1_equal_0"
        )
    )

    summary = summary.merge(
        below_half,
        on="feature",
    )

    summary = summary.merge(
        zero_count,
        on="feature",
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

    output_csv = (
        OUTPUT_DIR
        / "cross_domain_feature_summary.csv"
    )

    summary.to_csv(
        output_csv,
        index=False,
    )

    print("\nOverall statistics:")
    print(summary.to_string(index=False))

    print(
        f"\nSummary saved to:\n{output_csv}"
    )

    return summary


# ============================================================
# Grouped statistics
# ============================================================

def grouped_statistics(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Calculate the mean and standard deviation of F1-scores
    for each target-domain group.
    """
    return (
        dataframe
        .groupby(
            [group_column, "feature"],
            observed=True,
        )["f1"]
        .agg(
            mean="mean",
            std="std",
            count="count",
        )
        .reset_index()
    )


def save_grouped_tables(
    dataframe: pd.DataFrame,
) -> None:
    grouped_statistics(
        dataframe,
        "test_attack",
    ).to_csv(
        OUTPUT_DIR
        / "cross_domain_by_test_attack.csv",
        index=False,
    )

    grouped_statistics(
        dataframe,
        "test_nodes",
    ).to_csv(
        OUTPUT_DIR
        / "cross_domain_by_test_node_count.csv",
        index=False,
    )

    grouped_statistics(
        dataframe,
        "test_variant",
    ).to_csv(
        OUTPUT_DIR
        / "cross_domain_by_test_variant.csv",
        index=False,
    )


# ============================================================
# Save figures
# ============================================================

def save_figure(
    fig: plt.Figure,
    file_stem: str,
) -> None:
    png_path = (
        OUTPUT_DIR / f"{file_stem}.png"
    )

    pdf_path = (
        OUTPUT_DIR / f"{file_stem}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated:\n{png_path}")
    print(f"Generated:\n{pdf_path}")


# ============================================================
# Figure 1: boxplot and individual points
# ============================================================

def plot_overall_distribution(
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
        figsize=(7.2, 5.2)
    )

    ax.boxplot(
        values_by_feature,
        positions=positions,
        widths=0.48,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": 1.8,
        },
    )

    rng = np.random.default_rng(42)

    for position, feature, values in zip(
        positions,
        FEATURE_ORDER,
        values_by_feature,
    ):
        jitter = rng.normal(
            0.0,
            0.055,
            size=len(values),
        )

        ax.scatter(
            np.full(
                len(values),
                position,
            ) + jitter,
            values,
            s=11,
            alpha=0.18,
            label=feature,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(FEATURE_ORDER)

    ax.set_ylabel(
        "Cross-domain F1-score"
    )

    ax.set_ylim(
        -0.03,
        1.03,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "cross_domain_f1_distribution",
    )


# ============================================================
# Figure 2: attack, node count and variant
# ============================================================

def draw_group_panel(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    group_column: str,
    category_order: list[Any],
    x_label: str,
) -> None:
    statistics = grouped_statistics(
        dataframe,
        group_column,
    )

    x_positions = np.arange(
        len(category_order)
    )

    offsets = {
        "TX/RX-only": -0.12,
        "RSSI-only": 0.12,
    }

    markers = {
        "TX/RX-only": "o",
        "RSSI-only": "s",
    }

    for feature in FEATURE_ORDER:
        feature_statistics = (
            statistics.loc[
                statistics["feature"] == feature
            ]
            .set_index(group_column)
            .reindex(category_order)
        )

        ax.errorbar(
            x_positions + offsets[feature],
            feature_statistics["mean"].to_numpy(),
            yerr=feature_statistics["std"].to_numpy(),
            marker=markers[feature],
            linestyle="none",
            capsize=4,
            markersize=6,
            label=feature,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(category_order)

    ax.set_xlabel(x_label)
    ax.set_ylim(-0.03, 1.03)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

def plot_grouped_summary(
    dataframe: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        sharey=True,
    )

    draw_group_panel(
        axes[0],
        dataframe,
        group_column="test_attack",
        category_order=ATTACK_ORDER,
        x_label="Test attack type",
    )

    axes[0].set_ylabel(
        "Mean cross-domain F1-score"
    )

    axes[0].tick_params(
        axis="x",
        rotation=20,
    )

    draw_group_panel(
        axes[1],
        dataframe,
        group_column="test_nodes",
        category_order=NODE_ORDER,
        x_label="Test network size",
    )

    draw_group_panel(
        axes[2],
        dataframe,
        group_column="test_variant",
        category_order=VARIANT_ORDER,
        x_label="Test behavioral variant",
    )

    axes[2].tick_params(
        axis="x",
        rotation=15,
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    save_figure(
        fig,
        "cross_domain_grouped_summary",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        f"Reading results from:\n"
        f"{CROSS_TEST_DIR}\n"
    )

    dataframe = load_cross_domain_results()

    save_overall_summary(dataframe)

    save_grouped_tables(dataframe)

    plot_overall_distribution(dataframe)

    plot_grouped_summary(dataframe)

    print(
        f"\nAll output files are in:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()