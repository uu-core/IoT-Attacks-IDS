from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SRC_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "plots" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment definitions
# ============================================================

EXPERIMENTS = {
    1: "All features",
    2: "RSSI-only",
    3: "Baseline + TX/RX",
    4: "Baseline",
    5: "TX/RX-only",
}

FEATURE_ORDER = [
    "All features",
    "RSSI-only",
    "Baseline + TX/RX",
    "Baseline",
    "TX/RX-only",
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

NODE_ORDER = [5, 10, 15, 20]

VARIANT_ORDER = [
    "Base",
    "On-off",
    "Gradual change",
]


# ============================================================
# Domain parsing
# ============================================================

def parse_domain(
    domain_name: str,
) -> tuple[str, int, str]:
    """
    Example:
        dis_flooding_15_gc
        worst_parent_5_base
    """
    try:
        attack_key, node_text, variant_key = (
            domain_name.rsplit("_", 2)
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse domain name: {domain_name}"
        ) from exc

    if attack_key not in ALLOWED_ATTACKS:
        raise ValueError(
            f"Unknown or excluded attack: {domain_name}"
        )

    if variant_key not in VARIANT_LABELS:
        raise ValueError(
            f"Unknown variant: {domain_name}"
        )

    return (
        ALLOWED_ATTACKS[attack_key],
        int(node_text),
        VARIANT_LABELS[variant_key],
    )


def is_included_domain(
    domain_name: str,
) -> bool:
    try:
        attack_key, node_text, variant_key = (
            domain_name.rsplit("_", 2)
        )
    except ValueError:
        return False

    return (
        attack_key in ALLOWED_ATTACKS
        and node_text.isdigit()
        and variant_key in VARIANT_LABELS
    )


# ============================================================
# Read F1 from metrics.json
# ============================================================

def find_f1_recursively(
    value: Any,
) -> float | None:
    accepted_keys = {
        "f1",
        "f1_score",
        "f1-score",
        "f1score",
    }

    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()

            if (
                normalized_key in accepted_keys
                and isinstance(item, (int, float))
            ):
                return float(item)

        for item in value.values():
            result = find_f1_recursively(item)

            if result is not None:
                return result

    if isinstance(value, list):
        for item in value:
            result = find_f1_recursively(item)

            if result is not None:
                return result

    return None


def read_f1(
    metrics_path: Path,
) -> float:
    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    f1 = find_f1_recursively(metrics)

    if f1 is None:
        raise KeyError(
            f"No F1-score found in:\n{metrics_path}"
        )

    if not 0 <= f1 <= 1:
        raise ValueError(
            f"Invalid F1-score {f1} in:\n{metrics_path}"
        )

    return f1


# ============================================================
# Load all five experiments
# ============================================================

def load_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for exp_number, feature_name in EXPERIMENTS.items():
        experiment_dir = (
            RESULTS_DIR
            / f"exp_features_{exp_number}"
        )

        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"Directory not found:\n{experiment_dir}"
            )

        for domain_dir in sorted(
            experiment_dir.iterdir()
        ):
            if not domain_dir.is_dir():
                continue

            domain_name = domain_dir.name

            # Excludes failing_node and unrelated folders
            if not is_included_domain(domain_name):
                continue

            metrics_path = (
                domain_dir / "metrics.json"
            )

            if not metrics_path.exists():
                print(
                    f"Skipped: metrics.json not found in "
                    f"{domain_dir}"
                )
                continue

            attack, nodes, variant = parse_domain(
                domain_name
            )

            f1 = read_f1(metrics_path)

            rows.append(
                {
                    "experiment": exp_number,
                    "feature": feature_name,
                    "domain": domain_name,
                    "attack": attack,
                    "nodes": nodes,
                    "variant": variant,
                    "f1": f1,
                }
            )

    if not rows:
        raise RuntimeError(
            "No valid metrics.json results were loaded."
        )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        OUTPUT_DIR
        / "domain_factor_f1_results.csv",
        index=False,
    )

    print(
        f"Loaded {len(dataframe)} domain results."
    )

    print(
        dataframe.groupby("feature").size()
    )

    return dataframe


# ============================================================
# Statistics
# ============================================================

def calculate_statistics(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
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


# ============================================================
# Draw grouped bars
# ============================================================

def draw_grouped_bars(
    ax: plt.Axes,
    statistics: pd.DataFrame,
    group_column: str,
    category_order: list,
    x_label: str,
) -> None:
    x_positions = np.arange(
        len(category_order)
    )

    number_of_features = len(FEATURE_ORDER)
    total_group_width = 0.82
    bar_width = (
        total_group_width
        / number_of_features
    )

    for feature_index, feature_name in enumerate(
        FEATURE_ORDER
    ):
        feature_statistics = (
            statistics.loc[
                statistics["feature"]
                == feature_name
            ]
            .set_index(group_column)
            .reindex(category_order)
        )

        offset = (
            feature_index
            - (number_of_features - 1) / 2
        ) * bar_width

        ax.bar(
            x_positions + offset,
            feature_statistics["mean"].to_numpy(),
            width=bar_width,
            yerr=feature_statistics["std"].to_numpy(),
            capsize=2.5,
            label=feature_name,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(category_order)

    ax.set_xlabel(x_label)
    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        alpha=0.25,
    )


# ============================================================
# Generate figure
# ============================================================

def plot_domain_factor_performance(
    dataframe: pd.DataFrame,
) -> None:
    node_statistics = calculate_statistics(
        dataframe,
        "nodes",
    )

    variant_statistics = calculate_statistics(
        dataframe,
        "variant",
    )

    node_statistics.to_csv(
        OUTPUT_DIR
        / "f1_by_network_size.csv",
        index=False,
    )

    variant_statistics.to_csv(
        OUTPUT_DIR
        / "f1_by_behavioral_variant.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.3),
        sharey=True,
    )

    draw_grouped_bars(
        ax=axes[0],
        statistics=node_statistics,
        group_column="nodes",
        category_order=NODE_ORDER,
        x_label="Network size",
    )

    axes[0].set_ylabel("Mean F1-score")

    draw_grouped_bars(
        ax=axes[1],
        statistics=variant_statistics,
        group_column="variant",
        category_order=VARIANT_ORDER,
        x_label="Behavioral variant",
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.91)
    )

    output_path = (
        OUTPUT_DIR
        / "feature_performance_by_node_and_variant.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated:\n{output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    dataframe = load_results()

    plot_domain_factor_performance(
        dataframe
    )


if __name__ == "__main__":
    main()