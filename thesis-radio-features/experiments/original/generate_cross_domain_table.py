from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    SRC_DIR
    / "results"
    / "plots"
    / "cross_domain"
    / "cross_domain_feature_summary.csv"
)

OUTPUT_PATH = (
    SRC_DIR
    / "results"
    / "plots"
    / "cross_domain"
    / "cross_domain_summary_table.tex"
)

FEATURE_ORDER = [
    "TX/RX-only",
    "RSSI-only",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "feature",
        "count",
        "mean",
        "median",
        "std",
        "q1",
        "q3",
        "count_f1_below_0.5",
        "proportion_f1_below_0.5",
        "count_f1_equal_0",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing columns in summary CSV:\n"
            + ", ".join(sorted(missing_columns))
        )

    dataframe["feature"] = pd.Categorical(
        dataframe["feature"],
        categories=FEATURE_ORDER,
        ordered=True,
    )

    dataframe = (
        dataframe
        .sort_values("feature")
        .reset_index(drop=True)
    )

    dataframe["iqr"] = (
        dataframe["q3"] - dataframe["q1"]
    )

    dataframe["below_05"] = dataframe.apply(
        lambda row: (
            f"{int(row['count_f1_below_0.5'])} "
            f"({100 * row['proportion_f1_below_0.5']:.1f}\\%)"
        ),
        axis=1,
    )

    table_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Summary statistics for off-diagonal cross-domain F1-scores.}",
        r"\label{tab:cross_domain_summary}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        (
            r"Feature setting & Mean & Median & Std. & IQR "
            r"& $F1<0.5$ & $F1=0$ \\"
        ),
        r"\hline",
    ]

    for _, row in dataframe.iterrows():
        table_lines.append(
            f"{row['feature']} "
            f"& {row['mean']:.3f} "
            f"& {row['median']:.3f} "
            f"& {row['std']:.3f} "
            f"& {row['iqr']:.3f} "
            f"& {row['below_05']} "
            f"& {int(row['count_f1_equal_0'])} "
            r"\\"
        )

    table_lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(table_lines),
        encoding="utf-8",
    )

    print(f"Generated:\n{OUTPUT_PATH}")
    print("\nTable preview:\n")
    print("\n".join(table_lines))


if __name__ == "__main__":
    main()