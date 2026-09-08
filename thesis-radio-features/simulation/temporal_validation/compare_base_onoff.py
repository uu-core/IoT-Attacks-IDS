#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path.home() / "ids-WPLR"

BASE = (
    ROOT
    / "temporal_validation"
    / "cross_start_results_4attacks"
    / "cross_start_summary.csv"
)

ONOFF = (
    ROOT
    / "temporal_validation"
    / "cross_start_results_onoff"
    / "cross_start_summary.csv"
)

OUT = (
    ROOT
    / "temporal_validation"
    / "base_vs_onoff_results"
)

FEATURES = [
    "interval_txrx",
    "interval_rpl",
    "interval_rpl_plus_txrx",
]

ATTACK_ORDER = [
    "dis_flooding",
    "local_repair",
    "blackhole",
    "worst_parent",
]


def load_summary(path: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df[
        df["feature_set"].isin(FEATURES)
    ].copy()

    # Average across the three held-out start times.
    result = (
        df.groupby(
            ["attack", "feature_set"],
            as_index=False,
        )
        .agg(
            mean_f1=("mean_f1", "mean"),
            mean_auc=("mean_auc", "mean"),
        )
    )

    result["variant"] = variant

    return result


def main():
    if not BASE.exists():
        raise FileNotFoundError(BASE)

    if not ONOFF.exists():
        raise FileNotFoundError(ONOFF)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = load_summary(
        BASE,
        "base",
    )

    onoff = load_summary(
        ONOFF,
        "on-off",
    )

    combined = pd.concat(
        [base, onoff],
        ignore_index=True,
    )

    combined.to_csv(
        OUT / "base_vs_onoff_long.csv",
        index=False,
    )

    f1 = combined.pivot(
        index="attack",
        columns=["feature_set", "variant"],
        values="mean_f1",
    )

    rows = []

    for attack in ATTACK_ORDER:
        for feature in FEATURES:

            base_f1 = float(
                f1.loc[
                    attack,
                    (feature, "base"),
                ]
            )

            onoff_f1 = float(
                f1.loc[
                    attack,
                    (feature, "on-off"),
                ]
            )

            rows.append(
                {
                    "attack": attack,
                    "feature_set": feature,
                    "base_f1": base_f1,
                    "onoff_f1": onoff_f1,
                    "delta_onoff_minus_base": (
                        onoff_f1 - base_f1
                    ),
                }
            )

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        OUT / "base_vs_onoff_f1.csv",
        index=False,
    )

    print()
    print("BASE VS ON-OFF")
    print("=" * 86)

    print(
        comparison.to_string(
            index=False,
            formatters={
                "base_f1": "{:.3f}".format,
                "onoff_f1": "{:.3f}".format,
                "delta_onoff_minus_base":
                    "{:+.3f}".format,
            },
        )
    )

    print()
    print("Saved:")
    print(
        OUT / "base_vs_onoff_f1.csv"
    )


if __name__ == "__main__":
    main()
