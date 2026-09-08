from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ATTACKS = (
    "dis_flooding",
    "local_repair",
    "blackhole",
    "worst_parent",
)

START_TIMES = (300, 450, 600)
RUNS = range(1, 11)
HORIZONS = (10, 60)

FEATURES = (
    "delta_txraw_mean",
    "delta_txraw_std",
    "delta_rxraw_mean",
    "delta_rxraw_std",
)


def run_path(
    root: Path,
    attack: str,
    start_time: int,
    run_number: int,
) -> Path:
    return (
        root
        / attack
        / f"start_{start_time}"
        / f"run_{run_number:02d}"
        / "features_temporal_validation.csv"
    )


def paired_cohens_dz(differences: pd.Series) -> float:
    values = pd.to_numeric(
        differences,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return float("nan")

    standard_deviation = values.std(ddof=1)

    if standard_deviation == 0:
        if values.mean() == 0:
            return 0.0
        return float(
            np.sign(values.mean()) * np.inf
        )

    return float(
        values.mean() / standard_deviation
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path.home()
            / "ids-WPLR"
            / "applications"
            / "example-attacks"
            / "validation_outputs"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path.home()
            / "ids-WPLR"
            / "temporal_validation"
            / "tx_rx_effect_results"
        ),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, object]] = []

    for attack in ATTACKS:
        for start_time in START_TIMES:
            for run_number in RUNS:
                path = run_path(
                    args.data_root,
                    attack,
                    start_time,
                    run_number,
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing file: {path}"
                    )

                df = pd.read_csv(path)

                minutes = pd.to_numeric(
                    df["minute_start"],
                    errors="coerce",
                )

                for horizon in HORIZONS:
                    pre_mask = (
                        (minutes >= start_time - horizon)
                        & (minutes < start_time)
                    )

                    post_mask = (
                        (minutes >= start_time)
                        & (
                            minutes
                            < start_time + horizon
                        )
                    )

                    if pre_mask.sum() != horizon:
                        raise ValueError(
                            f"{path}: expected {horizon} "
                            f"pre-attack rows, found "
                            f"{int(pre_mask.sum())}"
                        )

                    if post_mask.sum() != horizon:
                        raise ValueError(
                            f"{path}: expected {horizon} "
                            f"post-attack rows, found "
                            f"{int(post_mask.sum())}"
                        )

                    for feature in FEATURES:
                        values = pd.to_numeric(
                            df[feature],
                            errors="coerce",
                        )

                        pre_mean = float(
                            values.loc[pre_mask].mean()
                        )

                        post_mean = float(
                            values.loc[post_mask].mean()
                        )

                        records.append(
                            {
                                "attack": attack,
                                "start_time":
                                    start_time,
                                "run":
                                    run_number,
                                "horizon_minutes":
                                    horizon,
                                "feature":
                                    feature,
                                "pre_mean":
                                    pre_mean,
                                "post_mean":
                                    post_mean,
                                "difference":
                                    post_mean
                                    - pre_mean,
                                "absolute_difference":
                                    abs(
                                        post_mean
                                        - pre_mean
                                    ),
                            }
                        )

    run_level = pd.DataFrame(records)

    run_level.to_csv(
        args.output_dir
        / "tx_rx_effect_run_level.csv",
        index=False,
    )

    by_attack_start = (
        run_level
        .groupby(
            [
                "attack",
                "start_time",
                "horizon_minutes",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            mean_pre=(
                "pre_mean",
                "mean",
            ),
            mean_post=(
                "post_mean",
                "mean",
            ),
            mean_difference=(
                "difference",
                "mean",
            ),
            std_difference=(
                "difference",
                "std",
            ),
            mean_absolute_difference=(
                "absolute_difference",
                "mean",
            ),
            runs_increased=(
                "difference",
                lambda s: int(
                    (s > 0).sum()
                ),
            ),
            runs_unchanged=(
                "difference",
                lambda s: int(
                    (s == 0).sum()
                ),
            ),
            runs_decreased=(
                "difference",
                lambda s: int(
                    (s < 0).sum()
                ),
            ),
        )
    )

    dz_start = (
        run_level
        .groupby(
            [
                "attack",
                "start_time",
                "horizon_minutes",
                "feature",
            ],
        )["difference"]
        .apply(paired_cohens_dz)
        .reset_index(
            name="cohens_dz"
        )
    )

    by_attack_start = (
        by_attack_start.merge(
            dz_start,
            on=[
                "attack",
                "start_time",
                "horizon_minutes",
                "feature",
            ],
            how="left",
        )
    )

    by_attack_start.to_csv(
        args.output_dir
        / "tx_rx_effect_by_attack_start.csv",
        index=False,
    )

    by_attack = (
        run_level
        .groupby(
            [
                "attack",
                "horizon_minutes",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            mean_pre=(
                "pre_mean",
                "mean",
            ),
            mean_post=(
                "post_mean",
                "mean",
            ),
            mean_difference=(
                "difference",
                "mean",
            ),
            std_difference=(
                "difference",
                "std",
            ),
            mean_absolute_difference=(
                "absolute_difference",
                "mean",
            ),
            runs_increased=(
                "difference",
                lambda s: int(
                    (s > 0).sum()
                ),
            ),
            runs_unchanged=(
                "difference",
                lambda s: int(
                    (s == 0).sum()
                ),
            ),
            runs_decreased=(
                "difference",
                lambda s: int(
                    (s < 0).sum()
                ),
            ),
        )
    )

    dz_attack = (
        run_level
        .groupby(
            [
                "attack",
                "horizon_minutes",
                "feature",
            ],
        )["difference"]
        .apply(paired_cohens_dz)
        .reset_index(
            name="cohens_dz"
        )
    )

    by_attack = by_attack.merge(
        dz_attack,
        on=[
            "attack",
            "horizon_minutes",
            "feature",
        ],
        how="left",
    )

    by_attack["absolute_cohens_dz"] = (
        by_attack["cohens_dz"].abs()
    )

    by_attack.to_csv(
        args.output_dir
        / "tx_rx_effect_by_attack.csv",
        index=False,
    )

    tx_rx_comparison = (
        by_attack.assign(
            radio_metric=np.where(
                by_attack["feature"]
                .str.contains("txraw"),
                "TX",
                "RX",
            ),
            statistic=np.where(
                by_attack["feature"]
                .str.endswith("_mean"),
                "mean",
                "std",
            ),
        )
        [
            [
                "attack",
                "horizon_minutes",
                "radio_metric",
                "statistic",
                "mean_pre",
                "mean_post",
                "mean_difference",
                "cohens_dz",
                "absolute_cohens_dz",
            ]
        ]
        .sort_values(
            [
                "attack",
                "horizon_minutes",
                "statistic",
                "radio_metric",
            ]
        )
    )

    tx_rx_comparison.to_csv(
        args.output_dir
        / "tx_vs_rx_effect_comparison.csv",
        index=False,
    )

    print("Completed.")
    print(
        args.output_dir
        / "tx_rx_effect_run_level.csv"
    )
    print(
        args.output_dir
        / "tx_rx_effect_by_attack_start.csv"
    )
    print(
        args.output_dir
        / "tx_rx_effect_by_attack.csv"
    )
    print(
        args.output_dir
        / "tx_vs_rx_effect_comparison.csv"
    )


if __name__ == "__main__":
    main()
