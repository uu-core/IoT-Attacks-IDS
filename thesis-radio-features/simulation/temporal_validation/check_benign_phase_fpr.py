from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train_cross_start_validation_4attacks as base


PHASES = {
    "early_0_300": (0, 300),
    "middle_300_600": (300, 600),
    "late_600_900": (600, 900),
}


def make_benign_phase_windows(
    df: pd.DataFrame,
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
    phase_start: int,
    phase_end: int,
) -> np.ndarray:
    values = df[columns].apply(pd.to_numeric, errors="coerce").copy()
    denominator = (global_max - global_min).replace(0, 1)
    values = (values - global_min) / denominator
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    minute_start = pd.to_numeric(df["minute_start"], errors="coerce").to_numpy()

    windows: list[np.ndarray] = []

    for start in range(0, len(df) - base.WINDOW + 1):
        end = start + base.WINDOW
        window_minutes = minute_start[start:end]

        if (
            np.all(window_minutes >= phase_start)
            and np.all(window_minutes < phase_end)
        ):
            windows.append(
                values.iloc[start:end].to_numpy(dtype=np.float32)
            )

    if not windows:
        return np.empty(
            (0, base.WINDOW, len(columns)),
            dtype=np.float32,
        )

    return np.asarray(windows, dtype=np.float32)


def build_phase_dataset(
    paths: list[Path],
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
    phase_start: int,
    phase_end: int,
) -> np.ndarray:
    parts = [
        make_benign_phase_windows(
            base.load_run(path),
            columns,
            global_min,
            global_max,
            phase_start,
            phase_end,
        )
        for path in paths
    ]

    nonempty = [part for part in parts if len(part) > 0]

    if not nonempty:
        return np.empty(
            (0, base.WINDOW, len(columns)),
            dtype=np.float32,
        )

    return np.concatenate(nonempty, axis=0)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, int, float]:
    base.set_seed(seed)

    train_loader = base.make_loader(
        x_train,
        y_train,
        True,
        seed,
    )

    val_loader = base.make_loader(
        x_val,
        y_val,
        False,
        seed,
    )

    model = base.TrueSequenceLSTM(
        x_train.shape[2]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base.LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    epochs = 0

    for epoch in range(1, base.MAX_EPOCHS + 1):
        model.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_loss, _ = base.evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        epochs = epoch

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= base.PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(
            "No best model state was saved"
        )

    model.load_state_dict(best_state)

    return model, epochs, float(best_val_loss)


def false_positive_rate(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> tuple[float, int, int]:
    if len(x) == 0:
        return float("nan"), 0, 0

    model.eval()
    predictions: list[int] = []

    with torch.no_grad():
        for start in range(0, len(x), base.BATCH_SIZE):
            batch = torch.tensor(
                x[start:start + base.BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )

            logits = model(batch)
            predicted = logits.argmax(dim=1)
            predictions.extend(
                predicted.cpu().numpy().tolist()
            )

    predictions_array = np.asarray(
        predictions,
        dtype=int,
    )

    false_positives = int(
        np.sum(predictions_array == 1)
    )

    total = int(
        len(predictions_array)
    )

    return (
        false_positives / total,
        false_positives,
        total,
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
            / "benign_phase_results"
        ),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")

    records: list[dict[str, object]] = []

    total_jobs = (
        len(base.ATTACKS)
        * len(base.START_TIMES)
        * len(base.FEATURE_SETS)
        * len(base.SEEDS)
    )

    job_number = 0

    for attack in base.ATTACKS:
        for held_out_start in base.START_TIMES:
            (
                train_paths,
                val_paths,
                _,
            ) = base.split_paths(
                args.data_root,
                attack,
                held_out_start,
            )

            benign_test_paths = base.benign_paths(
                args.data_root,
                range(9, 11),
            )

            base.validate_paths(
                benign_test_paths
            )

            for feature_name, columns in base.FEATURE_SETS.items():
                global_min, global_max = base.fit_minmax(
                    train_paths,
                    columns,
                )

                x_train, y_train = base.build_dataset(
                    train_paths,
                    columns,
                    global_min,
                    global_max,
                )

                x_val, y_val = base.build_dataset(
                    val_paths,
                    columns,
                    global_min,
                    global_max,
                )

                phase_datasets = {
                    phase_name: build_phase_dataset(
                        benign_test_paths,
                        columns,
                        global_min,
                        global_max,
                        phase_start,
                        phase_end,
                    )
                    for phase_name, (
                        phase_start,
                        phase_end,
                    ) in PHASES.items()
                }

                for seed in base.SEEDS:
                    job_number += 1

                    print(
                        f"[{job_number}/{total_jobs}] "
                        f"{attack} | "
                        f"test_start={held_out_start} | "
                        f"{feature_name} | "
                        f"seed={seed}"
                    )

                    model, epochs, best_val_loss = train_model(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        seed,
                        device,
                    )

                    for phase_name, x_phase in phase_datasets.items():
                        fpr, false_positives, total_windows = (
                            false_positive_rate(
                                model,
                                x_phase,
                                device,
                            )
                        )

                        records.append(
                            {
                                "attack":
                                    attack,
                                "held_out_start":
                                    held_out_start,
                                "feature_set":
                                    feature_name,
                                "seed":
                                    seed,
                                "phase":
                                    phase_name,
                                "false_positive_rate":
                                    fpr,
                                "false_positives":
                                    false_positives,
                                "total_benign_windows":
                                    total_windows,
                                "epochs":
                                    epochs,
                                "best_val_loss":
                                    best_val_loss,
                            }
                        )

                    pd.DataFrame(
                        records
                    ).to_csv(
                        args.output_dir
                        / "benign_phase_all_runs.csv",
                        index=False,
                    )

    results = pd.DataFrame(records)

    summary = (
        results
        .groupby(
            [
                "feature_set",
                "phase",
            ],
            as_index=False,
        )
        .agg(
            mean_fpr=(
                "false_positive_rate",
                "mean",
            ),
            std_fpr=(
                "false_positive_rate",
                "std",
            ),
            median_fpr=(
                "false_positive_rate",
                "median",
            ),
            total_false_positives=(
                "false_positives",
                "sum",
            ),
            total_benign_windows=(
                "total_benign_windows",
                "sum",
            ),
        )
    )

    summary["pooled_fpr"] = (
        summary["total_false_positives"]
        / summary["total_benign_windows"]
    )

    summary.to_csv(
        args.output_dir
        / "benign_phase_summary.csv",
        index=False,
    )

    by_attack = (
        results
        .groupby(
            [
                "attack",
                "feature_set",
                "phase",
            ],
            as_index=False,
        )
        .agg(
            mean_fpr=(
                "false_positive_rate",
                "mean",
            ),
            std_fpr=(
                "false_positive_rate",
                "std",
            ),
            total_false_positives=(
                "false_positives",
                "sum",
            ),
            total_benign_windows=(
                "total_benign_windows",
                "sum",
            ),
        )
    )

    by_attack["pooled_fpr"] = (
        by_attack["total_false_positives"]
        / by_attack["total_benign_windows"]
    )

    by_attack.to_csv(
        args.output_dir
        / "benign_phase_by_attack.csv",
        index=False,
    )

    print("\nCompleted.")
    print(
        args.output_dir
        / "benign_phase_all_runs.csv"
    )
    print(
        args.output_dir
        / "benign_phase_summary.csv"
    )
    print(
        args.output_dir
        / "benign_phase_by_attack.csv"
    )


if __name__ == "__main__":
    main()
