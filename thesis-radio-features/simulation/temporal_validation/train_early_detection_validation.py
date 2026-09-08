from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ATTACKS = (
    "dis_flooding",
    "local_repair",
    "blackhole",
    "worst_parent",
)

START_TIMES = (300, 450, 600)
SEEDS = (42, 123, 999)
HORIZONS = (1, 3, 5, 10, 30)

WINDOW = 10
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 10
FC_HIDDEN_SIZE = 10

FEATURE_SETS = {
    "interval_txrx": [
        "delta_txraw_mean",
        "delta_txraw_std",
        "delta_rxraw_mean",
        "delta_rxraw_std",
    ],
    "rpl_10": [
        "rank_mean",
        "rank_std",
        "delta_tots_mean",
        "delta_tots_std",
        "delta_diar_mean",
        "delta_diar_std",
        "delta_dior_mean",
        "delta_dior_std",
        "delta_dios_mean",
        "delta_dios_std",
    ],
    "rpl_10_plus_txrx": [
        "rank_mean",
        "rank_std",
        "delta_tots_mean",
        "delta_tots_std",
        "delta_diar_mean",
        "delta_diar_std",
        "delta_dior_mean",
        "delta_dior_std",
        "delta_dios_mean",
        "delta_dios_std",
        "delta_txraw_mean",
        "delta_txraw_std",
        "delta_rxraw_mean",
        "delta_rxraw_std",
    ],
    "rpl_14": [
        "rank_mean",
        "rank_std",
        "delta_tots_mean",
        "delta_tots_std",
        "delta_diar_mean",
        "delta_diar_std",
        "delta_dior_mean",
        "delta_dior_std",
        "delta_dios_mean",
        "delta_dios_std",
        "delta_disr_mean",
        "delta_disr_std",
        "delta_diss_mean",
        "delta_diss_std",
    ],
    "rpl_14_plus_txrx": [
        "rank_mean",
        "rank_std",
        "delta_tots_mean",
        "delta_tots_std",
        "delta_diar_mean",
        "delta_diar_std",
        "delta_dior_mean",
        "delta_dior_std",
        "delta_dios_mean",
        "delta_dios_std",
        "delta_disr_mean",
        "delta_disr_std",
        "delta_diss_mean",
        "delta_diss_std",
        "delta_txraw_mean",
        "delta_txraw_std",
        "delta_rxraw_mean",
        "delta_rxraw_std",
    ],
}


class TrueSequenceLSTM(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(HIDDEN_SIZE, FC_HIDDEN_SIZE)
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        feat = torch.relu(self.fc1(out[:, -1, :]))
        return self.fc2(feat)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_run(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def attack_paths(
    root: Path,
    attack: str,
    start: int,
    runs: Iterable[int],
) -> list[Path]:
    return [
        root
        / attack
        / f"start_{start}"
        / f"run_{run:02d}"
        / "features_temporal_validation.csv"
        for run in runs
    ]


def benign_paths(
    root: Path,
    runs: Iterable[int],
) -> list[Path]:
    return [
        root
        / "benign"
        / "all_benign"
        / f"run_{run:02d}"
        / "features_temporal_validation.csv"
        for run in runs
    ]


def validate_paths(paths: Iterable[Path]) -> None:
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files:\n"
            + "\n".join(missing)
        )


def split_paths(
    root: Path,
    attack: str,
    held_out_start: int,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    training_starts = [
        start
        for start in START_TIMES
        if start != held_out_start
    ]

    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for start in training_starts:
        train_paths.extend(
            attack_paths(
                root,
                attack,
                start,
                range(1, 9),
            )
        )
        val_paths.extend(
            attack_paths(
                root,
                attack,
                start,
                range(9, 11),
            )
        )

    train_paths.extend(
        benign_paths(root, range(1, 7))
    )
    val_paths.extend(
        benign_paths(root, range(7, 9))
    )

    test_attack_paths = attack_paths(
        root,
        attack,
        held_out_start,
        range(1, 11),
    )

    test_benign_paths = benign_paths(
        root,
        range(9, 11),
    )

    validate_paths(
        train_paths
        + val_paths
        + test_attack_paths
        + test_benign_paths
    )

    return (
        train_paths,
        val_paths,
        test_attack_paths,
        test_benign_paths,
    )


def fit_minmax(
    paths: list[Path],
    columns: list[str],
) -> tuple[pd.Series, pd.Series]:
    mins: list[pd.Series] = []
    maxs: list[pd.Series] = []

    for path in paths:
        df = load_run(path)
        numeric = df[columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        mins.append(numeric.min(axis=0))
        maxs.append(numeric.max(axis=0))

    global_min = pd.concat(
        mins,
        axis=1,
    ).min(axis=1)

    global_max = pd.concat(
        maxs,
        axis=1,
    ).max(axis=1)

    return global_min, global_max


def normalise(
    df: pd.DataFrame,
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
) -> pd.DataFrame:
    values = df[columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).copy()

    denominator = (
        global_max - global_min
    ).replace(0, 1)

    values = (
        values - global_min
    ) / denominator

    return (
        values
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )


def make_causal_windows(
    df: pd.DataFrame,
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each window is labelled by its LAST observation.

    This deliberately keeps boundary-crossing windows:
    for example, a window ending one minute after attack onset may contain
    nine benign observations and one attack observation. That is necessary
    for measuring real early detection with a 10-minute causal window.
    """
    values = normalise(
        df,
        columns,
        global_min,
        global_max,
    )

    labels = pd.to_numeric(
        df["label"],
        errors="coerce",
    ).fillna(0).astype(int).to_numpy()

    minutes = pd.to_numeric(
        df["minute_start"],
        errors="coerce",
    ).to_numpy(dtype=float)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    end_minutes: list[float] = []

    for start_index in range(
        0,
        len(df) - WINDOW + 1,
    ):
        end_index = start_index + WINDOW
        last_index = end_index - 1

        xs.append(
            values.iloc[
                start_index:end_index
            ].to_numpy(dtype=np.float32)
        )

        ys.append(
            int(labels[last_index])
        )

        end_minutes.append(
            float(minutes[last_index])
        )

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(end_minutes, dtype=float),
    )


def build_dataset(
    paths: list[Path],
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    parts = [
        make_causal_windows(
            load_run(path),
            columns,
            global_min,
            global_max,
        )
        for path in paths
    ]

    x = np.concatenate(
        [part[0] for part in parts],
        axis=0,
    )

    y = np.concatenate(
        [part[1] for part in parts],
        axis=0,
    )

    return x, y


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(
            x,
            dtype=torch.float32,
        ),
        torch.tensor(
            y,
            dtype=torch.long,
        ),
    )

    generator = (
        torch.Generator().manual_seed(seed)
        if shuffle
        else None
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(
                logits,
                y_batch,
            )

            total_loss += (
                loss.item()
                * len(y_batch)
            )
            total += len(y_batch)

    return total_loss / max(total, 1)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, int, float]:
    set_seed(seed)

    train_loader = make_loader(
        x_train,
        y_train,
        True,
        seed,
    )

    val_loader = make_loader(
        x_val,
        y_val,
        False,
        seed,
    )

    model = TrueSequenceLSTM(
        x_train.shape[2]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    completed_epochs = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

        current_val_loss = validation_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        completed_epochs = epoch

        if (
            current_val_loss
            < best_val_loss - 1e-8
        ):
            best_val_loss = current_val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0
        else:
            patience_counter += 1

            if (
                patience_counter
                >= PATIENCE
            ):
                break

    if best_state is None:
        raise RuntimeError(
            "No best model state was saved"
        )

    model.load_state_dict(best_state)

    return (
        model,
        completed_epochs,
        float(best_val_loss),
    )


def predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    predicted_classes: list[int] = []
    attack_probabilities: list[float] = []

    with torch.no_grad():
        for start_index in range(
            0,
            len(x),
            BATCH_SIZE,
        ):
            batch = torch.tensor(
                x[
                    start_index:
                    start_index + BATCH_SIZE
                ],
                dtype=torch.float32,
                device=device,
            )

            logits = model(batch)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            predictions = logits.argmax(
                dim=1
            )

            predicted_classes.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            attack_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            predicted_classes,
            dtype=int,
        ),
        np.asarray(
            attack_probabilities,
            dtype=float,
        ),
    )


def evaluate_attack_run(
    model: nn.Module,
    path: Path,
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
    attack_start: int,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    df = load_run(path)

    x, _, end_minutes = make_causal_windows(
        df,
        columns,
        global_min,
        global_max,
    )

    predictions, probabilities = predict(
        model,
        x,
        device,
    )

    pre_attack_mask = (
        end_minutes < attack_start
    )

    post_attack_mask = (
        end_minutes >= attack_start
    )

    pre_attack_predictions = predictions[
        pre_attack_mask
    ]

    pre_attack_fpr = (
        float(
            np.mean(
                pre_attack_predictions == 1
            )
        )
        if len(pre_attack_predictions) > 0
        else float("nan")
    )

    post_minutes = end_minutes[
        post_attack_mask
    ]

    post_predictions = predictions[
        post_attack_mask
    ]

    post_probabilities = probabilities[
        post_attack_mask
    ]

    positive_indices = np.flatnonzero(
        post_predictions == 1
    )

    if len(positive_indices) > 0:
        first_index = int(
            positive_indices[0]
        )

        first_detection_minute = float(
            post_minutes[first_index]
        )

        first_detection_delay = (
            first_detection_minute
            - float(attack_start)
        )
    else:
        first_detection_minute = float("nan")
        first_detection_delay = float("nan")

    summary = {
        "run_path": str(path),
        "pre_attack_fpr": pre_attack_fpr,
        "first_detection_minute":
            first_detection_minute,
        "first_detection_delay":
            first_detection_delay,
        "detected_within_30":
            bool(
                np.isfinite(
                    first_detection_delay
                )
                and first_detection_delay <= 30
            ),
    }

    horizon_records: list[dict[str, object]] = []

    for horizon in HORIZONS:
        horizon_mask = (
            (post_minutes >= attack_start)
            & (
                post_minutes
                <= attack_start + horizon
            )
        )

        horizon_predictions = post_predictions[
            horizon_mask
        ]

        horizon_probabilities = post_probabilities[
            horizon_mask
        ]

        detected = bool(
            np.any(
                horizon_predictions == 1
            )
        )

        minute_recall = (
            float(
                np.mean(
                    horizon_predictions == 1
                )
            )
            if len(horizon_predictions) > 0
            else float("nan")
        )

        mean_attack_probability = (
            float(
                np.mean(
                    horizon_probabilities
                )
            )
            if len(horizon_probabilities) > 0
            else float("nan")
        )

        horizon_records.append(
            {
                "horizon_minutes": horizon,
                "detected_by_horizon":
                    detected,
                "early_minute_recall":
                    minute_recall,
                "mean_attack_probability":
                    mean_attack_probability,
                "evaluated_windows":
                    int(
                        len(
                            horizon_predictions
                        )
                    ),
            }
        )

    return summary, horizon_records


def benign_false_positive_rate(
    model: nn.Module,
    paths: list[Path],
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
    device: torch.device,
) -> float:
    predictions_all: list[np.ndarray] = []

    for path in paths:
        x, _, _ = make_causal_windows(
            load_run(path),
            columns,
            global_min,
            global_max,
        )

        predictions, _ = predict(
            model,
            x,
            device,
        )

        predictions_all.append(
            predictions
        )

    predictions = np.concatenate(
        predictions_all,
        axis=0,
    )

    return float(
        np.mean(
            predictions == 1
        )
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
            / "early_detection_results"
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

    run_records: list[dict[str, object]] = []
    horizon_records: list[dict[str, object]] = []

    total_jobs = (
        len(ATTACKS)
        * len(START_TIMES)
        * len(FEATURE_SETS)
        * len(SEEDS)
    )

    job_number = 0

    for attack in ATTACKS:
        for held_out_start in START_TIMES:
            (
                train_paths,
                val_paths,
                test_attack_paths,
                test_benign_paths,
            ) = split_paths(
                args.data_root,
                attack,
                held_out_start,
            )

            for (
                feature_name,
                columns,
            ) in FEATURE_SETS.items():
                global_min, global_max = fit_minmax(
                    train_paths,
                    columns,
                )

                x_train, y_train = build_dataset(
                    train_paths,
                    columns,
                    global_min,
                    global_max,
                )

                x_val, y_val = build_dataset(
                    val_paths,
                    columns,
                    global_min,
                    global_max,
                )

                for seed in SEEDS:
                    job_number += 1

                    print(
                        f"[{job_number}/{total_jobs}] "
                        f"{attack} | "
                        f"test_start={held_out_start} | "
                        f"{feature_name} | "
                        f"seed={seed}"
                    )

                    (
                        model,
                        epochs,
                        best_val_loss,
                    ) = train_model(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        seed,
                        device,
                    )

                    benign_fpr = (
                        benign_false_positive_rate(
                            model,
                            test_benign_paths,
                            columns,
                            global_min,
                            global_max,
                            device,
                        )
                    )

                    for run_index, path in enumerate(
                        test_attack_paths,
                        start=1,
                    ):
                        (
                            run_summary,
                            run_horizons,
                        ) = evaluate_attack_run(
                            model,
                            path,
                            columns,
                            global_min,
                            global_max,
                            held_out_start,
                            device,
                        )

                        common = {
                            "attack": attack,
                            "held_out_start":
                                held_out_start,
                            "feature_set":
                                feature_name,
                            "seed": seed,
                            "test_run":
                                run_index,
                            "benign_test_fpr":
                                benign_fpr,
                            "epochs": epochs,
                            "best_val_loss":
                                best_val_loss,
                        }

                        run_records.append(
                            {
                                **common,
                                **run_summary,
                            }
                        )

                        for horizon_row in run_horizons:
                            horizon_records.append(
                                {
                                    **common,
                                    **horizon_row,
                                }
                            )

                    pd.DataFrame(
                        run_records
                    ).to_csv(
                        args.output_dir
                        / "early_detection_all_runs.csv",
                        index=False,
                    )

                    pd.DataFrame(
                        horizon_records
                    ).to_csv(
                        args.output_dir
                        / "early_detection_all_horizons.csv",
                        index=False,
                    )

    runs_df = pd.DataFrame(
        run_records
    )

    horizons_df = pd.DataFrame(
        horizon_records
    )

    delay_summary = (
        runs_df
        .groupby(
            [
                "attack",
                "feature_set",
            ],
            as_index=False,
        )
        .agg(
            median_detection_delay=(
                "first_detection_delay",
                "median",
            ),
            mean_detection_delay=(
                "first_detection_delay",
                "mean",
            ),
            detected_within_30_rate=(
                "detected_within_30",
                "mean",
            ),
            mean_pre_attack_fpr=(
                "pre_attack_fpr",
                "mean",
            ),
            mean_benign_test_fpr=(
                "benign_test_fpr",
                "mean",
            ),
        )
    )

    delay_summary.to_csv(
        args.output_dir
        / "early_detection_delay_summary.csv",
        index=False,
    )

    horizon_summary = (
        horizons_df
        .groupby(
            [
                "attack",
                "feature_set",
                "horizon_minutes",
            ],
            as_index=False,
        )
        .agg(
            detection_rate=(
                "detected_by_horizon",
                "mean",
            ),
            mean_early_minute_recall=(
                "early_minute_recall",
                "mean",
            ),
            mean_attack_probability=(
                "mean_attack_probability",
                "mean",
            ),
            mean_benign_test_fpr=(
                "benign_test_fpr",
                "mean",
            ),
        )
    )

    horizon_summary.to_csv(
        args.output_dir
        / "early_detection_horizon_summary.csv",
        index=False,
    )

    print("\nCompleted.")
    print(
        args.output_dir
        / "early_detection_all_runs.csv"
    )
    print(
        args.output_dir
        / "early_detection_all_horizons.csv"
    )
    print(
        args.output_dir
        / "early_detection_delay_summary.csv"
    )
    print(
        args.output_dir
        / "early_detection_horizon_summary.csv"
    )


if __name__ == "__main__":
    main()
