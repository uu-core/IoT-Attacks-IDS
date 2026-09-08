from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ATTACKS = ("dis_flooding", "local_repair", "blackhole", "worst_parent")
START_TIMES = (300, 450, 600)
SEEDS = (42, 123, 999)
WINDOW = 10
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 10
FC_HIDDEN_SIZE = 10

FEATURE_SETS = {
    # RPL-4: topology position plus total received activity.
    "rpl_4": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
    ],
    "rpl_4_plus_txrx": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
        "delta_txraw_mean", "delta_txraw_std",
        "delta_rxraw_mean", "delta_rxraw_std",
    ],

    # RPL-6: RPL-4 plus DAO received activity.
    "rpl_6": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
        "delta_diar_mean", "delta_diar_std",
    ],
    "rpl_6_plus_txrx": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
        "delta_diar_mean", "delta_diar_std",
        "delta_txraw_mean", "delta_txraw_std",
        "delta_rxraw_mean", "delta_rxraw_std",
    ],

    # RPL-10: RPL-6 plus DIO received and sent activity.
    "rpl_10": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
        "delta_diar_mean", "delta_diar_std",
        "delta_dior_mean", "delta_dior_std",
        "delta_dios_mean", "delta_dios_std",
    ],
    "rpl_10_plus_txrx": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
        "delta_diar_mean", "delta_diar_std",
        "delta_dior_mean", "delta_dior_std",
        "delta_dios_mean", "delta_dios_std",
        "delta_txraw_mean", "delta_txraw_std",
        "delta_rxraw_mean", "delta_rxraw_std",
    ],

    # RPL-14: full original RPL feature family, adding DIS received and sent.
    "rpl_14": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
        "delta_diar_mean", "delta_diar_std",
        "delta_dior_mean", "delta_dior_std",
        "delta_dios_mean", "delta_dios_std",
        "delta_disr_mean", "delta_disr_std",
        "delta_diss_mean", "delta_diss_std",
    ],
    "rpl_14_plus_txrx": [
        "rank_mean", "rank_std",
        "delta_tots_mean", "delta_tots_std",
        "delta_diar_mean", "delta_diar_std",
        "delta_dior_mean", "delta_dior_std",
        "delta_dios_mean", "delta_dios_std",
        "delta_disr_mean", "delta_disr_std",
        "delta_diss_mean", "delta_diss_std",
        "delta_txraw_mean", "delta_txraw_std",
        "delta_rxraw_mean", "delta_rxraw_std",
    ],

    # Compact radio-only reference.
    "interval_txrx": [
        "delta_txraw_mean", "delta_txraw_std",
        "delta_rxraw_mean", "delta_rxraw_std",
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
    df = pd.read_csv(path)
    df["relative_time"] = df["minute_start"].astype(float) / max(float(df["minute_start"].max()), 1.0)
    return df


def attack_paths(root: Path, attack: str, start: int, runs: Iterable[int]) -> list[Path]:
    return [
        root / attack / f"start_{start}" / f"run_{run:02d}" / "features_temporal_validation.csv"
        for run in runs
    ]


def benign_paths(root: Path, runs: Iterable[int]) -> list[Path]:
    return [
        root / "benign" / "all_benign" / f"run_{run:02d}" / "features_temporal_validation.csv"
        for run in runs
    ]


def validate_paths(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))


def fit_minmax(paths: list[Path], columns: list[str]) -> tuple[pd.Series, pd.Series]:
    mins, maxs = [], []
    for path in paths:
        df = load_run(path)
        mins.append(df[columns].min(axis=0))
        maxs.append(df[columns].max(axis=0))
    global_min = pd.concat(mins, axis=1).min(axis=1)
    global_max = pd.concat(maxs, axis=1).max(axis=1)
    return global_min, global_max


def make_windows(
    df: pd.DataFrame,
    columns: list[str],
    global_min: pd.Series,
    global_max: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    values = df[columns].apply(pd.to_numeric, errors="coerce").copy()
    denom = (global_max - global_min).replace(0, 1)
    values = (values - global_min) / denom
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    labels = df["label"].astype(int).to_numpy()

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in range(0, len(df) - WINDOW + 1):
        end = start + WINDOW
        window_labels = labels[start:end]
        if np.all(window_labels == 0):
            y = 0
        elif np.all(window_labels == 1):
            y = 1
        else:
            continue  # discard windows crossing the attack boundary
        xs.append(values.iloc[start:end].to_numpy(dtype=np.float32))
        ys.append(y)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def build_dataset(paths: list[Path], columns: list[str], gmin: pd.Series, gmax: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    parts = [make_windows(load_run(p), columns, gmin, gmax) for p in paths]
    x = np.concatenate([a for a, _ in parts], axis=0)
    y = np.concatenate([b for _, b in parts], axis=0)
    return x, y


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator if shuffle else None)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, object]]:
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all, preds_all, probs_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * len(y)
            total += len(y)
            labels_all.extend(y.cpu().numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(labels_all)
    y_pred = np.asarray(preds_all)
    y_prob = np.asarray(probs_all)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred).tolist()),
    }
    return total_loss / total, metrics


def train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed)
    train_loader = make_loader(x_train, y_train, True, seed)
    val_loader = make_loader(x_val, y_val, False, seed)
    test_loader = make_loader(x_test, y_test, False, seed)

    model = TrueSequenceLSTM(x_train.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")
    best_state = None
    patience = 0
    epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_loss, _ = evaluate(model, val_loader, criterion, device)
        epochs = epoch
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("No best model state was saved")
    model.load_state_dict(best_state)
    test_loss, metrics = evaluate(model, test_loader, criterion, device)
    metrics.update({
        "epochs": epochs,
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
    })
    return metrics


def split_paths(root: Path, attack: str, held_out_start: int) -> tuple[list[Path], list[Path], list[Path]]:
    training_starts = [s for s in START_TIMES if s != held_out_start]

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for start in training_starts:
        train_paths.extend(attack_paths(root, attack, start, range(1, 9)))
        val_paths.extend(attack_paths(root, attack, start, range(9, 11)))

    train_paths.extend(benign_paths(root, range(1, 7)))
    val_paths.extend(benign_paths(root, range(7, 9)))
    test_paths = attack_paths(root, attack, held_out_start, range(1, 11)) + benign_paths(root, range(9, 11))

    validate_paths(train_paths + val_paths + test_paths)
    return train_paths, val_paths, test_paths


def make_plots(results: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        results
        .groupby(
            ["attack", "held_out_start", "feature_set"],
            as_index=False,
        )
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
        )
    )
    summary.to_csv(
        out_dir / "nested_rpl_summary.csv",
        index=False,
    )

    overall = (
        results
        .groupby("feature_set", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    overall.to_csv(
        out_dir / "nested_rpl_overall.csv",
        index=False,
    )

    pairs = [
        ("rpl_4", "rpl_4_plus_txrx", "rpl_4_gain"),
        ("rpl_6", "rpl_6_plus_txrx", "rpl_6_gain"),
        ("rpl_10", "rpl_10_plus_txrx", "rpl_10_gain"),
        ("rpl_14", "rpl_14_plus_txrx", "rpl_14_gain"),
    ]

    gain_frames: list[pd.DataFrame] = []

    for baseline, augmented, comparison in pairs:
        baseline_rows = results.loc[
            results["feature_set"] == baseline,
            ["attack", "held_out_start", "seed", "f1"],
        ].rename(columns={"f1": "baseline_f1"})

        augmented_rows = results.loc[
            results["feature_set"] == augmented,
            ["attack", "held_out_start", "seed", "f1"],
        ].rename(columns={"f1": "augmented_f1"})

        merged = baseline_rows.merge(
            augmented_rows,
            on=["attack", "held_out_start", "seed"],
            how="inner",
        )

        merged["comparison"] = comparison
        merged["baseline_feature_set"] = baseline
        merged["augmented_feature_set"] = augmented
        merged["delta_f1"] = (
            merged["augmented_f1"]
            - merged["baseline_f1"]
        )

        gain_frames.append(merged)

    gains = pd.concat(gain_frames, ignore_index=True)
    gains.to_csv(
        out_dir / "nested_rpl_paired_gains_all_runs.csv",
        index=False,
    )

    gain_by_start = (
        gains
        .groupby(
            ["comparison", "attack", "held_out_start"],
            as_index=False,
        )
        .agg(
            baseline_mean_f1=("baseline_f1", "mean"),
            augmented_mean_f1=("augmented_f1", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            std_delta_f1=("delta_f1", "std"),
            seeds_improved=("delta_f1", lambda s: int((s > 0).sum())),
            seeds_equal=("delta_f1", lambda s: int((s == 0).sum())),
            seeds_declined=("delta_f1", lambda s: int((s < 0).sum())),
        )
    )
    gain_by_start.to_csv(
        out_dir / "nested_rpl_paired_gains_by_start.csv",
        index=False,
    )

    gain_by_attack = (
        gains
        .groupby(
            ["comparison", "attack"],
            as_index=False,
        )
        .agg(
            baseline_mean_f1=("baseline_f1", "mean"),
            augmented_mean_f1=("augmented_f1", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            std_delta_f1=("delta_f1", "std"),
            evaluations_improved=("delta_f1", lambda s: int((s > 0).sum())),
            evaluations_equal=("delta_f1", lambda s: int((s == 0).sum())),
            evaluations_declined=("delta_f1", lambda s: int((s < 0).sum())),
        )
    )
    gain_by_attack.to_csv(
        out_dir / "nested_rpl_paired_gains_by_attack.csv",
        index=False,
    )

    gain_overall = (
        gains
        .groupby("comparison", as_index=False)
        .agg(
            baseline_mean_f1=("baseline_f1", "mean"),
            augmented_mean_f1=("augmented_f1", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            std_delta_f1=("delta_f1", "std"),
            evaluations_improved=("delta_f1", lambda s: int((s > 0).sum())),
            evaluations_equal=("delta_f1", lambda s: int((s == 0).sum())),
            evaluations_declined=("delta_f1", lambda s: int((s < 0).sum())),
        )
    )
    gain_overall.to_csv(
        out_dir / "nested_rpl_paired_gains_overall.csv",
        index=False,
    )

    feature_order = [
        "rpl_4",
        "rpl_4_plus_txrx",
        "rpl_6",
        "rpl_6_plus_txrx",
        "rpl_10",
        "rpl_10_plus_txrx",
        "rpl_14",
        "rpl_14_plus_txrx",
        "interval_txrx",
    ]

    values = [
        results.loc[
            results["feature_set"] == feature_name,
            "f1",
        ].to_numpy()
        for feature_name in feature_order
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.boxplot(
        values,
        tick_labels=feature_order,
        showmeans=True,
    )
    ax.set_ylabel("Cross-start-time F1-score")
    ax.set_xlabel("Feature set")
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        out_dir / "nested_rpl_f1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "ids-WPLR/applications/example-attacks/validation_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "ids-WPLR/temporal_validation/nested_rpl_results",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")

    records: list[dict[str, object]] = []
    total_jobs = len(ATTACKS) * len(START_TIMES) * len(FEATURE_SETS) * len(SEEDS)
    job = 0

    for attack in ATTACKS:
        for held_out_start in START_TIMES:
            train_paths, val_paths, test_paths = split_paths(args.data_root, attack, held_out_start)
            for feature_name, columns in FEATURE_SETS.items():
                gmin, gmax = fit_minmax(train_paths, columns)
                x_train, y_train = build_dataset(train_paths, columns, gmin, gmax)
                x_val, y_val = build_dataset(val_paths, columns, gmin, gmax)
                x_test, y_test = build_dataset(test_paths, columns, gmin, gmax)

                for seed in SEEDS:
                    job += 1
                    print(f"[{job}/{total_jobs}] {attack} | test_start={held_out_start} | {feature_name} | seed={seed}")
                    metrics = train_one(x_train, y_train, x_val, y_val, x_test, y_test, seed, device)
                    records.append({
                        "attack": attack,
                        "held_out_start": held_out_start,
                        "feature_set": feature_name,
                        "seed": seed,
                        **metrics,
                    })
                    pd.DataFrame(records).to_csv(args.output_dir / "nested_rpl_all_runs.csv", index=False)

    results = pd.DataFrame(records)
    make_plots(results, args.output_dir)
    print("\nCompleted.")
    print(args.output_dir / "nested_rpl_all_runs.csv")
    print(args.output_dir / "nested_rpl_summary.csv")
    print(args.output_dir / "nested_rpl_overall.csv")
    print(args.output_dir / "nested_rpl_f1.png")


if __name__ == "__main__":
    main()
