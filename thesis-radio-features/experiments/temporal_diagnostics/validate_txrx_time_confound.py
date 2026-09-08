from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

import models


# ============================================================
# Paths and experimental settings
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_ROOT = PROJECT_ROOT / "attack_data"

MAPPING_CANDIDATES = [
    PROJECT_ROOT / "domain_details.xlsx",
    SRC_DIR / "domain_details.xlsx",
]

OUTPUT_DIR = SRC_DIR / "results" / "temporal_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_OUTPUT = OUTPUT_DIR / "temporal_validation_all_runs.csv"
DOMAIN_OUTPUT = OUTPUT_DIR / "temporal_validation_by_domain.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "temporal_validation_summary.csv"
ATTACK_OUTPUT = OUTPUT_DIR / "temporal_validation_by_attack.csv"
PLOT_OUTPUT = OUTPUT_DIR / "temporal_validation_f1.png"

FILE_SPLIT_SEEDS = [42, 123, 999]

WINDOW_SIZE = 10
BATCH_SIZE = 128
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.001
HIDDEN_SIZE = 10
FC_HIDDEN_SIZE = 10
OUTPUT_SIZE = 2
NUM_LAYERS = 1

TXRX_COLUMNS = [
    "tx",
    "tx.1",
    "rx",
    "rx.1",
]

CONDITIONS = [
    "cumulative_txrx",
    "differenced_txrx",
    "time_only",
]

ALLOWED_ATTACKS = {
    "blackhole",
    "dis_flooding",
    "local_repair",
    "worst_parent",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Metadata and domain discovery
# ============================================================

def normalize_attack(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")

    aliases = {
        "black_hole": "blackhole",
        "blackhole": "blackhole",
        "dis_flooding": "dis_flooding",
        "disflooding": "dis_flooding",
        "local_repair": "local_repair",
        "localrepair": "local_repair",
        "worst_parent": "worst_parent",
        "worstparent": "worst_parent",
    }

    return aliases.get(text, text)


def normalize_domain(value: object) -> str:
    text = str(value).strip().lower()

    match = re.search(
        r"domain[\s_-]*0*(\d+)",
        text,
    )

    if match:
        return f"domain{int(match.group(1)):02d}"

    return text


def normalize_variant(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("–", "-")

    aliases = {
        "base": "base",
        "oo": "oo",
        "on-off": "oo",
        "on_off": "oo",
        "onoff": "oo",
        "gc": "gc",
        "gradual change": "gc",
        "gradual-change": "gc",
        "gradual_change": "gc",
    }

    return aliases.get(text, text)


def find_mapping_file() -> Path:
    for candidate in MAPPING_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "domain_details.xlsx was not found. Checked:\n"
        + "\n".join(str(path) for path in MAPPING_CANDIDATES)
    )


def extract_file_index(path: Path) -> int:
    name = path.name

    patterns = [
        r"\((\d+)\)\.csv$",
        r"_(\d+)_60_sec\.csv$",
        r"(\d+)\.csv$",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)

        if match:
            return int(match.group(1))

    return 10**9


def discover_domains() -> list[dict[str, object]]:
    mapping_file = find_mapping_file()

    metadata = pd.read_excel(
        mapping_file,
        dtype=str,
    ).fillna("")

    metadata.columns = (
        metadata.columns
        .astype(str)
        .str.strip()
    )

    required_columns = {
        "Domain Name",
        "Attack Type",
        "Node",
        "Version",
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing columns in domain_details.xlsx: "
            f"{sorted(missing_columns)}"
        )

    metadata_map: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}

    for _, row in metadata.iterrows():
        attack = normalize_attack(
            row["Attack Type"]
        )

        if attack not in ALLOWED_ATTACKS:
            continue

        raw_domain = normalize_domain(
            row["Domain Name"]
        )

        node = int(float(row["Node"]))
        variant = normalize_variant(
            row["Version"]
        )

        metadata_map[(attack, raw_domain)] = {
            "attack": attack,
            "raw_domain": raw_domain,
            "node": node,
            "variant": variant,
            "canonical_domain":
                f"{attack}_{node}_{variant}",
        }

    domains: list[dict[str, object]] = []

    for attack_dir in sorted(
        DATA_ROOT.iterdir()
    ):
        if not attack_dir.is_dir():
            continue

        attack = normalize_attack(
            attack_dir.name
        )

        if attack not in ALLOWED_ATTACKS:
            continue

        for domain_dir in sorted(
            attack_dir.iterdir()
        ):
            if not domain_dir.is_dir():
                continue

            raw_domain = normalize_domain(
                domain_dir.name
            )

            metadata_entry = metadata_map.get(
                (attack, raw_domain)
            )

            if metadata_entry is None:
                print(
                    "[SKIP: no metadata] "
                    f"{attack_dir.name}/"
                    f"{domain_dir.name}"
                )
                continue

            csv_files = sorted(
                domain_dir.glob("*.csv"),
                key=extract_file_index,
            )[:20]

            if len(csv_files) < 20:
                print(
                    f"[WARNING] "
                    f"{metadata_entry['canonical_domain']} "
                    f"contains {len(csv_files)} CSV files."
                )

            domain_record = dict(
                metadata_entry
            )

            domain_record["csv_files"] = (
                csv_files
            )

            domains.append(
                domain_record
            )

    domains.sort(
        key=lambda item:
        str(item["canonical_domain"])
    )

    print(
        f"Discovered {len(domains)} domains."
    )

    if len(domains) != 48:
        print(
            "WARNING: 48 domains were expected."
        )

    return domains


# ============================================================
# Feature construction
# ============================================================

def load_run_dataframe(
    path: Path,
    condition: str,
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        index_col=0,
    )

    missing = (
        set(TXRX_COLUMNS + ["label"])
        - set(dataframe.columns)
    )

    if missing:
        raise KeyError(
            f"Missing columns in {path}:\n"
            f"{sorted(missing)}"
        )

    labels = (
        pd.to_numeric(
            dataframe["label"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    if condition == "cumulative_txrx":
        features = dataframe[
            TXRX_COLUMNS
        ].copy()

    elif condition == "differenced_txrx":
        # This computes first differences of the four already
        # aggregated cumulative columns. It is a diagnostic
        # approximation because the original node-level values
        # are not available in these CSV files.
        features = (
            dataframe[TXRX_COLUMNS]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .diff()
            .fillna(0.0)
        )

    elif condition == "time_only":
        features = pd.DataFrame(
            {
                "elapsed_index":
                    np.arange(
                        len(dataframe),
                        dtype=float,
                    )
            },
            index=dataframe.index,
        )

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    features = features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    output = features.copy()
    output["label"] = labels.to_numpy()

    return output


def fit_training_minmax(
    training_dataframes: list[pd.DataFrame],
) -> tuple[pd.Series, pd.Series]:
    feature_columns = [
        column
        for column in training_dataframes[0].columns
        if column != "label"
    ]

    training_mins = [
        dataframe[feature_columns].min(
            axis=0
        )
        for dataframe in training_dataframes
    ]

    training_maxs = [
        dataframe[feature_columns].max(
            axis=0
        )
        for dataframe in training_dataframes
    ]

    global_min = pd.concat(
        training_mins,
        axis=1,
    ).min(axis=1)

    global_max = pd.concat(
        training_maxs,
        axis=1,
    ).max(axis=1)

    return global_min, global_max


def normalize_dataframe(
    dataframe: pd.DataFrame,
    global_min: pd.Series,
    global_max: pd.Series,
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in dataframe.columns
        if column != "label"
    ]

    denominator = (
        global_max - global_min
    ).replace(0, 1)

    normalized = dataframe.copy()

    normalized[feature_columns] = (
        normalized[feature_columns]
        - global_min
    ) / denominator

    normalized[feature_columns] = (
        normalized[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return normalized


# ============================================================
# Exact sequence construction used by utils.py
# ============================================================

def make_sequences(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    feature_dataframe = dataframe.drop(
        columns=["label"]
    )

    labels = (
        dataframe["label"]
        .astype(int)
        .to_numpy()
    )

    attack_indices = np.where(
        labels == 1
    )[0]

    if len(attack_indices) == 0:
        attack_boundary = (
            len(labels) + WINDOW_SIZE
        )
    else:
        attack_boundary = max(
            0,
            int(attack_indices[0])
            - WINDOW_SIZE,
        )

    sequences = []

    # Reproduces:
    # range(len(df_feat) - sequence_length)
    for start in range(
        len(feature_dataframe)
        - WINDOW_SIZE
    ):
        window = (
            feature_dataframe
            .iloc[
                start:
                start + WINDOW_SIZE
            ]
            .to_numpy()
            .flatten()
        )

        sequences.append(window)

    if not sequences:
        feature_dimension = (
            feature_dataframe.shape[1]
            * WINDOW_SIZE
        )

        return (
            np.empty(
                (0, feature_dimension),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.int64,
            ),
        )

    sequence_array = np.asarray(
        sequences,
        dtype=np.float32,
    )

    number_of_benign = min(
        attack_boundary,
        len(sequence_array),
    )

    sequence_labels = np.concatenate(
        [
            np.zeros(
                number_of_benign,
                dtype=np.int64,
            ),
            np.ones(
                len(sequence_array)
                - number_of_benign,
                dtype=np.int64,
            ),
        ]
    )

    return (
        sequence_array,
        sequence_labels,
    )


def construct_datasets(
    training_paths: list[Path],
    test_paths: list[Path],
    condition: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    training_dataframes = [
        load_run_dataframe(
            path,
            condition,
        )
        for path in training_paths
    ]

    test_dataframes = [
        load_run_dataframe(
            path,
            condition,
        )
        for path in test_paths
    ]

    global_min, global_max = (
        fit_training_minmax(
            training_dataframes
        )
    )

    normalized_training = [
        normalize_dataframe(
            dataframe,
            global_min,
            global_max,
        )
        for dataframe
        in training_dataframes
    ]

    normalized_test = [
        normalize_dataframe(
            dataframe,
            global_min,
            global_max,
        )
        for dataframe
        in test_dataframes
    ]

    training_parts = [
        make_sequences(dataframe)
        for dataframe
        in normalized_training
    ]

    test_parts = [
        make_sequences(dataframe)
        for dataframe
        in normalized_test
    ]

    X_train = np.concatenate(
        [
            features
            for features, _
            in training_parts
            if len(features) > 0
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            labels
            for _, labels
            in training_parts
            if len(labels) > 0
        ],
        axis=0,
    )

    X_test = np.concatenate(
        [
            features
            for features, _
            in test_parts
            if len(features) > 0
        ],
        axis=0,
    )

    y_test = np.concatenate(
        [
            labels
            for _, labels
            in test_parts
            if len(labels) > 0
        ],
        axis=0,
    )

    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_test = np.nan_to_num(
        X_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Exact input representation used by the current model:
    # B x 1 x (WINDOW_SIZE * feature_count)
    X_train = X_train[:, None, :]
    X_test = X_test[:, None, :]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


# ============================================================
# Training and evaluation
# ============================================================

def create_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.long,
        ),
    )

    test_dataset = TensorDataset(
        torch.tensor(
            X_test,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_test,
            dtype=torch.long,
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(features)
        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * len(labels)
        )

        total_count += len(labels)

    return total_loss / total_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, dict[str, object]]:
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _ = model(features)

            loss = criterion(
                logits,
                labels,
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
                .cpu()
                .numpy()
            )

            predictions = (
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            label_array = (
                labels.cpu().numpy()
            )

            total_loss += (
                loss.item() * len(labels)
            )

            total_count += len(labels)

            all_predictions.extend(
                predictions.tolist()
            )

            all_labels.extend(
                label_array.tolist()
            )

            all_probabilities.extend(
                probabilities.tolist()
            )

    predictions_array = np.asarray(
        all_predictions
    )

    labels_array = np.asarray(
        all_labels
    )

    probabilities_array = np.asarray(
        all_probabilities
    )

    accuracy = float(
        np.mean(
            predictions_array
            == labels_array
        )
    )

    f1 = f1_score(
        labels_array,
        predictions_array,
        zero_division=0,
    )

    precision = precision_score(
        labels_array,
        predictions_array,
        zero_division=0,
    )

    recall = recall_score(
        labels_array,
        predictions_array,
        zero_division=0,
    )

    if len(
        np.unique(labels_array)
    ) > 1:
        auc = roc_auc_score(
            labels_array,
            probabilities_array,
        )
    else:
        auc = 0.0

    matrix = confusion_matrix(
        labels_array,
        predictions_array,
    ).tolist()

    metrics = {
        "accuracy": accuracy,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "confusion_matrix":
            json.dumps(matrix),
    }

    return (
        total_loss / total_count,
        metrics,
    )


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    set_all_seeds(seed)

    train_loader, test_loader = (
        create_loaders(
            X_train,
            y_train,
            X_test,
            y_test,
            seed,
        )
    )

    input_dimension = X_train.shape[2]

    model = models.LSTMClassifier(
        input_dim=input_dimension,
        hidden_dim=HIDDEN_SIZE,
        output_dim=OUTPUT_SIZE,
        num_layers=NUM_LAYERS,
        fc_hidden_dim=FC_HIDDEN_SIZE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_test_loss = float("inf")
    best_state = None
    patience_counter = 0
    epochs_completed = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        test_loss, _ = evaluate(
            model,
            test_loader,
            criterion,
        )

        epochs_completed = epoch

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(
            "No model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    final_test_loss, metrics = evaluate(
        model,
        test_loader,
        criterion,
    )

    metrics["test_loss"] = (
        float(final_test_loss)
    )

    metrics["epochs"] = (
        epochs_completed
    )

    metrics["input_dimension"] = (
        input_dimension
    )

    return metrics


# ============================================================
# Experiment
# ============================================================

def create_file_split(
    csv_files: list[Path],
    seed: int,
) -> tuple[list[Path], list[Path]]:
    selected_files = list(
        csv_files[:20]
    )

    random_generator = random.Random(
        seed
    )

    random_generator.shuffle(
        selected_files
    )

    return (
        selected_files[:16],
        selected_files[16:20],
    )


def run_validation(
    domains: list[dict[str, object]],
) -> pd.DataFrame:
    records = []

    total_jobs = (
        len(domains)
        * len(FILE_SPLIT_SEEDS)
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

        for seed in FILE_SPLIT_SEEDS:
            training_paths, test_paths = (
                create_file_split(
                    csv_files,
                    seed,
                )
            )

            for condition in CONDITIONS:
                job_number += 1

                print(
                    f"[{job_number}/{total_jobs}] "
                    f"{domain_name} | "
                    f"seed={seed} | "
                    f"{condition}"
                )

                (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                ) = construct_datasets(
                    training_paths,
                    test_paths,
                    condition,
                )

                metrics = train_and_evaluate(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    seed,
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
                            seed,
                        "train_sequences":
                            len(y_train),
                        "test_sequences":
                            len(y_test),
                        **metrics,
                    }
                )

                pd.DataFrame(
                    records
                ).to_csv(
                    DETAIL_OUTPUT,
                    index=False,
                )

    return pd.DataFrame(records)


# ============================================================
# Aggregation and plotting
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
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_auc=("auc", "mean"),
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
            mean_f1=("mean_f1", "mean"),
            median_f1=(
                "mean_f1",
                "median",
            ),
            std_f1=("mean_f1", "std"),
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
            mean_f1=("mean_f1", "mean"),
            std_f1=("mean_f1", "std"),
            mean_auc=(
                "mean_auc",
                "mean",
            ),
        )
    )

    by_attack.to_csv(
        ATTACK_OUTPUT,
        index=False,
    )

    plot_order = CONDITIONS

    plot_data = [
        by_domain.loc[
            by_domain["condition"]
            == condition,
            "mean_f1",
        ].to_numpy()
        for condition in plot_order
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.boxplot(
        plot_data,
        tick_labels=[
            "Cumulative\nTX/RX",
            "Differenced\nTX/RX",
            "Time only",
        ],
        showmeans=True,
    )

    jitter_generator = (
        np.random.default_rng(42)
    )

    for position, values in enumerate(
        plot_data,
        start=1,
    ):
        jitter = (
            jitter_generator.normal(
                0,
                0.045,
                size=len(values),
            )
        )

        axis.scatter(
            np.full(
                len(values),
                position,
            ) + jitter,
            values,
            alpha=0.45,
            s=18,
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

    plt.close(figure)

    print("\nSummary:")
    print(
        summary.to_string(index=False)
    )

    print(
        f"\nDetailed results:\n"
        f"{DETAIL_OUTPUT}"
    )

    print(
        f"\nPer-domain results:\n"
        f"{DOMAIN_OUTPUT}"
    )

    print(
        f"\nSummary results:\n"
        f"{SUMMARY_OUTPUT}"
    )

    print(
        f"\nAttack-wise results:\n"
        f"{ATTACK_OUTPUT}"
    )

    print(
        f"\nComparison plot:\n"
        f"{PLOT_OUTPUT}"
    )


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")

    domains = discover_domains()

    detailed_results = run_validation(
        domains
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