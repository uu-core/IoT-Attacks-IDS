import os
import random
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import logging
import argparse
import sys
import re


def safe_minmax_normalize(df, global_min, global_max, label_col="label"):
    feat_cols = [c for c in df.columns if c != label_col]
    denom = (global_max - global_min).replace(0, 1)  # avoid div/0
    out = df.copy()
    out[feat_cols] = (out[feat_cols] - global_min) / denom
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def seq_maker(df, sequence_length=10, label_col="label"):
    df_feat = df.drop(columns=[label_col])
    labels = df[label_col].astype(int).values

    attack_idxs = np.where(labels == 1)[0]
    if len(attack_idxs) == 0:
        start_attack = len(labels) + sequence_length
    else:
        start_attack = max(0, attack_idxs[0] - sequence_length)

    sequences = []
    for i in range(len(df_feat) - sequence_length):
        sequences.append(df_feat.iloc[i:i + sequence_length].values.flatten())

    if not sequences:
        return pd.DataFrame(columns=[*range(df_feat.shape[1] * sequence_length), "label"])

    seq_df = pd.DataFrame(sequences)
    zeros = [0] * min(start_attack, len(seq_df))
    ones = [1] * (len(seq_df) - len(zeros))
    seq_df["label"] = zeros + ones
    return seq_df


def extract_index(path):
    m = re.search(r"_(\d+)_60_sec\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def normalize_domain_id(name):
    s = str(name).strip().lower().replace("\\", "/").split("/")[-1]
    m = re.search(r"domain0*(\d+)$", s)
    if m:
        return f"domain{int(m.group(1))}"
    return s



DROP_COLS = ["Unnamed: 0", "cpu", "cpu.1", "lqi", "lqi.1"]

EXPERIMENT_FEATURES = {
    1: None,  # all remaining features after DROP_COLS => 20

    2: ["rssi", "rssi.1"],

    3: [
        "rank", "disr", "diss", "dior", "dios", "diar", "tots", "tx", "rx",
        "rank.1", "disr.1", "diss.1", "dior.1", "dios.1", "diar.1", "tots.1", "tx.1", "rx.1"
    ],  # all except rssi, rssi.1 => 18

    4: [
        "rank", "disr", "diss", "dior", "dios", "diar", "tots",
        "rank.1", "disr.1", "diss.1", "dior.1", "dios.1", "diar.1", "tots.1"
    ],  # all except rssi, rssi.1, rx, rx.1, tx, tx.1 => 14

    5: ["rx", "rx.1", "tx", "tx.1"],  # => 4
}


def load_csv(path, feature_cols=None):
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    assert "label" in df.columns, f"'label' column missing in {os.path.basename(path)}"

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns in {os.path.basename(path)}: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )
        df = df[feature_cols + ["label"]]

    return df


def save_results_as_json(results, filename, save_folder="results"):
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {filepath}")


def create_sliding_windows(X, y, window_size, step_size):
    sequences, labels = [], []
    for i in range(0, len(X) - window_size, step_size):
        sequences.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])
    return np.array(sequences), np.array(labels)


def load_data(domain_path, domain_dataset, window_size=10, batch_size=128, feature_cols=None):
    folder_key, domain_dataset = domain_dataset

    files = sorted(domain_dataset, key=extract_index)[:20]

    random.seed(42)
    random.shuffle(files)
    train_files_wo_path = files[:16]
    test_files_wo_path = files[16:20]

    train_files = [domain_path + "/" + folder_key + "/" + f for f in train_files_wo_path]
    test_files = [domain_path + "/" + folder_key + "/" + f for f in test_files_wo_path]

    train_dfs = [load_csv(p, feature_cols) for p in train_files]
    test_dfs = [load_csv(p, feature_cols) for p in test_files]

    feat_cols = [c for c in train_dfs[0].columns if c != "label"]
    train_feat_mins = [df[feat_cols].min(axis=0) for df in train_dfs]
    train_feat_maxs = [df[feat_cols].max(axis=0) for df in train_dfs]
    global_min = pd.concat(train_feat_mins, axis=1).min(axis=1)
    global_max = pd.concat(train_feat_maxs, axis=1).max(axis=1)

    norm_train = [safe_minmax_normalize(df, global_min, global_max, "label") for df in train_dfs]
    norm_test = [safe_minmax_normalize(df, global_min, global_max, "label") for df in test_dfs]

    seq_train_parts = [seq_maker(df, window_size, "label") for df in norm_train]
    seq_test_parts = [seq_maker(df, window_size, "label") for df in norm_test]

    seq_train_parts = [df for df in seq_train_parts if not df.empty]
    seq_test_parts = [df for df in seq_test_parts if not df.empty]

    seq_train = pd.concat(seq_train_parts, ignore_index=True)
    seq_test = pd.concat(seq_test_parts, ignore_index=True)

    X_train = torch.tensor(seq_train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(seq_train.iloc[:, -1].values.astype(int), dtype=torch.long)
    X_test = torch.tensor(seq_test.iloc[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(seq_test.iloc[:, -1].values.astype(int), dtype=torch.long)

    X_train = torch.nan_to_num(X_train, nan=0.0)
    X_test = torch.nan_to_num(X_test, nan=0.0)

    feature_dim = X_train.shape[1]
    X_train = X_train.view(-1, 1, feature_dim)
    X_test = X_test.view(-1, 1, feature_dim)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)

    return train_loader, test_loader


def create_domains(domains_path):
    details_path = os.path.join(os.path.dirname(domains_path), "domain_details.xlsx")
    df = pd.read_excel(details_path)

    df.columns = df.columns.str.strip()
    df["Domain Name"] = df["Domain Name"].astype(str).str.strip()
    df["Attack Type"] = df["Attack Type"].astype(str).str.strip().str.replace(" ", "_")
    df["Node"] = df["Node"].astype(str).str.strip()
    df["Version"] = df["Version"].astype(str).str.strip()

    df["Domain Name_norm"] = df["Domain Name"].apply(normalize_domain_id)

    domain_label_map = {
        row["Domain Name_norm"]: f"{row['Attack Type']}_{row['Node']}_{row['Version']}"
        for _, row in df.iterrows()
    }

    domains = {}
    for attack_type in os.listdir(domains_path):
        attack_path = os.path.join(domains_path, attack_type)
        if not os.path.isdir(attack_path):
            continue

        for domain in os.listdir(attack_path):
            domain_path = os.path.join(attack_path, domain)
            if not os.path.isdir(domain_path):
                continue

            files = sorted(f for f in os.listdir(domain_path) if f.endswith('.csv'))
            if not files:
                continue

            folder_key = os.path.join(attack_type, domain)
            domain_norm = normalize_domain_id(domain)
            label = domain_label_map.get(domain_norm, folder_key)
            domains[label] = (folder_key, files)

    logging.info(f"Domains found: {list(domains.keys())}")
    logging.info(f"Number of domains found: {len(domains)}")
    return domains


def compute_mmd(X1, X2, gamma=None):
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    Kxx = np.exp(-cdist(X1, X1, 'sqeuclidean') * gamma)
    Kyy = np.exp(-cdist(X2, X2, 'sqeuclidean') * gamma)
    Kxy = np.exp(-cdist(X1, X2, 'sqeuclidean') * gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def cluster_domains(base_path, distance_threshold=2.0):
    domain_paths = {}
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csvs:
            continue
        domain_paths[folder_name] = os.path.join(folder_path, csvs[0])

    scaler = StandardScaler()
    domain_features = {}
    for domain, path in domain_paths.items():
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
        X = df.drop(columns=['label'], errors='ignore').values
        X_scaled = scaler.fit_transform(X)
        domain_features[domain] = X_scaled

    domain_list = list(domain_features.keys())
    n = len(domain_list)
    mmd_matrix = np.zeros((n, n))
    for i in range(n):
        Xi = domain_features[domain_list[i]]
        for j in range(i, n):
            Xj = domain_features[domain_list[j]]
            m = compute_mmd(Xi, Xj)
            mmd_matrix[i, j] = m
            mmd_matrix[j, i] = m

    condensed = squareform(mmd_matrix)
    Z = linkage(condensed, method='ward')
    cluster_assignments = fcluster(Z, t=distance_threshold, criterion='distance')

    clusters = defaultdict(list)
    cluster_map = {}
    for idx, cid in enumerate(cluster_assignments):
        dom = domain_list[idx]
        cluster_map[dom] = cid
        clusters[cid].append(dom)

    return dict(clusters), cluster_map


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def confidence_from_logits(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    return probs, preds, confs


def _json_safe(obj):
    import numpy as np
    import torch

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    try:
        import pandas as pd
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
    except Exception:
        pass

    if isinstance(obj, (set, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with W&B logging")

    parser.add_argument("--domain", type=str, default="all",
                        help="Domain to train: 'all' or a specific domain name e.g. dis_flooding_15_gc")
    parser.add_argument("--project", type=str, default="attack_CL")
    parser.add_argument("--entity", type=str, default="sourasb05")
    parser.add_argument("--run_name", type=str, default="experiment-1")

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--architecture", type=str, default="LSTM")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--algorithm", type=str, default="GR")
    parser.add_argument("--scenario", type=str, default="random")
    parser.add_argument("--exp_no", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--input_size", type=int, default=140)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--output_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--forgetting_threshold", type=float, default=0.01)
    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--enc_lr_scale", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--memory_size", type=int, default=2000)
    parser.add_argument("--per_domain_cap", type=int, default=250)
    parser.add_argument("--replay_batch_size", type=int, default=128)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--replay_seen_only", action="store_true")

    parser.add_argument("--si_c", type=float, default=0.08)
    parser.add_argument("--si_xi", type=float, default=1e-3)
    parser.add_argument("--si_c_warmup_epochs", type=int, default=3)
    parser.add_argument("--si_c_schedule", type=str, default="cosine", choices=["const", "linear", "cosine"])
    parser.add_argument("--si_omega_clip", type=float, default=50.0)
    parser.add_argument("--si_exclude_bias_norm", action="store_true", default=True)
    parser.add_argument("--si_micro_consolidate_k", type=int, default=0)

    parser.add_argument("--alpha_min", type=float, default=0.3)
    parser.add_argument("--alpha_task_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--alpha_warmup_epochs", type=int, default=3)
    parser.add_argument("--alpha_warmup_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--T_max", type=float, default=5.0)
    parser.add_argument("--T_task_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--T_warmup_epochs", type=int, default=0)
    parser.add_argument("--T_warmup_schedule", type=str, default="const", choices=["const", "linear", "cosine"])

    parser.add_argument("--ewc_lambda", type=float, default=1400.0)
    parser.add_argument("--lambda_min", type=float, default=None)
    parser.add_argument("--lambda_task_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--lambda_warmup_epochs", type=int, default=4)
    parser.add_argument("--lambda_warmup_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--fisher_n_samples", type=int, default=64)
    parser.add_argument("--exclude_bias_norm", action="store_true", default=True)
    parser.add_argument("--fisher_clip", type=float, default=None)

    parser.add_argument("--gr_replay_ratio", type=float, default=0.5)
    parser.add_argument("--replay_samples_per_epoch", type=int, default=0)
    parser.add_argument("--use_teacher_labels", action="store_true", default=True)
    parser.add_argument("--distill_T", type=float, default=4.0)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--vae_hidden", type=int, default=64)
    parser.add_argument("--vae_latent", type=int, default=32)
    parser.add_argument("--vae_epochs", type=int, default=30)
    parser.add_argument("--vae_lr", type=float, default=1e-3)
    parser.add_argument("--vae_batch_size", type=int, default=128)
    parser.add_argument("--vae_beta_start", type=float, default=0.0)
    parser.add_argument("--vae_beta_end", type=float, default=1.0)
    parser.add_argument("--vae_window_size", type=int, default=10)

    parser.add_argument("--num_features", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.lambda_min is None:
        args.lambda_min = 0.6 * args.ewc_lambda

    return args