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
import wandb

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
        start_attack = len(labels) + sequence_length   # all zeros
    else:
        start_attack = max(0, attack_idxs[0] - sequence_length)

    sequences = []
    for i in range(len(df_feat) - sequence_length):
        sequences.append(df_feat.iloc[i:i+sequence_length].values.flatten())

    if not sequences:
        return pd.DataFrame(columns=[*range(df_feat.shape[1]*sequence_length), "label"])

    seq_df = pd.DataFrame(sequences)
    zeros = [0] * min(start_attack, len(seq_df))
    ones  = [1] * (len(seq_df) - len(zeros))
    seq_df["label"] = zeros + ones
    return seq_df

def extract_index(path):
    # pull the number between ..._ and _60_sec.csv
    m = re.search(r"_(\d+)_60_sec\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9  # push unknowns to end
DROP_COLS = ["Unnamed: 0"]
def load_csv(path):
    
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    assert "label" in df.columns, f"'label' column missing in {os.path.basename(path)}"
    return df


def save_results_as_json(results, filename, save_folder="results"):
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {filepath}")


# Function to create sliding windows
def create_sliding_windows(X, y, window_size, step_size):
    sequences, labels = [], []
    for i in range(0, len(X) - window_size, step_size):
        sequences.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])  # Label is the last time step in the window
    return np.array(sequences), np.array(labels)


def load_data(domain_path, key, domain_dataset, window_size=10, step_size=3, batch_size=128):

    files = sorted(domain_dataset, key=extract_index)[:20]  # ensure exactly 20, ordered

    DROP_COLS = ["Unnamed: 0"]
    
    random.seed(42)
    random.shuffle(files)
    train_files_wo_path = files[:16]
    test_files_wo_path  = files[16:20]


    train_files = [domain_path + "/" + key  + "/" + f for f in train_files_wo_path]
    test_files = [domain_path + "/" + key + "/" + f for f in test_files_wo_path]


    

    # print("Train files:", train_files) 
    # print("Test  files:", test_files)

    # Load your data (assuming it's already loaded as `data`)
    # -----------------------
    # Load all, compute train-only global min/max (excluding 'label')
    # -----------------------
    train_dfs = [load_csv(p) for p in train_files]
    test_dfs  = [load_csv(p) for p in test_files]

    feat_cols = [c for c in train_dfs[0].columns if c != "label"]
    train_feat_mins = [df[feat_cols].min(axis=0) for df in train_dfs]
    train_feat_maxs = [df[feat_cols].max(axis=0) for df in train_dfs]
    global_min = pd.concat(train_feat_mins, axis=1).min(axis=1)
    global_max = pd.concat(train_feat_maxs, axis=1).max(axis=1)

    # -----------------------
    # Normalize using train stats
    # -----------------------
    norm_train = [safe_minmax_normalize(df, global_min, global_max, "label") for df in train_dfs]
    norm_test  = [safe_minmax_normalize(df, global_min, global_max, "label") for df in test_dfs]

    # -----------------------
    # Sequence-ify and concat
    # -----------------------
    seq_train_parts = [seq_maker(df, window_size, "label") for df in norm_train]
    seq_test_parts  = [seq_maker(df, window_size, "label") for df in norm_test]

    seq_train_parts = [df for df in seq_train_parts if not df.empty]
    seq_test_parts  = [df for df in seq_test_parts if not df.empty]

    seq_train = pd.concat(seq_train_parts, ignore_index=True)
    seq_test  = pd.concat(seq_test_parts,  ignore_index=True)


    # -----------------------
    # Tensors & Dataloaders
    # -----------------------
    X_train = torch.tensor(seq_train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(seq_train.iloc[:,  -1].values.astype(int), dtype=torch.long)
    X_test  = torch.tensor(seq_test.iloc[:,  :-1].values, dtype=torch.float32)
    y_test  = torch.tensor(seq_test.iloc[:,   -1].values.astype(int), dtype=torch.long)

    X_train = torch.nan_to_num(X_train, nan=0.0)
    X_test  = torch.nan_to_num(X_test,  nan=0.0)

    feature_dim = X_train.shape[1]  # (#features * SEQUENCE_LENGTH)
    X_train = X_train.view(-1, 1, feature_dim)
    X_test  = X_test.view(-1, 1, feature_dim)
    
    # print(f"After view: X_train={X_train.shape}, X_test={X_test.shape}")


    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  len(test_dataset), shuffle=False)


    # print("X_train:", X_train.shape, "y_train:", y_train.shape)
    # print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    return train_loader, test_loader

def create_domains(domains_path):

        # Iterate over each item in the source folder.
    domains = {}
    for domain  in os.listdir(domains_path):
        
        domain_path = os.path.join(domains_path, domain)
        
        if os.path.isdir(domain_path):
            files = sorted(os.listdir(domain_path))
            selected_files = files
           # Use the subfolder name as the dictionary key and its list of file contents as the value.
            domains[domain] = selected_files
    logging.info(f"Domains found: {domains.keys()}")
    logging.info(f"Number of domains found: {len(domains)}")

    # for key, value_list in domains.items():
    #     print(f"key : {key} value : {value_list} num_element : {len(value_list)}")
    return domains



def compute_mmd(X1, X2, gamma=None):
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    Kxx = np.exp(-cdist(X1, X1, 'sqeuclidean') * gamma)
    Kyy = np.exp(-cdist(X2, X2, 'sqeuclidean') * gamma)
    Kxy = np.exp(-cdist(X1, X2, 'sqeuclidean') * gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def cluster_domains(base_path, distance_threshold=2.0):
    # 1) find one CSV per domain‐folder
    domain_paths = {}
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csvs:
            continue
        domain_paths[folder_name] = os.path.join(folder_path, csvs[0])

    # 2) load & standardize features
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

    # 3) hierarchical clustering on the condensed form of mmd_matrix
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
    probs = torch.softmax(logits, dim=1)     # (B, C)
    confs, preds = probs.max(dim=1)          # (B,), (B,)
    return probs, preds, confs



def _json_safe(obj):
    """Recursively convert NumPy/PyTorch/Pandas/sets/tuples into JSON-serializable types."""
    import numpy as np
    import torch

    # NumPy scalars
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # PyTorch tensors
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # Pandas types
    try:
        import pandas as pd
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
    except Exception:
        pass

    # Builtins that need conversion
    if isinstance(obj, (set, tuple)):
        return [_json_safe(x) for x in obj]

    # Dicts / lists: recurse
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]

    # Everything else: leave as is (int/float/str/bool/None)
    return obj



import argparse  # make sure this is imported

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with W&B logging")

    # --- W&B ---
    parser.add_argument("--project", type=str, default="attack_CL")
    parser.add_argument("--entity", type=str, default="sourasb05")
    parser.add_argument("--run_name", type=str, default="experiment-1")

    # --- Model / training ---
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--architecture", type=str, default="LSTM")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--algorithm", type=str, default="GR")
    parser.add_argument("--scenario", type=str, default="random")
    parser.add_argument("--exp_no", type=int, default=1)
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

    # --- Distillation (LwF core) ---
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="LwF KD weight (higher=stability↑, plasticity↓).")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="LwF softmax temperature (2–5).")
    parser.add_argument("--enc_lr_scale", type=float, default=0.5,
                        help="Encoder LR scale vs head LR (0.3–0.7).")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Head-only warmup epochs before joint training.")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # --- Replay (unchanged) ---
    parser.add_argument("--memory_size", type=int, default=2000)
    parser.add_argument("--per_domain_cap", type=int, default=250)
    parser.add_argument("--replay_batch_size", type=int, default=128)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--replay_seen_only", action="store_true")

    # --- SI specific (you already had these; keep as-is) ---
    parser.add_argument("--si_c", type=float, default=0.08)
    parser.add_argument("--si_xi", type=float, default=1e-3)
    parser.add_argument("--si_c_warmup_epochs", type=int, default=3)
    parser.add_argument("--si_c_schedule", type=str, default="cosine", choices=["const","linear","cosine"])
    parser.add_argument("--si_omega_clip", type=float, default=50.0)
    parser.add_argument("--si_exclude_bias_norm", action="store_true", default=True)
    parser.add_argument("--si_micro_consolidate_k", type=int, default=0)

    # === NEW: LwF scheduling knobs (to match your updated LwF trainer) ===
    parser.add_argument("--alpha_min", type=float, default=0.3,
                        help="Across-task α start; ramps to --alpha (cos/lin).")
    parser.add_argument("--alpha_task_schedule", type=str, default="cosine",
                        choices=["linear","cosine"], help="Across-task α schedule.")
    parser.add_argument("--alpha_warmup_epochs", type=int, default=3,
                        help="Within-task α warmup epochs 0→α_target.")
    parser.add_argument("--alpha_warmup_schedule", type=str, default="cosine",
                        choices=["linear","cosine"], help="Within-task α schedule.")
    parser.add_argument("--T_max", type=float, default=5.0,
                        help="Across-task T target (>= --temperature).")
    parser.add_argument("--T_task_schedule", type=str, default="linear",
                        choices=["linear","cosine"], help="Across-task T schedule.")
    parser.add_argument("--T_warmup_epochs", type=int, default=0,
                        help="Within-task T warmup epochs (0=off).")
    parser.add_argument("--T_warmup_schedule", type=str, default="const",
                        choices=["const","linear","cosine"], help="Within-task T schedule.")

    # === NEW: EWC knobs + schedules (to match your updated EWC trainer) ===
    parser.add_argument("--ewc_lambda", type=float, default=1400.0,
                        help="Target EWC λ (stability↑ with larger values).")
    parser.add_argument("--lambda_min", type=float, default=None,
                        help="Across-task λ start; if None, set to 0.6*ewc_lambda.")
    parser.add_argument("--lambda_task_schedule", type=str, default="cosine",
                        choices=["linear","cosine"], help="Across-task λ schedule.")
    parser.add_argument("--lambda_warmup_epochs", type=int, default=4,
                        help="Within-task λ warmup epochs 0→λ_target.")
    parser.add_argument("--lambda_warmup_schedule", type=str, default="cosine",
                        choices=["linear","cosine"], help="Within-task λ schedule.")
    parser.add_argument("--fisher_n_samples", type=int, default=64,
                        help="Batches to estimate Fisher (None=all).")
    parser.add_argument("--exclude_bias_norm", action="store_true", default=True,
                        help="Exclude bias/Norm params from EWC penalty.")
    parser.add_argument("--fisher_clip", type=float, default=None,
                        help="Max clip for Fisher values (None=off).")
    

    # --- GR (Generative Replay) specific (NEW) ---
    # === Generative Replay (GR) ===
    parser.add_argument("--gr_replay_ratio", type=float, default=0.5,
        help="r in loss mix: loss = r*CE(real) + (1-r)*KL(replay). "
            "Use this for GR to avoid confusion with exemplar replay's replay_ratio.")
    parser.add_argument("--replay_samples_per_epoch", type=int, default=0,
        help="If >0, fixed #synthetic sequences generated per epoch; else auto from gr_replay_ratio & #real.")

    parser.add_argument("--use_teacher_labels", action="store_true", default=True,
        help="Use previous solver as a teacher to provide soft targets for generated samples.")
    parser.add_argument("--distill_T", type=float, default=4.0,
        help="Distillation temperature for teacher soft targets (Hinton T^2 scaling).")
    parser.add_argument("--num_classes", type=int, default=2,
        help="Fallback classes when no teacher is available (uniform soft targets).")

    # VAE generator
    parser.add_argument("--vae_hidden", type=int, default=64,
        help="LSTM hidden size for the VAE.")
    parser.add_argument("--vae_latent", type=int, default=32,
        help="Latent dimension for the VAE.")
    parser.add_argument("--vae_epochs", type=int, default=30,
        help="Epochs to train VAE each domain.")
    parser.add_argument("--vae_lr", type=float, default=1e-3,
        help="VAE learning rate.")
    parser.add_argument("--vae_batch_size", type=int, default=128,
        help="Batch size for VAE training (often max(32, batch_size//2)).")
    parser.add_argument("--vae_beta_start", type=float, default=0.0,
        help="β-VAE schedule start value.")
    parser.add_argument("--vae_beta_end", type=float, default=1.0,
        help="β-VAE schedule end value.")
    parser.add_argument("--vae_window_size", type=int, default=10)


    # Shapes / reproducibility
    parser.add_argument("--num_features", type=int, default=140,
        help="Per-timestep feature dimension (usually = input_size).")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for GR/VAEs.")

    args = parser.parse_args()

    # Derived default: if lambda_min not set, tie to ewc_lambda
    if args.lambda_min is None:
        args.lambda_min = 0.6 * args.ewc_lambda

    return args
