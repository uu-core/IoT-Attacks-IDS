"""
Cross-domain evaluation script.

Two modes (--mode):

  single   Load one model and test it on selected domains.
  sweep    For every model in --model_exp, test it across all (or selected) domains.

Results saved to:
  results/cross_test/exp{model_exp}/{model_domain}/vs_{test_domain}.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from tqdm import tqdm

import models as models
import utils as utils


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            total += len(y_batch)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return {
        "accuracy": round(float((all_preds == all_labels).mean()), 4),
        "f1": round(float(f1_score(all_labels, all_preds, zero_division=0)), 4),
        "precision": round(float(precision_score(all_labels, all_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(all_labels, all_preds, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(all_labels, all_probs)
                           if len(np.unique(all_labels)) > 1 else 0.0), 4),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


def load_model(model_path, input_size, args, device):
    model = models.LSTMClassifier(
        input_dim=input_size,
        hidden_dim=args.hidden_size,
        output_dim=args.output_size,
        num_layers=args.num_layers,
        fc_hidden_dim=10,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_test_loader(domains_path, domain_data, args, feature_cols):
    _, test_loader = utils.load_data(
        domains_path,
        domain_data,
        window_size=args.window_size,
        batch_size=args.batch_size,
        feature_cols=feature_cols,
    )
    return test_loader


def save_result(result, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=4)
    return path


def run_single(args, domains, domains_path, model_dir, feature_cols, input_size, device, current_dir):
    model_path = os.path.join(model_dir, f"{args.model_domain}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path, input_size, args, device)

    test_domains = domains if args.test_domain == "all" else {
        args.test_domain: domains[args.test_domain]
    }

    results = {}
    for test_key, test_data in tqdm(test_domains.items(), desc=f"Testing {args.model_domain}"):
        loader = get_test_loader(domains_path, test_data, args, feature_cols)
        metrics = evaluate(model, loader, device)
        results[test_key] = metrics

        out_dir = os.path.join(
            current_dir, "results", "cross_test",
            f"exp{args.model_exp}", args.model_domain
        )
        path = save_result(metrics, out_dir, f"vs_{test_key}.json")

        tqdm.write(
            f"  {args.model_domain} -> {test_key} | "
            f"acc={metrics['accuracy']} f1={metrics['f1']} auc={metrics['auc']}  [{path}]"
        )
        logging.info(f"{args.model_domain} -> {test_key}: {metrics}")

    return results


def run_sweep(args, domains, domains_path, model_dir, feature_cols, input_size, device, current_dir):
    test_domains = domains if args.test_domain == "all" else {
        args.test_domain: domains[args.test_domain]
    }

    model_keys = [k for k in domains.keys() if os.path.exists(os.path.join(model_dir, f"{k}.pt"))]
    if not model_keys:
        raise FileNotFoundError(f"No .pt files found in {model_dir}")

    all_results = {}
    model_bar = tqdm(model_keys, desc="Models", unit="model", position=0)

    for model_key in model_bar:
        model_bar.set_description(f"Model: {model_key}")
        model_path = os.path.join(model_dir, f"{model_key}.pt")
        model = load_model(model_path, input_size, args, device)

        all_results[model_key] = {}
        test_bar = tqdm(
            test_domains.items(),
            desc="  Testing",
            unit="domain",
            position=1,
            leave=False
        )

        for test_key, test_data in test_bar:
            test_bar.set_description(f"  {model_key} -> {test_key}")
            loader = get_test_loader(domains_path, test_data, args, feature_cols)
            metrics = evaluate(model, loader, device)
            all_results[model_key][test_key] = metrics

            out_dir = os.path.join(
                current_dir, "results", "cross_test",
                f"exp{args.model_exp}", model_key
            )
            save_result(metrics, out_dir, f"vs_{test_key}.json")
            logging.info(f"{model_key} -> {test_key}: {metrics}")

        test_bar.close()
        model_bar.write(f"  {model_key} done — tested on {len(test_domains)} domain(s)")

    model_bar.close()
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "sweep"],
        required=True
    )
    parser.add_argument("--model_exp", type=int, required=True)
    parser.add_argument("--model_domain", type=str, default=None)
    parser.add_argument("--test_domain", type=str, default="all")

    parser.add_argument("--exp_no", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--output_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "single" and args.model_domain is None:
        raise ValueError("--model_domain is required in single mode")

    exp_no = args.exp_no if args.exp_no is not None else args.model_exp
    args.exp_no = exp_no

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    current_dir = os.getcwd()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    domains_path = os.path.join(os.path.dirname(current_dir), "attack_data")
    domains = utils.create_domains(domains_path)

    if args.test_domain != "all" and args.test_domain not in domains:
        raise ValueError(f"Test domain '{args.test_domain}' not found. Available: {list(domains.keys())}")

    feature_cols = utils.EXPERIMENT_FEATURES[exp_no]

    if feature_cols is not None:
        num_features = len(feature_cols)
    else:
        sample_domain_key = next(iter(domains))
        folder_key, files = domains[sample_domain_key]
        sample_file = os.path.join(domains_path, folder_key, files[0])
        sample_df = utils.load_csv(sample_file, feature_cols=None)
        num_features = len([c for c in sample_df.columns if c != "label"])

    input_size = num_features * args.window_size

    model_dir = os.path.join(current_dir, "saved_models", f"exp{args.model_exp}")

    if args.mode == "single":
        run_single(args, domains, domains_path, model_dir, feature_cols, input_size, device, current_dir)
    else:
        run_sweep(args, domains, domains_path, model_dir, feature_cols, input_size, device, current_dir)


if __name__ == "__main__":
    main()