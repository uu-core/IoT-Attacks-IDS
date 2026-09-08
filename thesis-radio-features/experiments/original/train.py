import numpy as np
import torch
import torch.nn as nn
import os
import logging
import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import models as models
import utils as utils


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
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

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc       = (all_preds == all_labels).mean()
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    auc       = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    cm        = confusion_matrix(all_labels, all_preds).tolist()

    return total_loss / total, {
        "accuracy":         round(float(acc), 4),
        "f1":               round(float(f1), 4),
        "precision":        round(float(precision), 4),
        "recall":           round(float(recall), 4),
        "auc":              round(float(auc), 4),
        "confusion_matrix": cm,
    }


def train_all_domains(args, domains, domains_path, input_size, feature_cols, device, current_directory, timestamp):
    criterion = nn.CrossEntropyLoss()
    exp_no = args.exp_no
    total_domains = len(domains)

    model_dir = os.path.join(current_directory, "saved_models", f"exp{exp_no}")
    os.makedirs(model_dir, exist_ok=True)

    results = {}
    domain_bar = tqdm(domains.items(), total=total_domains, desc="Domains", unit="domain", position=0)

    for domain_idx, (domain_key, domain_data) in enumerate(domain_bar, start=1):
        domain_bar.set_description(f"Domain [{domain_idx}/{total_domains}] {domain_key}")
        logging.info(f"=== Training domain: {domain_key} ({domain_idx}/{total_domains}) ===")

        train_loader, test_loader = utils.load_data(
            domains_path, domain_data,
            window_size=args.window_size,
            batch_size=args.batch_size,
            feature_cols=feature_cols,
        )

        model = models.LSTMClassifier(
            input_dim=input_size,
            hidden_dim=args.hidden_size,
            output_dim=args.output_size,
            num_layers=args.num_layers,
            fc_hidden_dim=10,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val_loss = float('inf')
        patience_counter = 0

        epoch_bar = tqdm(range(1, args.epochs + 1), desc="  Epochs", unit="epoch", position=1, leave=False)
        for epoch in epoch_bar:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_metrics = evaluate(model, test_loader, criterion, device)
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}",     val_acc=f"{val_metrics['accuracy']:.4f}",
            )
            logging.info(f"  [{domain_key}] Epoch {epoch}/{args.epochs} | "
                         f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                         f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, f"{domain_key}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logging.info(f"  [{domain_key}] Early stopping at epoch {epoch}")
                    break
        epoch_bar.close()

        model.load_state_dict(torch.load(os.path.join(model_dir, f"{domain_key}.pt"), map_location=device))
        _, test_metrics = evaluate(model, test_loader, criterion, device)
        results[domain_key] = test_metrics
        domain_bar.write(f"  [{domain_idx}/{total_domains}] {domain_key} | "
                         f"acc={test_metrics['accuracy']} f1={test_metrics['f1']} "
                         f"auc={test_metrics['auc']}")
        logging.info(f"  [{domain_key}] accuracy={test_metrics['accuracy']} f1={test_metrics['f1']} "
                     f"precision={test_metrics['precision']} recall={test_metrics['recall']} "
                     f"auc={test_metrics['auc']} cm={test_metrics['confusion_matrix']}")

    domain_bar.close()
    logging.info("=== All domains done ===")

    for domain_key, metrics in results.items():
        domain_results_dir = os.path.join(current_directory, "results", f"exp_features_{exp_no}", domain_key)
        os.makedirs(domain_results_dir, exist_ok=True)
        results_file = os.path.join(domain_results_dir, "metrics.json")
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Results saved to {results_file}")

    return results