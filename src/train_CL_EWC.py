import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import logging
import sys
import warnings
warnings.filterwarnings("ignore")
from utils import save_results_as_json, _sync
import evaluate_model
# import result_utils
import evaluation as evaluate
import result_utils as result_utils
import wandb
from tqdm import tqdm, trange

# ===================================
# Elastic Weight Consolidation Class
# ===================================
class EWC:
    def __init__(self, model, dataloader, device, lambda_=1150, fisher_n_samples=None):
        self.model = model
        self.device = device
        self.lambda_ = lambda_

        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.prev_params = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher = self._compute_fisher(dataloader, fisher_n_samples)

    def _compute_fisher(self, dataloader, n_samples=None):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()}
        self.model.train()
        criterion = nn.CrossEntropyLoss()

        count = 0
        for X_batch, y_batch in dataloader:
            if n_samples is not None and count >= n_samples:
                break
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.model.zero_grad()
            outputs, _ = self.model(X_batch)
            loss = criterion(outputs, y_batch.long())
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1

        for n in fisher:
            fisher[n] /= count if count > 0 else 1

        return fisher

    def penalty(self):
        reg = 0.0
        for n, p in self.params.items():
            delta = p - self.prev_params[n]
            reg += (self.fisher[n] * delta ** 2).sum()
        return self.lambda_ * reg

def tdim_ewc_random(
    args, run_wandb, train_domain_loader, test_domain_loader, train_domain_order, device,
    model, exp_no, num_epochs=500, learning_rate=0.01, patience=3
):
    """
    - Track F1,AUC and confusion matrices
    -   Compute BWT for F1 and AUC
    -   Save richer JSON (plasticity/stability for F1 & AUC, CMs, costs)
    -   plain EWC penalty, no λ scheduling) remains the same.
    """
   

    exp_no = exp_no
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Metric containers (mirroring Code 1) ---
    performance_stability   = {d: [] for d in test_domain_loader.keys()}  # F1 on prior domains after training current
    performance_plasticity  = {d: [] for d in test_domain_loader.keys()}  # F1 pre & post on current domain
    roc_auc_stability       = {d: [] for d in test_domain_loader.keys()}
    roc_auc_plasticity      = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_stability  = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_plasticity = {d: [] for d in test_domain_loader.keys()}
    domain_training_cost    = {d: [] for d in test_domain_loader.keys()}

    seen_domain = set()
    print(f"Training on {len(train_domain_order)} domains: {train_domain_order}")
    domain_to_id = {name: i for i, name in enumerate(train_domain_order)}
    ewc_list = []  # Keep EWC objects (plain/fixed-λ)

    # ---- W&B config (make explicit EWC) ----
    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CrossEntropyLoss + EWC",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "train_domains": train_domain_order
    })
    run_wandb.watch(model, criterion=criterion, log="all", log_freq=50)

    previous_domain = None
    best_model_state = None

    # Helper: total EWC penalty (plain, no scheduling)
    def total_ewc_penalty():
        return sum([ewc.penalty() for ewc in ewc_list]) if ewc_list else 0.0

    for idx, train_domain in enumerate(tqdm(list(train_domain_order),
                                            desc="Train Domains", total=len(train_domain_order))):

        domain_id = domain_to_id[train_domain]
        domain_epoch = 0
        if args.use_wandb:
            import wandb
            wandb.define_metric(f"{train_domain}/epoch")
            wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        logging.info(f"====== Evaluate current domain {train_domain} with model from: {previous_domain} ======")

        # ---- Pre-train evaluation on current domain (Plasticity: PRE) ----
        if idx != 0:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            else:
                logging.warning("best_model_state is uninitialized. Skipping model loading.")
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                m_pre = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
            else:
                m_pre = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)

            f1_pre  = float(m_pre["f1"])
            auc_pre = float(m_pre.get("roc_auc", 0.0))
            cm_pre  = m_pre.get("confusion_matrix", None)

            performance_plasticity[train_domain].append(f1_pre)
            roc_auc_plasticity[train_domain].append(auc_pre)
            confusion_matrix_plasticity[train_domain].append(cm_pre)

            run_wandb.log({
                f"{train_domain}/pretrain_f1": f1_pre,
                f"{train_domain}/pretrain_ROC_AUC": auc_pre
            })
            logging.info(f"[PRE] {train_domain}: F1={f1_pre:.4f} | AUC={auc_pre:.4f}")

        logging.info(f"====== Training on Domain: {train_domain} (EWC) ======")

        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        # ---- Train loop ----
        for epoch in trange(num_epochs, desc="training Epochs"):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                if args.architecture == "LSTM_Attention_adapter":
                    outputs, _ = model(X_batch, domain_id=domain_id)
                else:
                    outputs, _ = model(X_batch)

                ce = criterion(outputs, y_batch.long())
                loss = ce + total_ewc_penalty()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.item())

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),
            })

            # ---- Validation on current domain ----
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for Xb, yb in test_domain_loader[train_domain]:
                    Xb, yb = Xb.to(device), yb.to(device)
                    if args.architecture == "LSTM_Attention_adapter":
                        out, _ = model(Xb, domain_id=domain_id)
                    else:
                        out, _ = model(Xb)
                    loss_val = criterion(out, yb.long())
                    _, pred = torch.max(out.data, 1)
                    all_y_true.extend(yb.cpu().numpy())
                    all_y_pred.extend(pred.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(out, dim=1)[:, 1].cpu().numpy())
                    test_loss += float(loss_val.item())
            test_loss /= max(1, len(test_domain_loader[train_domain]))

            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
                train_domain, train_domain
            )
            current_f1 = float(metrics["f1"])
            current_auc_roc = float(metrics.get("roc_auc", 0.0))

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} | Val Loss: {test_loss:.4f} | F1: {current_f1:.4f} | AUC-ROC: {current_auc_roc:.4f}")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(current_f1),
                f"{train_domain}/val_ROC_AUC": float(current_auc_roc)
            })

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
                logging.info(f"New best F1 for {train_domain}: {best_f1:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement. Count: {epochs_no_improve}")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered for {train_domain} at epoch {epoch+1}")
                break

        _sync(device)
        domain_training_time = time.perf_counter() - t0
        logging.info(f"Training time for {train_domain}: {domain_training_time:.2f} seconds")
        domain_training_cost[train_domain].append(float(domain_training_time))

        # ---- Restore best and save checkpoint ----
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}_{args.architecture}_{args.algorithm}_{args.scenario}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
            previous_domain = train_domain
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")

        # ---- Build EWC object for this domain (plain/fixed-λ) ----
        model.train()
        ewc_instance = EWC(model, train_domain_loader[train_domain], device, lambda_=1200, fisher_n_samples=None)
        ewc_list.append(ewc_instance)

        # ---- Post-train evaluation on current domain (Plasticity: POST; Stability: add current too) ----
        model.load_state_dict(best_model_state)
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            m_post = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            m_post = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)

        f1_post  = float(m_post["f1"])
        auc_post = float(m_post.get("roc_auc", 0.0))
        cm_post  = m_post.get("confusion_matrix", None)

        performance_plasticity[train_domain].append(f1_post)
        performance_stability[train_domain].append(f1_post)
        roc_auc_plasticity[train_domain].append(auc_post)
        roc_auc_stability[train_domain].append(auc_post)
        confusion_matrix_plasticity[train_domain].append(cm_post)
        confusion_matrix_stability[train_domain].append(cm_post)

        logging.info(f"[POST] {train_domain}: F1={f1_post:.4f} | AUC={auc_post:.4f}")

        # ---- Stability across seen (previous) domains ----
        logging.info(f"====== Evaluating on all previous domains after training on {train_domain} ======")
        seen_domain.add(train_domain)
        for test_domain in tqdm(seen_domain, desc="Stability test"):
            if test_domain == train_domain:
                continue
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                m_prev = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=domain_to_id[test_domain])
            else:
                m_prev = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=None)

            f1_prev  = float(m_prev["f1"])
            auc_prev = float(m_prev.get("roc_auc", 0.0))
            cm_prev  = m_prev.get("confusion_matrix", None)

            performance_stability[test_domain].append(f1_prev)
            roc_auc_stability[test_domain].append(auc_prev)
            confusion_matrix_stability[test_domain].append(cm_prev)

            logging.info(f"Stability | {test_domain}: F1={f1_prev:.4f} | AUC={auc_prev:.4f}")

        print(f"====== Finished Training on Domain: {train_domain} ======")

    # ===== Final Metrics (F1) =====
    logging.info("====== Final Metrics (F1) ======")
    bwt_values_f1, bwt_dict_f1, bwt_values_dict_f1 = result_utils.compute_BWT(performance_stability, train_domain_order)
    plasticity_values_f1, plasticity_dict_f1       = result_utils.compute_plasticity(performance_plasticity, train_domain_order)
    logging.info(f"BWT (F1): {bwt_values_f1}")
    logging.info(f"BWT per domain (F1): {bwt_dict_f1}")
    logging.info(f"Plasticity (F1): {plasticity_values_f1}")

    # ===== Final Metrics (AUC) =====
    logging.info("====== Final Metrics (ROC-AUC) ======")
    bwt_values_auc, bwt_dict_auc, bwt_values_dict_auc = result_utils.compute_BWT(roc_auc_stability, train_domain_order)
    plasticity_values_auc, plasticity_dict_auc        = result_utils.compute_plasticity(roc_auc_plasticity, train_domain_order)
    logging.info(f"BWT (AUC): {bwt_values_auc}")
    logging.info(f"BWT per domain (AUC): {bwt_dict_auc}")
    logging.info(f"Plasticity (AUC): {plasticity_values_auc}")

    # ===== Save richer JSON =====
    results_to_save = {
        "exp_no": exp_no,
        "train_domain_order": train_domain_order,

        # F1 series
        "performance_stability": performance_stability,
        "performance_m": performance_plasticity,

        # AUC series
        "roc_auc_stability": roc_auc_stability,
        "roc_auc_plasticity": roc_auc_plasticity,

        # Confusions & cost
        "confusion_matrix_stability": confusion_matrix_stability,
        "confusion_matrix_plasticity": confusion_matrix_plasticity,
        "domain_training_cost": domain_training_cost,

        # Final aggregates (F1)
        "BWT_values_f1": bwt_values_f1,
        "BWT_dict_f1": bwt_dict_f1,
        "plasticity_values_f1": plasticity_values_f1,
        "plasticity_dict_f1": plasticity_dict_f1,

        # Final aggregates (AUC)
        "BWT_values_auc": bwt_values_auc,
        "BWT_dict_auc": bwt_dict_auc,
        "plasticity_values_auc": plasticity_values_auc,
        "plasticity_dict_auc": plasticity_dict_auc
    }

    from utils import _json_safe, save_results_as_json
    results_to_save = _json_safe(results_to_save)
    out_name = f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_{args.scenario}.json"
    save_results_as_json(results_to_save, filename=out_name)
    logging.info(f"Final training complete. Results saved to {out_name}")

    # W&B summaries (optional)
    run_wandb.summary["BWT_F1/list"]  = bwt_values_f1
    run_wandb.summary["BWT_AUC/list"] = bwt_values_auc
