import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from copy import deepcopy
import logging
import sys
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm, trange
import wandb

from utils import save_results_as_json, _sync, _json_safe
import evaluate_model
import evaluation as evaluate
import result_utils as result_utils


# ===================================
# Synaptic Intelligence (SI) Class
# ===================================
class SynapticIntelligence:
    """
    Implements SI (Zenke, Poole, Ganguli, ICML'17).
      - Online credit ω_k: accumulate sum_t g_k(t) * Δθ_k(t) within the current task
      - Consolidation at task end:
            Ω_k += (-ω_k) / ((Δ_k_task)^2 + ξ),
            θ*_k ← θ_k,
            ω_k ← 0
      - Penalty for subsequent tasks:
            c * Σ_k Ω_k (θ_k - θ*_k)^2
    """
    def __init__(self, model, device, c=0.1, xi=1e-3):
        self.model = model
        self.device = device
        self.c = float(c)     # trade-off old vs new
        self.xi = float(xi)   # damping to avoid division by zero

        # Track only trainable params
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        # Reference weights (θ* at last consolidation), cumulative importance Ω, online credits ω
        self.theta_star = {n: p.detach().clone() for n, p in self.params.items()}
        self.Omega      = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}
        self.omega      = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}

        # Per-step snapshot for Δθ
        self.p_old      = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}

    @torch.no_grad()
    def snapshot_before_step(self):
        """Call right BEFORE optimizer.step(): stores θ_old for Δθ."""
        for n, p in self.params.items():
            self.p_old[n].copy_(p.detach())

    @torch.no_grad()
    def accumulate_from(self, grads_by_name):
        """
        Call right AFTER optimizer.step(): uses grads of pure task loss (no SI penalty)
        and actual Δθ to accumulate ω_k ← ω_k + g_k * Δθ_k
        """
        for n, p in self.params.items():
            g = grads_by_name.get(n, None)
            if g is None:
                continue
            delta = (p.detach() - self.p_old[n])
            self.omega[n] += g.detach() * delta

    def penalty(self):
        """c * Σ_k Ω_k (θ - θ*)^2"""
        reg = 0.0
        for n, p in self.params.items():
            reg = reg + (self.Omega[n] * (p - self.theta_star[n]).pow(2)).sum()
        return self.c * reg

    @torch.no_grad()
    def consolidate_task_end(self):
        """
        At the end of the current task:
          Ω_k += (-ω_k) / ((Δ_k_task)^2 + ξ),
          θ*_k ← θ_k,
          ω_k ← 0
        where Δ_k_task = θ_k(end) - θ*_k(previous)
        """
        for n, p in self.params.items():
            delta_task = (p.detach() - self.theta_star[n])
            self.Omega[n] += (-self.omega[n]) / (delta_task.pow(2) + self.xi)
            self.theta_star[n].copy_(p.detach())
            self.omega[n].zero_()


# ===================================
# Main Training Function with SI (+ Code-1 metrics; NO FWT)
# ===================================
def tdim_si(args, run_wandb, train_domain_loader, test_domain_loader, train_domain_order, device,
            model, exp_no, num_epochs=500, learning_rate=0.01, patience=3,
            si_c=0.1, si_xi=1e-3):
    """
    SI training with:
      - one SI object across the whole run
      - per-domain early stopping by F1
      - pretrain eval (plasticity), post-train stability eval on seen domains
      - W&B logging
      - saves best model per domain
      - computes BWT & Plasticity for F1/AUC, stores confusion matrices & training cost
      - NO FWT anywhere
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- SI object (persists across domains) ---
    si = SynapticIntelligence(model, device, c=si_c, xi=si_xi)

    # ===== Metrics (F1 kept; add AUC + Confusion Matrices like Code 1) =====
    performance_stability   = {d: [] for d in test_domain_loader.keys()}   # F1 stability
    performance_plasticity  = {d: [] for d in test_domain_loader.keys()}   # F1 plasticity (pre + post)
    roc_auc_stability       = {d: [] for d in test_domain_loader.keys()}
    roc_auc_plasticity      = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_stability  = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_plasticity = {d: [] for d in test_domain_loader.keys()}
    domain_training_cost    = {d: [] for d in test_domain_loader.keys()}

    seen_domain = set()
    domain_to_id = {name: i for i, name in enumerate(train_domain_order)}

    # ---- W&B config -----
    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CrossEntropyLoss + SI",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "train_domains": train_domain_order,
        "si_c": si_c,
        "si_xi": si_xi
    })
    run_wandb.watch(model, criterion=criterion, log="all", log_freq=50)

    previous_domain = None
    best_model_state = None

    for idx, train_domain in enumerate(tqdm(list(train_domain_order),
                                            desc="Train Domains", total=len(train_domain_order))):
        domain_id = domain_to_id[train_domain]
        domain_epoch = 0
        if args.use_wandb:
            wandb.define_metric(f"{train_domain}/epoch")
            wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        logging.info(f"====== Pre-eval current domain {train_domain} using model from: {previous_domain} ======")

        # ===== Pre-train evaluation on current domain (F1/AUC/CM) =====
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
            auc_pre = float(m_pre["roc_auc"])
            cm_pre  = m_pre["confusion_matrix"]

            performance_plasticity[train_domain].append(f1_pre)
            roc_auc_plasticity[train_domain].append(auc_pre)
            confusion_matrix_plasticity[train_domain].append(cm_pre)

            logging.info(f"[PRE] {train_domain}: F1={f1_pre:.4f} | AUC={auc_pre:.4f}")
            run_wandb.log({
                f"{train_domain}/pretrain_f1": f1_pre,
                f"{train_domain}/pretrain_ROC_AUC": auc_pre
            })

        logging.info(f"====== Training on Domain: {train_domain} (SI) ======")

        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        for epoch in trange(num_epochs, desc=f"Epochs for {train_domain}"):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # ---- Forward (pure task loss) ----
                if args.architecture == "LSTM_Attention_adapter":
                    outputs, _ = model(X_batch, domain_id=domain_id)
                else:
                    outputs, _ = model(X_batch)
                task_loss = criterion(outputs, y_batch.long())

                # ---- Per-parameter grads of *task loss only* (for SI's ω accumulation) ----
                grads_list = torch.autograd.grad(
                    task_loss, [p for _, p in si.params.items()],
                    retain_graph=True, allow_unused=True
                )
                grads_by_name = {n: g for (n, _), g in zip(si.params.items(), grads_list)}

                # ---- Total loss = task_loss + SI quadratic penalty ----
                total_loss = task_loss + si.penalty()

                optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                si.snapshot_before_step()
                optimizer.step()

                # ---- SI online credit accumulation ----
                si.accumulate_from(grads_by_name)

                epoch_loss += float(total_loss.item())

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),
            })

            # ---- Eval on this domain (for early stopping with F1) ----
            model.eval()
            test_loss = 0.0
            all_y_true, all_y_pred, all_y_prob = [], [], []
            with torch.no_grad():
                for Xb, yb in test_domain_loader[train_domain]:
                    Xb, yb = Xb.to(device), yb.to(device)
                    if args.architecture == "LSTM_Attention_adapter":
                        out, _ = model(Xb, domain_id=domain_id)
                    else:
                        out, _ = model(Xb)
                    loss = criterion(out, yb.long())
                    _, pred = torch.max(out.data, 1)
                    all_y_true.extend(yb.cpu().numpy())
                    all_y_pred.extend(pred.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(out, dim=1)[:, 1].cpu().numpy())
                    test_loss += loss.item()
            test_loss /= max(1, len(test_domain_loader[train_domain]))

            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
                train_domain, train_domain
            )
            current_f1 = float(metrics["f1"])
            current_auc_roc = float(metrics["roc_auc"])

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} | Val Loss: {test_loss:.4f} | F1: {current_f1:.4f} | AUC: {current_auc_roc:.4f}")
            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": current_f1,
                f"{train_domain}/val_ROC_AUC": current_auc_roc
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

        # ====== SI CONSOLIDATION at TASK END ======
        si.consolidate_task_end()

        # ---- Post-train eval on current domain (append to plasticity & stability, incl. AUC & CM) ----
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            m_post = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            m_post = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)

        f1_post  = float(m_post["f1"])
        auc_post = float(m_post["roc_auc"])
        cm_post  = m_post["confusion_matrix"]

        performance_plasticity[train_domain].append(f1_post)   # post
        performance_stability[train_domain].append(f1_post)
        roc_auc_plasticity[train_domain].append(auc_post)
        roc_auc_stability[train_domain].append(auc_post)
        confusion_matrix_plasticity[train_domain].append(cm_post)
        confusion_matrix_stability[train_domain].append(cm_post)

        logging.info(f"[POST] {train_domain}: F1={f1_post:.4f} | AUC={auc_post:.4f}")

        # ---- Evaluate on all previous domains (stability across seen domains) ----
        logging.info(f"====== Stability eval on all seen domains after {train_domain} ======")
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
            auc_prev = float(m_prev["roc_auc"])
            cm_prev  = m_prev["confusion_matrix"]

            performance_stability[test_domain].append(f1_prev)
            roc_auc_stability[test_domain].append(auc_prev)
            confusion_matrix_stability[test_domain].append(cm_prev)

            logging.info(f"Stability | {test_domain}: F1={f1_prev:.4f} | AUC={auc_prev:.4f}")

        print(f"====== Finished Training on Domain: {train_domain} ======")

    # ===== Final Metrics =====
    logging.info("====== Final Metrics (F1) ======")
    bwt_values_f1, bwt_dict_f1, bwt_values_dict_f1 = result_utils.compute_BWT(performance_stability, train_domain_order)
    plasticity_values_f1, plasticity_dict_f1 = result_utils.compute_plasticity(performance_plasticity, train_domain_order)

    logging.info(f"BWT (F1): {bwt_values_f1}")
    logging.info(f"BWT per domain (F1): {bwt_dict_f1}")
    logging.info(f"Plasticity (F1): {plasticity_values_f1}")

    logging.info("====== Final Metrics (ROC-AUC) ======")
    bwt_values_auc, bwt_dict_auc, bwt_values_dict_auc = result_utils.compute_BWT(roc_auc_stability, train_domain_order)
    plasticity_values_auc, plasticity_dict_auc = result_utils.compute_plasticity(roc_auc_plasticity, train_domain_order)

    logging.info(f"BWT (AUC): {bwt_values_auc}")
    logging.info(f"BWT per domain (AUC): {bwt_dict_auc}")
    logging.info(f"Plasticity (AUC): {plasticity_values_auc}")

    # ===== Prepare JSON (ensure python-native types) =====
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
        "plasticity_dict_auc": plasticity_dict_auc,
    }

    # Convert to plain Python (handle numpy, tensors, etc.)
    results_to_save = _json_safe(results_to_save)

    # Save
    out_name = f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_{args.scenario}.json"
    save_results_as_json(results_to_save, filename=out_name)
    logging.info(f"Final training complete. Results saved to {out_name}")

    # --- W&B summaries (NO FWT) ---
    try:
        run_wandb.summary["BWT_F1/list"] = bwt_values_f1
        run_wandb.summary["Plasticity_F1/list"] = plasticity_values_f1
        run_wandb.summary["BWT_AUC/list"] = bwt_values_auc
        run_wandb.summary["Plasticity_AUC/list"] = plasticity_values_auc
    except Exception as e:
        logging.warning(f"W&B summary logging skipped: {e}")
