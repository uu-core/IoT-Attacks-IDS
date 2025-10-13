import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from copy import deepcopy
import logging
import warnings
warnings.filterwarnings("ignore")

from utils import save_results_as_json, _sync, _json_safe
import evaluate_model
import evaluation as evaluate
import result_utils as result_utils
import wandb
from tqdm import tqdm, trange

# ===================================
# LwF Utilities
# ===================================
def kd_loss(student_logits, teacher_logits, T=2.0):
    """
    Knowledge distillation loss (KL Divergence between soft targets).
    Returns a scalar tensor.
    """
    import torch.nn.functional as F
    s_log_prob = F.log_softmax(student_logits / T, dim=1)
    t_prob     = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T ** 2)

def make_frozen_teacher(model):
    """Return a frozen deepcopy of the model for distillation."""
    teacher = deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher

def _freeze_encoder_unfreeze_head_single_head(model):
    """
    For LSTMClassifier: freeze 'lstm' (encoder), unfreeze 'fc1'/'fc2' (head).
    Adjust if your layer names differ.
    """
    for n, p in model.named_parameters():
        if n.startswith("lstm"):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad_(True)

def _build_optimizer_param_groups(model, base_lr=1e-3, enc_lr_scale=0.5, weight_decay=0.0):
    """
    Two param groups: lower LR for encoder (lstm), base LR for head (fc1/fc2).
    """
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if n.startswith("lstm") else head_params).append(p)
    # Fallback if names differ: if either group ends up empty, just use all params
    if len(enc_params) == 0 or len(head_params) == 0:
        return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    return optim.AdamW(
        [
            {"params": enc_params, "lr": base_lr * enc_lr_scale},
            {"params": head_params, "lr": base_lr},
        ],
        lr=base_lr,
        weight_decay=weight_decay
    )

# ===================================
# Main Training Function with LwF (with rich metrics, no FWT)
# ===================================
def tdim_lwf_random(
    args, run_wandb, train_domain_loader, test_domain_loader, train_domain_order, device,
    model, exp_no, num_epochs=500, learning_rate=0.01, patience=3,
    alpha=0.5, T=2.0, warmup_epochs=3, enc_lr_scale=0.5, weight_decay=0.0
):
    """
    Continual/domain-incremental training using Learning without Forgetting (LwF) for a single-head model.
    - For each domain, create a frozen teacher = copy(model_before_training_domain).
    - Warm-up: freeze encoder (LSTM) and train head (fc1/fc2) with CE only.
    - Joint phase: train all params with CE(new labels) + alpha * KD(student, teacher).
    - On the first domain (no prior knowledge), KD term is skipped (teacher=None).

    Metrics collected (per domain):
      - Plasticity: [pre, post] F1 and ROC-AUC, plus confusion matrices.
      - Stability: post F1 and ROC-AUC (current + previous seen domains), plus confusion matrices.
      - Domain training cost: seconds per trained domain.
      - Aggregates: BWT (F1), BWT (AUC), Plasticity (F1), Plasticity (AUC).
    """
    criterion = nn.CrossEntropyLoss()

    # ===== Metrics containers (richer, like Code 1) =====
    performance_stability   = {d: [] for d in test_domain_loader.keys()}   # F1 stability
    performance_plasticity  = {d: [] for d in test_domain_loader.keys()}   # F1 plasticity (pre + post)
    roc_auc_stability       = {d: [] for d in test_domain_loader.keys()}
    roc_auc_plasticity      = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_stability  = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_plasticity = {d: [] for d in test_domain_loader.keys()}
    domain_training_cost    = {d: [] for d in test_domain_loader.keys()}

    seen_domain  = set()
    domain_to_id = {name: i for i, name in enumerate(train_domain_order)}

    # ---- W&B config -----
    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CrossEntropyLoss + LwF",
        "optimizer": "AdamW (param groups: enc {:.2f}x)".format(enc_lr_scale),
        "alpha_lwf": float(alpha),
        "temperature": float(T),
        "weight_decay": float(weight_decay),
        "warmup_epochs": int(warmup_epochs),
        "train_domains": train_domain_order
    })
    run_wandb.watch(model, criterion=criterion, log="all", log_freq=50)

    previous_domain   = None
    best_model_state  = None

    for idx, train_domain in enumerate(tqdm(train_domain_order, desc="Train Domains", total=len(train_domain_order))):
        domain_id = domain_to_id[train_domain]
        domain_epoch = 0
        if args.use_wandb:
            wandb.define_metric(f"{train_domain}/epoch")
            wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        # ===== Pre-train eval on upcoming domain (plasticity pre) =====
        logging.info(f"====== Pre-eval on {train_domain} using model from: {previous_domain} ======")
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

            run_wandb.log({
                f"{train_domain}/pretrain_f1": f1_pre,
                f"{train_domain}/pretrain_ROC_AUC": auc_pre
            })
            logging.info(f"[PRE] {train_domain}: F1={f1_pre:.4f} | AUC={auc_pre:.4f}")

        logging.info(f"====== Training on Domain: {train_domain} (LwF) ======")

        # Teacher = model BEFORE learning the new domain
        if best_model_state is not None:
            temp_model = deepcopy(model)
            temp_model.load_state_dict(best_model_state)
            teacher = make_frozen_teacher(temp_model)
        else:
            teacher = None  # First domain

        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        # -----------------------------
        # Warm-up: train head only (CE)
        # -----------------------------
        _freeze_encoder_unfreeze_head_single_head(model)
        warm_opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        model.train()
        for we in range(max(0, warmup_epochs)):
            warm_epoch_loss = 0.0
            i = -1
            for i, (Xb, yb) in enumerate(train_domain_loader[train_domain]):
                Xb, yb = Xb.to(device), yb.to(device).long()
                logits, _ = (model(Xb, domain_id=domain_id) if args.architecture == "LSTM_Attention_adapter" else model(Xb))
                ce = criterion(logits, yb)
                warm_opt.zero_grad()
                ce.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                warm_opt.step()
                warm_epoch_loss += float(ce.item())
            warm_epoch_loss /= (i + 1) if i >= 0 else 1
            run_wandb.log({f"{train_domain}/warmup_loss": float(warm_epoch_loss), f"{train_domain}/epoch": domain_epoch})
            logging.info(f"[{train_domain}] Warm-up Epoch {we+1}/{warmup_epochs} | CE: {warm_epoch_loss:.4f}")
        _sync(device)

        # ------------------------------------
        # Joint phase: unfreeze all, CE + α·KD
        # ------------------------------------
        _unfreeze_all(model)
        optimizer = _build_optimizer_param_groups(model, base_lr=learning_rate, enc_lr_scale=enc_lr_scale, weight_decay=weight_decay)

        for epoch in trange(num_epochs, desc=f"Epochs for {train_domain}"):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()

                optimizer.zero_grad()

                # Student forward
                if args.architecture == "LSTM_Attention_adapter":
                    student_logits, _ = model(X_batch, domain_id=domain_id)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits, _ = teacher(X_batch, domain_id=domain_id)
                    else:
                        teacher_logits = None
                else:
                    student_logits, _ = model(X_batch)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits, _ = teacher(X_batch)
                    else:
                        teacher_logits = None

                # CE loss on current labels
                ce = criterion(student_logits, y_batch)

                # KD loss against frozen teacher
                if teacher_logits is not None:
                    kd = kd_loss(student_logits, teacher_logits, T=T)
                    loss = ce + alpha * kd
                else:
                    loss = ce

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

            # ---- Eval on this domain ----
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for Xb, yb in test_domain_loader[train_domain]:
                    Xb, yb = Xb.to(device), yb.to(device).long()
                    if args.architecture == "LSTM_Attention_adapter":
                        outputs, _ = model(Xb, domain_id=domain_id)
                    else:
                        outputs, _ = model(Xb)
                    loss_val = criterion(outputs, yb)
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_true.extend(yb.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    # assumes binary classification -> prob of class 1
                    all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    test_loss += float(loss_val.item())
            test_loss /= max(1, len(test_domain_loader[train_domain]))

            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob), train_domain, train_domain
            )
            current_f1 = float(metrics["f1"])
            current_auc_roc = float(metrics["roc_auc"])

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} | Val Loss: {test_loss:.4f} | F1: {current_f1:.4f} | AUC-ROC: {current_auc_roc:.4f}")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(current_f1),
                f"{train_domain}/val_ROC_AUC": float(current_auc_roc),
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

        # ---- Post-train eval on trained domain (append to plasticity & stability, incl. AUC & CM) ----
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            m_post = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            m_post = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)

        f1_post = float(m_post["f1"])
        auc_post = float(m_post["roc_auc"])
        cm_post  = m_post["confusion_matrix"]

        performance_plasticity[train_domain].append(f1_post)   # post
        performance_stability[train_domain].append(f1_post)
        roc_auc_plasticity[train_domain].append(auc_post)
        roc_auc_stability[train_domain].append(auc_post)
        confusion_matrix_plasticity[train_domain].append(cm_post)
        confusion_matrix_stability[train_domain].append(cm_post)

        logging.info(f"[POST] {train_domain}: F1={f1_post:.4f} | AUC={auc_post:.4f}")

        # ---- Stability on all seen domains ----
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
            auc_prev = float(m_prev["roc_auc"])
            cm_prev  = m_prev["confusion_matrix"]

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

    # ===== Final Metrics (ROC-AUC) =====
    logging.info("====== Final Metrics (ROC-AUC) ======")
    bwt_values_auc, bwt_dict_auc, bwt_values_dict_auc = result_utils.compute_BWT(roc_auc_stability, train_domain_order)
    plasticity_values_auc, plasticity_dict_auc        = result_utils.compute_plasticity(roc_auc_plasticity, train_domain_order)
    logging.info(f"BWT (AUC): {bwt_values_auc}")
    logging.info(f"BWT per domain (AUC): {bwt_dict_auc}")
    logging.info(f"Plasticity (AUC): {plasticity_values_auc}")

    # ===== Prepare JSON (json-safe) =====
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

        # LwF settings (for reproducibility)
        "lwf_settings": {
            "alpha": float(alpha),
            "temperature": float(T),
            "weight_decay": float(weight_decay),
            "enc_lr_scale": float(enc_lr_scale),
            "warmup_epochs": int(warmup_epochs),
            "num_epochs": int(num_epochs),
            "patience": int(patience),
            "learning_rate": float(learning_rate),
        }
    }

    results_to_save = _json_safe(results_to_save)
    out_name = f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_{args.scenario}_alpha_{alpha}_T_{T}.json"
    save_results_as_json(results_to_save, filename=out_name)
    logging.info(f"Final training complete. Results saved to {out_name}")

    # Optional: log aggregates to W&B summary
    run_wandb.summary["BWT_F1/list"] = bwt_values_f1
    run_wandb.summary["Plasticity_F1/list"] = plasticity_values_f1
    run_wandb.summary["BWT_AUC/list"] = bwt_values_auc
    run_wandb.summary["Plasticity_AUC/list"] = plasticity_values_auc
