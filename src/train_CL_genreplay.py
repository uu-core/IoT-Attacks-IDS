import os
import time
import sys
import random
import logging
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd  # only if you actually use it elsewhere
from tqdm import tqdm, trange
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from utils import save_results_as_json, _sync, _json_safe
import evaluate_model
import evaluation as evaluate
import result_utils as result_utils


class LSTMVAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, latent_dim=32, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_init  = nn.Linear(latent_dim, hidden_dim)
        self.dec_lstm  = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.out_fc    = nn.Linear(hidden_dim, feature_dim)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)     # h: (1, B, H)
        h = h[-1]                        # (B, H)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_len=None):
        """Decode latent z to a sequence. Honors target_len if provided."""
        T = self.seq_len if target_len is None else target_len
        h0 = self.dec_init(z).unsqueeze(0)           # (1,B,H)
        c0 = torch.zeros_like(h0)                    # (1,B,H)
        # Start-of-seq zeros (no teacher forcing), USE T here (fix)
        seq0 = torch.zeros(z.size(0), T, self.feature_dim, device=z.device)
        out, _ = self.dec_lstm(seq0, (h0, c0))       # (B,T,H)
        return self.out_fc(out)                      # (B,T,F)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target_len=x.size(1))
        return recon, mu, logvar


def vae_loss_beta(recon, x, mu, logvar, beta=1.0):
    recon_mse = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_mse + beta * kl


def train_vae_on_dataset(vae, device, dataset, num_epochs=5, lr=1e-3, batch_size=64,
                         beta_start=0.0, beta_end=1.0, log_prefix="VAE"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(num_epochs):
        beta = beta_start + (beta_end - beta_start) * (epoch / max(1, num_epochs-1))
        total = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            loss = vae_loss_beta(recon, xb, mu, logvar, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(loader))
        logging.info(f"{log_prefix} Epoch {epoch+1}/{num_epochs} | β={beta:.3f} | Loss: {avg:.4f}")


@torch.no_grad()
def teacher_predict_soft(teacher_model, xb, device, architecture=None, domain_id=None, T=2.0):
    teacher_model.eval()
    xb = xb.to(device)
    if architecture == "LSTM_Attention_adapter":
        logits, _ = teacher_model(xb, domain_id=domain_id)
    else:
        logits, _ = teacher_model(xb)
    return F.softmax(logits / T, dim=1)  # soft targets


def distillation_loss(student_logits, teacher_probs, T=2.0):
    # KL(student || teacher) with temperature scaling; use teacher as target
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    loss_kl = F.kl_div(log_p_s, teacher_probs, reduction="batchmean") * (T * T)
    return loss_kl


def _match_time_and_feat(x: torch.Tensor, target_T: int, target_F: int) -> torch.Tensor:
    """
    Ensure x has shape [B, target_T, target_F].
    - If time length differs: center-crop (if longer) or right-pad with zeros (if shorter).
    - If feature dim mismatches: raise (you generally shouldn't change features here).
    """
    assert x.dim() == 3, f"expected [B,T,F], got {tuple(x.shape)}"
    B, T, F = x.shape
    if F != target_F:
        raise ValueError(f"Feature dim mismatch: replay F={F} vs target F={target_F}.")
    if T == target_T:
        return x
    if T > target_T:
        start = (T - target_T) // 2
        return x[:, start:start+target_T, :]
    else:
        pad_T = target_T - T
        pad = torch.zeros(B, pad_T, F, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)


def tdim_gr_random(
    args, run_wandb, train_domain_loader, test_domain_loader, train_domain_order, device,
    model, exp_no, num_epochs=10, learning_rate=0.01, patience=3,
    vae_hidden=64, vae_latent=32, window_size=10, num_features=140,
    vae_epochs=5, vae_lr=1e-3,
    replay_samples_per_epoch=0,   # if 0 -> computed from r & real count
    replay_ratio=0.5,             # r: weight on *real* loss; (1-r) on replay loss
    use_teacher_labels=True, T=2.0
):
    """
    Domain-incremental training with Generative Replay (DGR) + richer metrics:
      - Training loop unchanged.
      - Metrics: plasticity (pre/post), AUC, confusion matrices, BWT for F1 & AUC.
      - FWT removed.
    """
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # ====== Metric containers (F1 + AUC + CMs) ======
    performance_stability   = {d: [] for d in test_domain_loader.keys()}  # F1 stability
    performance_plasticity  = {d: [] for d in test_domain_loader.keys()}  # F1 plasticity (pre+post)
    roc_auc_stability       = {d: [] for d in test_domain_loader.keys()}
    roc_auc_plasticity      = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_stability  = {d: [] for d in test_domain_loader.keys()}
    confusion_matrix_plasticity = {d: [] for d in test_domain_loader.keys()}
    domain_training_cost    = {d: [] for d in test_domain_loader.keys()}

    domain_to_id = {name: i for i, name in enumerate(train_domain_order)}

    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CE(real) + KL(replay)",
        "optimizer": "AdamW",
        "train_domains": train_domain_order,
        "algorithm": "GR",
        "replay_ratio_r": replay_ratio,
        "replay_samples_per_epoch": replay_samples_per_epoch,
        "vae_epochs": vae_epochs,
        "vae_lr": vae_lr,
        "vae_hidden": vae_hidden,
        "vae_latent": vae_latent,
        "window_size": window_size,
        "num_features": num_features,
        "distill_T": T,
        "use_teacher_labels": use_teacher_labels,
    })
    run_wandb.watch(model, criterion=criterion_ce, log="all", log_freq=50)

    best_model_state = None
    prev_solver = None    # frozen teacher (previous best solver)
    replay_vae  = None    # current VAE
    prev_vae    = None    # frozen previous VAE (for replay)

    seen_domain = set()

    for idx, train_domain in enumerate(tqdm(train_domain_order, desc="Train Domains", total=len(train_domain_order))):
        domain_id = domain_to_id[train_domain]
        if args.use_wandb:
            wandb.define_metric(f"{train_domain}/epoch")
            wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        # Build teacher from previous best
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            prev_solver = deepcopy(model).to(device)
            for p in prev_solver.parameters():
                p.requires_grad = False
            prev_solver.eval()
        else:
            prev_solver = None

        # ===== Pre-train evaluation (plasticity PRE) on incoming domain =====
        if idx != 0:
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

        logging.info(f"====== Training on Domain: {train_domain} (Generative Replay) ======")
        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        # Prepare real loader and collect current sequences once
        real_loader = train_domain_loader[train_domain]
        real_X_list = []
        for Xb, _ in real_loader:
            real_X_list.append(Xb)
        if len(real_X_list) > 0:
            X_current = torch.cat(real_X_list, dim=0)
            print(f"[DEBUG] X_current shape: {tuple(X_current.shape)}")
        else:
            X_current = torch.empty(0, window_size, num_features)
        X_current = X_current.to(device)

        # Initialize a fresh VAE for this domain (student generator)
        replay_vae = LSTMVAE(
            feature_dim=num_features, hidden_dim=vae_hidden,
            latent_dim=vae_latent, seq_len=window_size
        ).to(device)

        # ===== Train VAE on: current real (+ replay from prev_vae, if present) =====
        if prev_vae is not None and X_current.size(0) > 0:
            n_replay_for_gen = X_current.size(0)
            z_gen = torch.randn(n_replay_for_gen, prev_vae.latent_dim, device=device)

            # generate **with the target time length** to avoid mismatch
            target_T = X_current.size(1)
            target_F = X_current.size(2)
            X_replay_for_gen = prev_vae.decode(z_gen, target_len=target_T).detach()
            # final guard (crop/pad if still mismatched)
            X_replay_for_gen = _match_time_and_feat(X_replay_for_gen, target_T, target_F)

            X_gen_train = torch.cat([X_current, X_replay_for_gen], dim=0)
        else:
            X_gen_train = X_current

        vae_dataset = TensorDataset(X_gen_train.detach().cpu(), torch.zeros(len(X_gen_train)))
        train_vae_on_dataset(
            replay_vae, device, vae_dataset,
            num_epochs=vae_epochs, lr=vae_lr, batch_size=max(32, args.batch_size//2),
            beta_start=0.0, beta_end=1.0, log_prefix=f"VAE[{train_domain}]"
        )

        # Freeze trained VAE for solver replay
        replay_vae.eval()
        for p in replay_vae.parameters():
            p.requires_grad = False

        # ===== Epoch loop for solver =====
        domain_epoch = 0
        for epoch in trange(num_epochs, desc="training Epochs"):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            # Decide #synthetic samples for this epoch
            num_real = X_current.size(0)
            if replay_samples_per_epoch > 0:
                n_syn = replay_samples_per_epoch
            else:
                n_syn = int(num_real * (1 - replay_ratio) / max(replay_ratio, 1e-6))
                n_syn = max(n_syn, args.batch_size)

            # Generate synthetic sequences (solver training)
            if n_syn > 0:
                z = torch.randn(n_syn, replay_vae.latent_dim, device=device)
                # decode with the **current window_size** to keep shapes consistent
                syn_X = replay_vae.decode(z, target_len=window_size).detach()
                if use_teacher_labels and (prev_solver is not None):
                    syn_soft = teacher_predict_soft(
                        prev_solver, syn_X, device,
                        architecture=args.architecture,
                        domain_id=domain_id if args.architecture=="LSTM_Attention_adapter" else None,
                        T=T
                    )
                else:
                    num_classes = getattr(args, "num_classes", 2)
                    syn_soft = torch.full((syn_X.size(0), num_classes), 1.0/num_classes, device=device)
            else:
                syn_X, syn_soft = None, None

            syn_ptr = 0
            syn_idx = torch.tensor([], device=device)
            if syn_X is not None:
                syn_idx = torch.randperm(syn_X.size(0), device=device)
                syn_ptr = 0

            for i, (Xr, yr) in enumerate(real_loader):
                Xr = Xr.to(device); yr = yr.to(device)

                # Pick a replay chunk ~ batch size
                if syn_X is not None and syn_soft is not None:
                    # Align Xs to Xr shape if needed (guard rails)
                    if syn_X.size(1) != Xr.size(1) or syn_X.size(2) != Xr.size(2):
                        syn_X = _match_time_and_feat(syn_X, Xr.size(1), Xr.size(2))

                    take = min(args.batch_size, syn_X.size(0) - syn_ptr)
                    if take <= 0:
                        syn_idx = torch.randperm(syn_X.size(0), device=device)
                        syn_ptr = 0
                        take = min(args.batch_size, syn_X.size(0))
                    idx_take = syn_idx[syn_ptr:syn_ptr+take]
                    Xs = syn_X[idx_take]
                    Ps = syn_soft[idx_take]
                    syn_ptr += take
                else:
                    Xs, Ps = None, None

                optimizer.zero_grad()

                # Real pass (CE)
                if args.architecture == "LSTM_Attention_adapter":
                    logits_real, _ = model(Xr, domain_id=domain_id)
                else:
                    logits_real, _ = model(Xr)
                loss_real = criterion_ce(logits_real, yr.long())

                # Replay pass (KL distillation)
                if Xs is not None:
                    if args.architecture == "LSTM_Attention_adapter":
                        logits_syn, _ = model(Xs, domain_id=domain_id)
                    else:
                        logits_syn, _ = model(Xs)
                    loss_replay = distillation_loss(logits_syn, Ps, T=T)
                else:
                    loss_replay = torch.tensor(0.0, device=device)

                # Mixed loss
                loss = replay_ratio * loss_real + (1.0 - replay_ratio) * loss_replay
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] "
                         f"| Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),
            })

            # ===== Validation on this domain =====
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for Xb, yb in test_domain_loader[train_domain]:
                    Xb, yb = Xb.to(device), yb.to(device)
                    if args.architecture == "LSTM_Attention_adapter":
                        logits, _ = model(Xb, domain_id=domain_id)
                    else:
                        logits, _ = model(Xb)
                    loss_b = criterion_ce(logits, yb.long())
                    probs = F.softmax(logits, dim=1)
                    pred  = probs.argmax(dim=1)

                    all_y_true.extend(yb.cpu().numpy())
                    all_y_pred.extend(pred.cpu().numpy())
                    all_y_prob.extend(probs[:, 1].detach().cpu().numpy())
                    test_loss += loss_b.item()

            test_loss /= max(1, len(test_domain_loader[train_domain]))
            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred),
                np.array(all_y_prob), train_domain, train_domain
            )
            cur_f1 = float(metrics["f1"])
            cur_auc_roc = float(metrics.get("roc_auc", 0.0))

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} | "
                         f"Val Loss: {test_loss:.4f} | F1: {cur_f1:.4f} | AUC-ROC: {cur_auc_roc:.4f}")
            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(cur_f1),
                f"{train_domain}/val_ROC_AUC": float(cur_auc_roc),
            })

            # Early stopping on F1
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
                logging.info(f"New best F1 for {train_domain}: {best_f1:.4f}")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping for {train_domain} at epoch {epoch+1}")
                break

        # ===== end epochs =====
        _sync(device)
        domain_training_time = time.perf_counter() - t0
        logging.info(f"Training time for {train_domain}: {domain_training_time:.2f} s")
        domain_training_cost[train_domain].append(domain_training_time)

        # Restore best and save checkpoint
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}_{args.architecture}_{args.algorithm}_{args.scenario}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")

        # ===== Post-train plasticity & stability (current domain) =====
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

        # ===== Stability on previously seen domains =====
        seen_domain.add(train_domain)
        logging.info(f"====== Evaluating on all previous domains after training on {train_domain} ======")
        for td in tqdm(seen_domain, desc="Stability test"):
            if td == train_domain:
                continue
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                m_prev = evaluate_model.eval_model(args, model, test_domain_loader, td, device, domain_id=domain_to_id[td])
            else:
                m_prev = evaluate_model.eval_model(args, model, test_domain_loader, td, device, domain_id=None)

            f1_prev  = float(m_prev["f1"])
            auc_prev = float(m_prev.get("roc_auc", 0.0))
            cm_prev  = m_prev.get("confusion_matrix", None)

            performance_stability[td].append(f1_prev)
            roc_auc_stability[td].append(auc_prev)
            confusion_matrix_stability[td].append(cm_prev)

        # Move current VAE to prev_vae for next domain
        prev_vae = deepcopy(replay_vae).to(device).eval()
        for p in prev_vae.parameters():
            p.requires_grad = False

    # ===== Final Metrics (F1) =====
    logging.info("====== Final Metrics (F1) ======")
    bwt_values_f1, bwt_dict_f1, bwt_values_dict_f1 = result_utils.compute_BWT(performance_stability, train_domain_order)
    plasticity_values_f1, plasticity_dict_f1       = result_utils.compute_plasticity(performance_plasticity, train_domain_order)

    # ===== Final Metrics (AUC) =====
    logging.info("====== Final Metrics (ROC-AUC) ======")
    bwt_values_auc, bwt_dict_auc, bwt_values_dict_auc = result_utils.compute_BWT(roc_auc_stability, train_domain_order)
    plasticity_values_auc, plasticity_dict_auc        = result_utils.compute_plasticity(roc_auc_plasticity, train_domain_order)

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

    # JSON-safe conversion for numpy/tensors/etc.
    results_to_save = _json_safe(results_to_save)
    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_{args.scenario}.json")
    logging.info("Final training complete. Results saved.")

    run_wandb.summary["BWT_F1/list"]  = bwt_values_f1
    run_wandb.summary["BWT_AUC/list"] = bwt_values_auc
