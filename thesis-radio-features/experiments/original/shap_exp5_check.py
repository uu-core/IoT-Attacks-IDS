import os
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

import utils
import models


DOMAINS = [
    "dis_flooding_15_oo",
    "dis_flooding_15_gc",
]

EXP_NO = 5
WINDOW_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 2

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 50


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WrappedModel(nn.Module):
    """
    Return class-1 logit for SHAP GradientExplainer.
    Input shape: (B, 1, input_size)
    Output shape: (B, 1)
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logits, _ = self.base_model(x)
        return logits[:, 1:2]


def build_model(model_path, input_size, device):
    base_model = models.LSTMClassifier(
        input_dim=input_size,
        hidden_dim=HIDDEN_SIZE,
        output_dim=OUTPUT_SIZE,
        num_layers=NUM_LAYERS,
        fc_hidden_dim=10,
    ).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.eval()
    return base_model


def save_prediction_debug(base_model, explain_tensor, out_dir):
    with torch.no_grad():
        logits, _ = base_model(explain_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    np.save(os.path.join(out_dir, "pred_probs.npy"), probs)

    print("  Pred prob stats:")
    print(f"    min={probs.min():.6f}, max={probs.max():.6f}, mean={probs.mean():.6f}")
    print(f"    std={probs.std():.6f}")
    print(f"    first10={np.round(probs[:10], 6)}")

    return probs


def normalize_shap_values(shap_values):
    """
    Make SHAP output 2D: (n_samples, n_features)
    Handles shapes like:
      (B, 1, F, 1)
      (B, 1, F)
      (B, F, 1)
      (B, F)
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 4:
        # e.g. (B, 1, F, 1)
        shap_values = np.squeeze(shap_values)
    if shap_values.ndim == 3:
        # e.g. (B, 1, F) or (B, F, 1)
        shap_values = np.squeeze(shap_values)

    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected shap_values shape after squeeze: {shap_values.shape}")

    return shap_values


def main():
    current_dir = os.getcwd()
    domains_path = os.path.join(os.path.dirname(current_dir), "attack_data")
    domains = utils.create_domains(domains_path)

    feature_cols = utils.EXPERIMENT_FEATURES[EXP_NO]
    if feature_cols is None:
        raise ValueError("feature_cols is None, but exp5 should have explicit features.")

    out_root = os.path.join(current_dir, "results", "shap_exp5_grad")
    os.makedirs(out_root, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    for domain_name in DOMAINS:
        print(f"\n=== SHAP GradientExplainer for {domain_name} (exp5) ===")

        if domain_name not in domains:
            raise ValueError(f"{domain_name} not found in domains")

        domain_data = domains[domain_name]

        train_loader, test_loader = utils.load_data(
            domains_path,
            domain_data,
            window_size=WINDOW_SIZE,
            batch_size=128,
            feature_cols=feature_cols,
        )

        X_test, y_test = test_loader.dataset.tensors
        total_samples = X_test.shape[0]
        needed = BACKGROUND_SIZE + EXPLAIN_SIZE

        if total_samples < needed:
            raise ValueError(
                f"Not enough test samples for {domain_name}: need at least {needed}, got {total_samples}"
            )

        input_size = X_test.shape[-1]

        background = X_test[:BACKGROUND_SIZE].to(device)
        explain_x = X_test[BACKGROUND_SIZE:BACKGROUND_SIZE + EXPLAIN_SIZE].to(device)

        model_path = os.path.join(current_dir, "saved_models", f"exp{EXP_NO}", f"{domain_name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        base_model = build_model(model_path, input_size, device)
        wrapped_model = WrappedModel(base_model).to(device)
        wrapped_model.eval()

        domain_out = os.path.join(out_root, domain_name)
        os.makedirs(domain_out, exist_ok=True)

        
        probs = save_prediction_debug(base_model, explain_x, domain_out)

        
        explain_x_np = explain_x.detach().cpu().numpy().reshape(explain_x.shape[0], -1)
        explain_y_np = y_test[BACKGROUND_SIZE:BACKGROUND_SIZE + EXPLAIN_SIZE].detach().cpu().numpy()
        np.save(os.path.join(domain_out, "samples.npy"), explain_x_np)
        np.save(os.path.join(domain_out, "labels.npy"), explain_y_np)

        
        if np.std(probs) == 0:
            print("  WARNING: model outputs are constant on explained samples. SHAP may be near-zero everywhere.")

        explainer = shap.GradientExplainer(wrapped_model, background)
        raw_shap_values = explainer.shap_values(explain_x)
        shap_values_2d = normalize_shap_values(raw_shap_values)

        flat_feature_names = []
        for t in range(WINDOW_SIZE):
            for feat in feature_cols:
                flat_feature_names.append(f"{feat}_t{t}")

        if shap_values_2d.shape[1] != len(flat_feature_names):
            raise ValueError(
                f"Feature count mismatch: shap has {shap_values_2d.shape[1]} cols, "
                f"but feature_names has {len(flat_feature_names)}"
            )

        # summary plot
        plt.figure()
        shap.summary_plot(
            shap_values_2d,
            explain_x_np,
            feature_names=flat_feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(domain_out, "summary_plot.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # bar plot
        plt.figure()
        shap.summary_plot(
            shap_values_2d,
            explain_x_np,
            feature_names=flat_feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(domain_out, "bar_plot.png"), dpi=200, bbox_inches="tight")
        plt.close()

        np.save(os.path.join(domain_out, "shap_values.npy"), shap_values_2d)

        print(f"  Saved outputs to: {domain_out}")


if __name__ == "__main__":
    main()