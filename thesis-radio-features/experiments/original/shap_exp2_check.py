import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

import utils
import models



DOMAINS = [
    "dis_flooding_15_oo",  # F1 = 0
    "dis_flooding_15_gc",  
]

EXP_NO = 2
WINDOW_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 2

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 20


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_path, input_size, device):
    model = models.LSTMClassifier(
        input_dim=input_size,
        hidden_dim=HIDDEN_SIZE,
        output_dim=OUTPUT_SIZE,
        num_layers=NUM_LAYERS,
        fc_hidden_dim=10,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_fn_factory(model, device, input_size):
    def predict_fn(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x = x.view(-1, 1, input_size)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.detach().cpu().numpy()
    return predict_fn


def main():
    current_dir = os.getcwd()
    domains_path = os.path.join(os.path.dirname(current_dir), "attack_data")
    domains = utils.create_domains(domains_path)

    feature_cols = utils.EXPERIMENT_FEATURES[EXP_NO]
    if feature_cols is None:
        raise ValueError("This script is for exp2-style selected features; feature_cols is None.")

    out_root = os.path.join(current_dir, "results", "shap_exp2")
    os.makedirs(out_root, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    for domain_name in DOMAINS:
        print(f"\n=== SHAP for {domain_name} ===")

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
        X_test_np = X_test.numpy()   # (N, 1, input_size)

        total_samples = X_test_np.shape[0]
        needed = BACKGROUND_SIZE + EXPLAIN_SIZE
        if total_samples < needed:
            raise ValueError(
                f"Not enough test samples for {domain_name}: need at least {needed}, got {total_samples}"
            )

        input_size = X_test_np.shape[-1]

        background = X_test_np[:BACKGROUND_SIZE].reshape(BACKGROUND_SIZE, -1)
        explain_x = X_test_np[BACKGROUND_SIZE:BACKGROUND_SIZE + EXPLAIN_SIZE].reshape(EXPLAIN_SIZE, -1)

        model_path = os.path.join(current_dir, "saved_models", f"exp{EXP_NO}", f"{domain_name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = build_model(model_path, input_size, device)
        predict_fn = predict_fn_factory(model, device, input_size)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explain_x, nsamples=100)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        flat_feature_names = []
        for t in range(WINDOW_SIZE):
            for feat in feature_cols:
                flat_feature_names.append(f"{feat}_t{t}")

        domain_out = os.path.join(out_root, domain_name)
        os.makedirs(domain_out, exist_ok=True)

        # summary plot
        plt.figure()
        shap.summary_plot(
            shap_values,
            explain_x,
            feature_names=flat_feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(domain_out, "summary_plot.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # bar plot
        plt.figure()
        shap.summary_plot(
            shap_values,
            explain_x,
            feature_names=flat_feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(domain_out, "bar_plot.png"), dpi=200, bbox_inches="tight")
        plt.close()

        np.save(os.path.join(domain_out, "shap_values.npy"), shap_values)
        np.save(os.path.join(domain_out, "samples.npy"), explain_x)

        print(f"Saved SHAP outputs to: {domain_out}")


if __name__ == "__main__":
    main()