import os
import json
import csv
from pathlib import Path


def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def parse_confusion_matrix(cm):
    """
    Expecting:
    [[tn, fp],
     [fn, tp]]
    """
    tn = fp = fn = tp = None
    if isinstance(cm, list) and len(cm) == 2:
        if all(isinstance(row, list) and len(row) == 2 for row in cm):
            tn, fp = cm[0]
            fn, tp = cm[1]
    return tn, fp, fn, tp


def collect_cross_test_results(base_dir):
    rows = []

    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Cross-test directory not found: {base_dir}")

    # Expected layout:
    # results/cross_test/exp{N}/{model_domain}/vs_{test_domain}.json
    for exp_dir in sorted(base_path.glob("exp*")):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name  # e.g. exp1
        model_exp = exp_name.replace("exp", "").strip()

        for model_domain_dir in sorted(exp_dir.iterdir()):
            if not model_domain_dir.is_dir():
                continue

            model_domain = model_domain_dir.name

            for json_file in sorted(model_domain_dir.glob("vs_*.json")):
                test_domain = json_file.stem.replace("vs_", "", 1)

                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Skipping unreadable file: {json_file} ({e})")
                    continue

                accuracy = safe_get(data, "accuracy")
                f1 = safe_get(data, "f1")
                precision = safe_get(data, "precision")
                recall = safe_get(data, "recall")
                auc = safe_get(data, "auc")
                cm = safe_get(data, "confusion_matrix")

                tn, fp, fn, tp = parse_confusion_matrix(cm)

                row = {
                    "model_exp": model_exp,
                    "model_domain": model_domain,
                    "test_domain": test_domain,
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "auc": auc,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "json_path": str(json_file).replace("\\", "/"),
                }
                rows.append(row)

    return rows


def save_csv(rows, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_exp",
        "model_domain",
        "test_domain",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "json_path",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to: {output_path}")


def main():
    # Run from src/, so results/cross_test is usually here
    base_dir = os.path.join("results", "cross_test")
    output_csv = os.path.join("results", "cross_test_summary.csv")

    rows = collect_cross_test_results(base_dir)
    save_csv(rows, output_csv)


if __name__ == "__main__":
    main()