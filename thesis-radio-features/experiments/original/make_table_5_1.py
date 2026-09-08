import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path("results")

EXP_DIRS = {
    1: BASE_DIR / "exp_features_1",
    2: BASE_DIR / "exp_features_2",
    3: BASE_DIR / "exp_features_3",
    4: BASE_DIR / "exp_features_4",
    5: BASE_DIR / "exp_features_5",
}

EXP_LABELS = {
    1: "All features",
    2: "RSSI-only",
    3: "Baseline + TX/RX",
    4: "Baseline",
    5: "TX/RX-only",
}

ATTACK_ORDER = [
    "blackhole",
    "dis_flooding",
    "local_repair",
    "worst_parent",
]

ATTACK_DISPLAY = {
    "blackhole": "Blackhole",
    "dis_flooding": "DIS flooding",
    "local_repair": "Local repair",
    "worst_parent": "Worst parent",
}

EXCLUDE_ATTACKS = {"failing_node"}
METRIC = "f1"


def parse_domain(domain_name: str):
    parts = domain_name.split("_")
    if len(parts) < 3:
        return None, None, None

    node = parts[-2]
    version = parts[-1]
    attack = "_".join(parts[:-2])

    return attack, node, version


def load_metric(metrics_path: Path, metric: str = METRIC):
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if metric not in data:
        raise KeyError(f"Metric '{metric}' not found in {metrics_path}")

    return float(data[metric])


rows = []

for exp_no, exp_dir in EXP_DIRS.items():
    if not exp_dir.exists():
        print(f"Warning: missing folder: {exp_dir}")
        continue

    for domain_dir in exp_dir.iterdir():
        if not domain_dir.is_dir():
            continue

        domain_name = domain_dir.name
        attack, node, version = parse_domain(domain_name)

        if attack is None:
            continue

        if attack in EXCLUDE_ATTACKS:
            continue

        if attack not in ATTACK_ORDER:
            continue

        metrics_path = domain_dir / "metrics.json"

        if not metrics_path.exists():
            print(f"Warning: missing metrics.json: {metrics_path}")
            continue

        f1 = load_metric(metrics_path, METRIC)

        rows.append({
            "exp_no": exp_no,
            "feature_setting": EXP_LABELS[exp_no],
            "domain": domain_name,
            "attack": attack,
            "node": node,
            "version": version,
            "f1": f1,
        })


df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(
        "No valid metrics found. Put this script in attack_cl_project/src/, "
        "where the results folder is located."
    )


count_table = (
    df.groupby(["attack", "exp_no"])
    .size()
    .unstack(fill_value=0)
)

print("\nDomain counts per attack and experiment:")
print(count_table)
print("\nExpected count is 12 for each attack and each experiment.\n")


summary = (
    df.groupby(["attack", "exp_no"])["f1"]
    .mean()
    .reset_index()
)

pivot = summary.pivot(index="attack", columns="exp_no", values="f1")
pivot = pivot.reindex(ATTACK_ORDER)
pivot = pivot.rename(columns=EXP_LABELS)
pivot.index = [ATTACK_DISPLAY[a] for a in pivot.index]

pivot_rounded = pivot.round(3)

print("\nAverage F1 table:")
print(pivot_rounded)

out_csv = BASE_DIR / "table_5_1_average_f1.csv"
pivot_rounded.to_csv(out_csv)
print(f"\nSaved CSV to: {out_csv}")

print("\nLaTeX table rows:\n")

for attack_name, row in pivot_rounded.iterrows():
    print(
        f"{attack_name} & "
        f"{row['All features']:.3f} & "
        f"{row['RSSI-only']:.3f} & "
        f"{row['Baseline + TX/RX']:.3f} & "
        f"{row['Baseline']:.3f} & "
        f"{row['TX/RX-only']:.3f} \\\\"
    )

print("\nFull LaTeX table:\n")

print(r"""\begin{table}[H]
\centering
\caption{Average F1-score of different feature configurations across attack scenarios.}
\label{tab:overall_f1_comparison}
\begin{tabular}{lccccc}
\toprule
Attack scenario & All features & RSSI-only & Baseline + TX/RX & Baseline & TX/RX-only \\
\midrule""")

for attack_name, row in pivot_rounded.iterrows():
    print(
        f"{attack_name} & "
        f"{row['All features']:.3f} & "
        f"{row['RSSI-only']:.3f} & "
        f"{row['Baseline + TX/RX']:.3f} & "
        f"{row['Baseline']:.3f} & "
        f"{row['TX/RX-only']:.3f} \\\\"
    )

print(r"""\bottomrule
\end{tabular}
\end{table}""")