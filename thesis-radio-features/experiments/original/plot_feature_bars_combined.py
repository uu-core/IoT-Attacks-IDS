import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("results")
EXP_DIRS = {
    1: BASE_DIR / "exp_features_1",
    2: BASE_DIR / "exp_features_2",
    3: BASE_DIR / "exp_features_3",
    4: BASE_DIR / "exp_features_4",
}
EXP_LABELS = {
    1: "All features",
    2: "TX/RX only",
    3: "All except RX",
    4: "All except TX",
}


ATTACK_ORDER = [
    "blackhole",       
    "dis_flooding",    
    "worst_parent",    
    "local_repair",    
]

ATTACK_TITLES = {
    "blackhole": "Blackhole",
    "dis_flooding": "DIS-Flooding",
    "worst_parent": "Worst Parent",
    "local_repair": "Local Repair",
}

OUT_DIR = BASE_DIR / "plots" / "per_attack"
METRIC = "f1"   


def load_metric(exp_no: int, domain: str, metric: str = METRIC):
    json_path = EXP_DIRS[exp_no] / domain / "metrics.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(metric)


def get_all_domains():
    domains = set()
    for exp_dir in EXP_DIRS.values():
        if not exp_dir.exists():
            continue
        for p in exp_dir.iterdir():
            if p.is_dir() and (p / "metrics.json").exists():
                domains.add(p.name)
    return sorted(domains)


def parse_domain(domain: str):
    
    # blackhole_5_base
    # dis_flooding_10_gc
    # local_repair_20_oo
    # worst_parent_15_base
    parts = domain.split("_")
    if len(parts) < 3:
        return None, None, None

    node = parts[-2]
    version = parts[-1]
    attack = "_".join(parts[:-2])
    return attack, node, version


def build_dataframe():
    rows = []
    domains = get_all_domains()

    for domain in domains:
        attack, node, version = parse_domain(domain)
        if attack is None:
            continue

        for exp_no in [1, 2, 3, 4]:
            val = load_metric(exp_no, domain, METRIC)
            if val is None:
                continue

            rows.append({
                "domain": domain,
                "attack": attack,
                "node": node,
                "version": version,
                "exp_no": exp_no,
                "exp_label": EXP_LABELS[exp_no],
                "metric": float(val),
            })

    return pd.DataFrame(rows)


def get_bar_colors(values):
    default = "C0"
    highest = "C2"
    lowest = "C3"

    vmax = max(values)
    vmin = min(values)

    if vmax == vmin:
        return [default] * len(values)

    colors = []
    for v in values:
        if v == vmax:
            colors.append(highest)
        elif v == vmin:
            colors.append(lowest)
        else:
            colors.append(default)
    return colors


def plot_combined_attack_figure(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, attack in zip(axes, ATTACK_ORDER):
        sub = df[df["attack"] == attack].copy()

        if sub.empty:
           #ax.set_title(f"{ATTACK_TITLES[attack]} (no data)", fontsize=18)
            ax.axis("off")
            continue

        
        agg = sub.groupby(["exp_no", "exp_label"], as_index=False)["metric"].mean()
        agg = agg.sort_values("exp_no")

        labels = agg["exp_label"].tolist()
        values = agg["metric"].tolist()
        colors = get_bar_colors(values)

        bars = ax.bar(labels, values, color=colors)

        
        ax.bar_label(
            bars,
            labels=[f"{v:.3f}" for v in values],
            padding=6,
            fontsize=16
        )

       #ax.set_title(ATTACK_TITLES[attack], fontsize=18)
        ax.set_ylabel(METRIC.upper(), fontsize=15)
        ax.set_ylim(0.94, 1.01)

        
        ax.tick_params(axis="x", labelrotation=20, labelsize=13)
        ax.tick_params(axis="y", labelsize=12)

   #fig.suptitle(
       #f"Average {METRIC.upper()} by Attack Type and Feature Configuration",
       #fontsize=20
   #)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = OUT_DIR / f"combined_attacks_{METRIC}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


def main():
    df = build_dataframe()
    if df.empty:
        print("No data found.")
        return
    plot_combined_attack_figure(df)


if __name__ == "__main__":
    main()