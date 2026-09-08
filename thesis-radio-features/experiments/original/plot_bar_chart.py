import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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

OUT_DIR_DOMAIN = BASE_DIR / "plots_pretty" / "per_domain"
OUT_DIR_ATTACK = BASE_DIR / "plots_pretty" / "per_attack"

METRIC = "f1"

# If you only want four attacks, keep failing_node excluded.
EXCLUDE_ATTACKS = {"failing_node"}

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "axes.linewidth": 1.2,
})

BAR_STYLES = {
    1: {"color": "#1f77b4", "hatch": "//"},     # blue
    2: {"color": "#d62728", "hatch": "\\\\"},   # red
    3: {"color": "#2ca02c", "hatch": "xx"},     # green
    4: {"color": "#ff7f0e", "hatch": ".."},     # orange
    5: {"color": "#bdbdbd", "hatch": "oo"},     # gray
}


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
    parts = domain.split("_")

    if len(parts) < 3:
        return None, None, None

    node = parts[-2]
    version = parts[-1]
    attack = "_".join(parts[:-2])

    return attack, node, version


def ensure_dirs():
    OUT_DIR_DOMAIN.mkdir(parents=True, exist_ok=True)
    OUT_DIR_ATTACK.mkdir(parents=True, exist_ok=True)


def get_ylim(values):
    vmin = min(values)
    vmax = max(values)

    lower = max(0.0, vmin - 0.06)
    upper = min(1.03, vmax + 0.04)

    return lower, upper


def get_legend_handles():
    handles = []

    for exp_no in [1, 2, 3, 4, 5]:
        style = BAR_STYLES[exp_no]
        handles.append(
            Patch(
                facecolor=style["color"],
                edgecolor="black",
                hatch=style["hatch"],
                linewidth=1.1,
                label=EXP_LABELS[exp_no],
            )
        )

    return handles


def add_top_legend(ax):
    ax.legend(
        handles=get_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        borderaxespad=0.3,
    )


def draw_bars(ax, labels, values, ylabel):
    x = list(range(len(labels)))
    ylim_low, ylim_high = get_ylim(values)

    bars = []

    for i, (xi, val) in enumerate(zip(x, values), start=1):
        style = BAR_STYLES[i]

        bar = ax.bar(
            xi,
            val,
            width=0.62,
            color=style["color"],
            edgecolor="black",
            linewidth=1.1,
            hatch=style["hatch"],
            zorder=3,
        )

        bars.append(bar[0])

    offset = (ylim_high - ylim_low) * 0.015

    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim_low, ylim_high)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_per_domain_plots(domains):
    for domain in domains:
        attack, _, _ = parse_domain(domain)

        if attack in EXCLUDE_ATTACKS:
            continue

        values = []
        valid = True

        for exp_no in [1, 2, 3, 4, 5]:
            val = load_metric(exp_no, domain, METRIC)

            if val is None:
                valid = False
                break

            values.append(float(val))

        if not valid:
            continue

        labels = [EXP_LABELS[i] for i in [1, 2, 3, 4, 5]]

        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        draw_bars(ax, labels, values, ylabel=METRIC.upper())

        plt.tight_layout()

        out_path = OUT_DIR_DOMAIN / f"{domain}_{METRIC}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")


def make_per_attack_plots(domains):
    rows = []

    for domain in domains:
        attack, node, version = parse_domain(domain)

        if attack is None:
            continue

        if attack in EXCLUDE_ATTACKS:
            continue

        for exp_no in [1, 2, 3, 4, 5]:
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

    df = pd.DataFrame(rows)

    if df.empty:
        print("No attack data found.")
        return

    attacks = sorted(df["attack"].unique())

    for attack in attacks:
        sub = df[df["attack"] == attack].copy()

        agg = sub.groupby(["exp_no", "exp_label"], as_index=False)["metric"].mean()
        agg = agg.sort_values("exp_no")

        labels = agg["exp_label"].tolist()
        values = agg["metric"].tolist()

        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        draw_bars(ax, labels, values, ylabel=f"Average {METRIC.upper()}")

        # Only the first figure, fixed as blackhole if available, has legend.
        if attack == "blackhole":
            add_top_legend(ax)
            plt.tight_layout(rect=[0, 0, 1, 0.90])
        else:
            plt.tight_layout()

        out_path = OUT_DIR_ATTACK / f"{attack}_{METRIC}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")


def main():
    ensure_dirs()

    domains = get_all_domains()
    print(f"Found {len(domains)} domains")

    make_per_domain_plots(domains)
    make_per_attack_plots(domains)

    print(f"Per-domain plots saved to: {OUT_DIR_DOMAIN}")
    print(f"Per-attack plots saved to: {OUT_DIR_ATTACK}")


if __name__ == "__main__":
    main()