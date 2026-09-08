import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path("results")

EXP_DIRS = {
    1: BASE_DIR / "exp_features_1",
    2: BASE_DIR / "exp_features_2",
    3: BASE_DIR / "exp_features_3",
    4: BASE_DIR / "exp_features_4",
}

EXP_LABELS = {
    1: "Exp 1\nAll features",
    2: "Exp 2\nTX/RX only",
    3: "Exp 3\nAll except RX",
    4: "Exp 4\nAll except TX",
}

SELECTED_DOMAINS = [
    "dis_flooding_5_oo",
    "dis_flooding_10_oo",
    "dis_flooding_15_oo",
    "dis_flooding_20_oo",
    "blackhole_20_oo",
]

OUT_DIR = BASE_DIR / "plots" / "selected_5_domains"
METRIC = "f1"   


def load_metric(exp_no: int, domain: str, metric: str = METRIC):
    json_path = EXP_DIRS[exp_no] / domain / "metrics.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(metric)


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


def add_value_labels(ax, bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


def plot_one_domain(domain):
    values = []
    for exp_no in [1, 2, 3, 4]:
        val = load_metric(exp_no, domain, METRIC)
        values.append(float(val) if val is not None else 0.0)

    labels = [EXP_LABELS[i] for i in [1, 2, 3, 4]]
    colors = get_bar_colors(values)

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = ax.bar(labels, values, color=colors)

    add_value_labels(ax, bars, values)

    ax.set_title(domain)
    ax.set_ylabel(METRIC.upper())
    ax.set_ylim(0, 1.08)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_png = OUT_DIR / f"{domain}_{METRIC}.png"
    out_pdf = OUT_DIR / f"{domain}_{METRIC}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain in SELECTED_DOMAINS:
        plot_one_domain(domain)

    print(f"Done. Plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()