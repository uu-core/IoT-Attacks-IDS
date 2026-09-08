import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("results") / "cross_test_summary.csv"
OUT_DIR = Path("results") / "plots" / "cross_test_heatmaps"

EXP_TITLES = {
    1: "Exp 1: All features",
    2: "Exp 2: RSSI only",
    3: "Exp 3: All except RSSI",
    4: "Exp 4: All except RSSI/RX/TX",
    5: "Exp 5: RX and TX only",
}


def plot_one_heatmap(df_exp, exp_no, out_dir):
    pivot = df_exp.pivot(index="model_domain", columns="test_domain", values="f1")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)

    ax.set_title(EXP_TITLES.get(exp_no, f"Exp {exp_no}") + " - F1 Heatmap")
    ax.set_xlabel("Test Domain")
    ax.set_ylabel("Model Domain")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("F1")

    fig.tight_layout()
    out_path = out_dir / f"exp{exp_no}_f1_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_heatmaps(df, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    axes = axes.flatten()

    for i, exp_no in enumerate([1, 2, 3, 4, 5]):
        ax = axes[i]
        df_exp = df[df["model_exp"] == exp_no].copy()

        if df_exp.empty:
            ax.set_title(f"Exp {exp_no} (no data)")
            ax.axis("off")
            continue

        pivot = df_exp.pivot(index="model_domain", columns="test_domain", values="f1")
        pivot = pivot.sort_index().sort_index(axis=1)

        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)

        ax.set_title(EXP_TITLES.get(exp_no, f"Exp {exp_no}"))
        ax.set_xlabel("Test Domain")
        ax.set_ylabel("Model Domain")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)

    
    for j in range(5, len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(right=0.92, wspace=0.25, hspace=0.25)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=None)
    sm.set_array([0, 1])
    sm.set_clim(0, 1)
    fig.colorbar(sm, cax=cbar_ax, label="F1")

    fig.suptitle("Cross-Domain F1 Heatmaps", fontsize=16)
    out_path = out_dir / "combined_f1_heatmaps.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = {"model_exp", "model_domain", "test_domain", "f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["model_exp"] = df["model_exp"].astype(int)

    for exp_no in [1, 2, 3, 4,5]:
        df_exp = df[df["model_exp"] == exp_no].copy()
        if not df_exp.empty:
            plot_one_heatmap(df_exp, exp_no, OUT_DIR)

    plot_combined_heatmaps(df, OUT_DIR)


if __name__ == "__main__":
    main()