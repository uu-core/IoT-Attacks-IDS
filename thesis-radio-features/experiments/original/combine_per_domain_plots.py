from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = Path("results")
PER_DOMAIN_DIR = BASE_DIR / "plots" / "per_domain"
OUT_BASE = BASE_DIR / "plots" / "combined_per_domain"
OUT_VERSION_DIR = OUT_BASE / "versions"
OUT_NODE_DIR = OUT_BASE / "nodes"

METRIC = "f1"

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

NODE_ORDER = ["5", "10", "15", "20"]


VERSION_ORDER = ["gc", "base", "oo"]


def ensure_dirs():
    OUT_VERSION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NODE_DIR.mkdir(parents=True, exist_ok=True)


def get_image_path(domain: str):
    return PER_DOMAIN_DIR / f"{domain}_{METRIC}.png"


def draw_missing(ax, text):
    ax.axis("off")
    ax.text(
        0.5, 0.5, text,
        ha="center", va="center",
        fontsize=14, color="red"
    )


def plot_version_composites():
    """
    One figure per behavioural variant:
    Rows = attack types (4)
    Columns = node counts (4)
    16 subplots in total
    """
    for version in VERSION_ORDER:
        fig, axes = plt.subplots(
            nrows=len(ATTACK_ORDER),
            ncols=len(NODE_ORDER),
            figsize=(22, 18),
            squeeze=False
        )

        for r, attack in enumerate(ATTACK_ORDER):
            for c, node in enumerate(NODE_ORDER):
                ax = axes[r][c]
                domain = f"{attack}_{node}_{version}"
                img_path = get_image_path(domain)

                if img_path.exists():
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis("off")
                else:
                    draw_missing(ax, f"Missing:\n{domain}")

                
                if r == 0:
                    ax.set_title(f"Node {node}", fontsize=16, pad=12)

                
                if c == 0:
                    ax.annotate(
                        ATTACK_TITLES[attack],
                        xy=(-0.08, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold"
                    )

        fig.suptitle(
            f"Per-Domain {METRIC.upper()} Bar Plots - Version {version}",
            fontsize=22
        )
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

        out_png = OUT_VERSION_DIR / f"combined_version_{version}_{METRIC}.png"
        out_pdf = OUT_VERSION_DIR / f"combined_version_{version}_{METRIC}.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")


def plot_node_composites():
    """
    One figure per node count:
    Rows = attack types (4)
    Columns = behavioural variants (3)
    12 subplots in total
    """
    for node in NODE_ORDER:
        fig, axes = plt.subplots(
            nrows=len(ATTACK_ORDER),
            ncols=len(VERSION_ORDER),
            figsize=(18, 18),
            squeeze=False
        )

        for r, attack in enumerate(ATTACK_ORDER):
            for c, version in enumerate(VERSION_ORDER):
                ax = axes[r][c]
                domain = f"{attack}_{node}_{version}"
                img_path = get_image_path(domain)

                if img_path.exists():
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis("off")
                else:
                    draw_missing(ax, f"Missing:\n{domain}")

                
                if r == 0:
                    ax.set_title(f"Version {version}", fontsize=16, pad=12)

                
                if c == 0:
                    ax.annotate(
                        ATTACK_TITLES[attack],
                        xy=(-0.08, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold"
                    )

        fig.suptitle(
            f"Per-Domain {METRIC.upper()} Bar Plots - Node {node}",
            fontsize=22
        )
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

        out_png = OUT_NODE_DIR / f"combined_node_{node}_{METRIC}.png"
        out_pdf = OUT_NODE_DIR / f"combined_node_{node}_{METRIC}.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")


def main():
    ensure_dirs()
    plot_version_composites()
    plot_node_composites()
    print("Done.")


if __name__ == "__main__":
    main()