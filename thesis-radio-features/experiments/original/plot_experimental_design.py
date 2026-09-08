from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "plots" / "methodology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    items,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height - 0.42,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    item_text = "\n".join(items)

    ax.text(
        x + width / 2,
        y + height / 2 - 0.18,
        item_text,
        ha="center",
        va="center",
        fontsize=9.5,
        linespacing=1.45,
    )


def add_arrow(
    ax,
    start_x,
    start_y,
    end_x,
    end_y,
):
    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color="black",
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    box_y = 2.65
    box_width = 2.55
    box_height = 2.35

    attack_x = 0.35
    node_x = 3.35
    variant_x = 6.35

    add_box(
        ax,
        attack_x,
        box_y,
        box_width,
        box_height,
        "Attack type",
        [
            "Blackhole",
            "DIS Flooding",
            "Local Repair",
            "Worst Parent",
        ],
    )

    add_box(
        ax,
        node_x,
        box_y,
        box_width,
        box_height,
        "Network size",
        [
            "5 nodes",
            "10 nodes",
            "15 nodes",
            "20 nodes",
        ],
    )

    add_box(
        ax,
        variant_x,
        box_y,
        box_width,
        box_height,
        "Behavioral variant",
        [
            "Base",
            "On-off",
            "Gradual change",
        ],
    )

    # Multiplication symbols between factors
    ax.text(
        3.12,
        box_y + box_height / 2,
        r"$\times$",
        ha="center",
        va="center",
        fontsize=22,
    )

    ax.text(
        6.12,
        box_y + box_height / 2,
        r"$\times$",
        ha="center",
        va="center",
        fontsize=22,
    )

    # Summary box
    summary_x = 9.45
    summary_y = 3.05
    summary_width = 2.15
    summary_height = 1.55

    summary_box = FancyBboxPatch(
        (summary_x, summary_y),
        summary_width,
        summary_height,
        boxstyle="round,pad=0.03",
        linewidth=1.7,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(summary_box)

    ax.text(
        summary_x + summary_width / 2,
        summary_y + summary_height * 0.68,
        "48 domains",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        summary_x + summary_width / 2,
        summary_y + summary_height * 0.31,
        r"$4 \times 4 \times 3$",
        ha="center",
        va="center",
        fontsize=11,
    )

    add_arrow(
        ax,
        variant_x + box_width,
        box_y + box_height / 2,
        summary_x,
        summary_y + summary_height / 2,
    )

    # Runs and total dataset size
    runs_y = 0.65

    runs_box = FancyBboxPatch(
        (2.20, runs_y),
        3.10,
        1.05,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(runs_box)

    ax.text(
        3.75,
        runs_y + 0.69,
        "20 independent runs per domain",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    ax.text(
        3.75,
        runs_y + 0.30,
        "16 training runs + 4 test runs",
        ha="center",
        va="center",
        fontsize=9.5,
    )

    total_box = FancyBboxPatch(
        (7.00, runs_y),
        2.80,
        1.05,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(total_box)

    ax.text(
        8.40,
        runs_y + 0.69,
        "960 simulation runs",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        8.40,
        runs_y + 0.30,
        r"$48 \times 20$",
        ha="center",
        va="center",
        fontsize=10,
    )

    add_arrow(
        ax,
        5.30,
        runs_y + 0.525,
        7.00,
        runs_y + 0.525,
    )

    add_arrow(
        ax,
        summary_x + summary_width / 2,
        summary_y,
        8.40,
        runs_y + 1.05,
    )

    ax.set_title(
        "Experimental Domain Design",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "experimental_domain_design.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated:\n{output_path}")


if __name__ == "__main__":
    main()