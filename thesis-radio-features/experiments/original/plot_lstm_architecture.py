from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============================================================
# Output directory
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "plots" / "methodology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Drawing functions
# ============================================================

def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    detail,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025",
        linewidth=1.4,
        facecolor="white",
        edgecolor="black",
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.text(
        x + width / 2,
        y + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=8.5,
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
        mutation_scale=13,
        linewidth=1.3,
        color="black",
    )

    ax.add_patch(arrow)


# ============================================================
# Main figure
# ============================================================

def main():
    fig, ax = plt.subplots(figsize=(12.5, 4.5))

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    y = 1.45
    box_height = 1.35

    boxes = [
        {
            "x": 0.25,
            "width": 2.30,
            "title": "Input sequence",
            "detail": (
                r"Shape: $B \times 10 \times d_f$"
                "\n"
                "10 time steps"
            ),
        },
        {
            "x": 3.05,
            "width": 2.10,
            "title": "LSTM layer",
            "detail": (
                r"Input size: $d_f$"
                "\n"
                "Hidden size: 10"
            ),
        },
        {
            "x": 5.65,
            "width": 2.10,
            "title": "Final hidden state",
            "detail": (
                r"$h_T \in \mathbb{R}^{10}$"
                "\n"
                "Temporal representation"
            ),
        },
        {
            "x": 8.25,
            "width": 1.85,
            "title": "Linear layer",
            "detail": (
                "Input: 10"
                "\n"
                "Output: 2"
            ),
        },
        {
            "x": 10.60,
            "width": 1.65,
            "title": "Prediction",
            "detail": (
                "0: Benign"
                "\n"
                "1: Attack"
            ),
        },
    ]

    for item in boxes:
        add_box(
            ax=ax,
            x=item["x"],
            y=y,
            width=item["width"],
            height=box_height,
            title=item["title"],
            detail=item["detail"],
        )

    for current_box, next_box in zip(
        boxes[:-1],
        boxes[1:],
    ):
        add_arrow(
            ax=ax,
            start_x=current_box["x"] + current_box["width"],
            start_y=y + box_height / 2,
            end_x=next_box["x"],
            end_y=y + box_height / 2,
        )

    ax.text(
        6.25,
        3.75,
        "LSTM-Based Binary Intrusion Detection Architecture",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        6.25,
        0.73,
        (
            r"Feature dimensions: "
            r"All = 20, RSSI-only = 2, Baseline + TX/RX = 18, "
            r"Baseline = 14, TX/RX-only = 4"
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.text(
        6.25,
        0.38,
        (
            "The architecture remains fixed across experiments; "
            "only the input feature dimension changes."
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "lstm_ids_architecture.png"
    pdf_path = OUTPUT_DIR / "lstm_ids_architecture.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated:\n{png_path}")
    print(f"Generated:\n{pdf_path}")


if __name__ == "__main__":
    main()