from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


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
    detail,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.4,
        facecolor="white",
        edgecolor="black",
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.63,
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


def main():
    fig, ax = plt.subplots(figsize=(12, 6.2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    box_width = 2.05
    box_height = 1.05

    # First row: left to right
    first_row = [
        (
            0.25,
            4.45,
            "Raw simulation logs",
            "Contiki-NG and Cooja",
        ),
        (
            2.65,
            4.45,
            "Feature aggregation",
            "60-second intervals\nmean and standard deviation",
        ),
        (
            5.05,
            4.45,
            "Domain definition",
            "Attack type × network size\n× behavioral variant",
        ),
        (
            7.45,
            4.45,
            "Run-level split",
            "16 training runs\n4 test runs",
        ),
        (
            9.85,
            4.45,
            "Feature selection",
            "Selected experimental\nfeature configuration",
        ),
    ]

    for x, y, title, detail in first_row:
        add_box(
            ax,
            x,
            y,
            box_width,
            box_height,
            title,
            detail,
        )

    for index in range(len(first_row) - 1):
        x1 = first_row[index][0] + box_width
        y1 = first_row[index][1] + box_height / 2

        x2 = first_row[index + 1][0]
        y2 = first_row[index + 1][1] + box_height / 2

        add_arrow(ax, x1, y1, x2, y2)

    # Arrow from first row to second row
    add_arrow(
        ax,
        10.88,
        4.45,
        10.88,
        3.20,
    )

    # Second row: right to left
    second_row = [
        (
            9.85,
            1.95,
            "Normalization",
            "Min–max values computed\nfrom training runs only",
        ),
        (
            7.45,
            1.95,
            "Sequence construction",
            "Length = 10 time steps\nStep = 1 time step",
        ),
        (
            5.05,
            1.95,
            "Sequence labeling",
            "Benign or attack based\non attack-phase transition",
        ),
        (
            2.65,
            1.95,
            "Tensor preparation",
            "Input shape:\nsequence length × features",
        ),
        (
            0.25,
            1.95,
            "LSTM input",
            "Binary intrusion\ndetection model",
        ),
    ]

    for x, y, title, detail in second_row:
        add_box(
            ax,
            x,
            y,
            box_width,
            box_height,
            title,
            detail,
        )

    for index in range(len(second_row) - 1):
        x1 = second_row[index][0]
        y1 = second_row[index][1] + box_height / 2

        x2 = second_row[index + 1][0] + box_width
        y2 = second_row[index + 1][1] + box_height / 2

        add_arrow(ax, x1, y1, x2, y2)

    ax.text(
        6,
        5.95,
        "Data Preprocessing and Sequence Construction Pipeline",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "data_preprocessing_pipeline.png"
    pdf_path = OUTPUT_DIR / "data_preprocessing_pipeline.pdf"

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