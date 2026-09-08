from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "plots" / "methodology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_SETTINGS = [
    {
        "name": "All features",
        "parts": [
            ("Baseline routing features", 14),
            ("RSSI", 2),
            ("TX/RX", 4),
        ],
        "total": 20,
    },
    {
        "name": "RSSI-only",
        "parts": [
            ("RSSI", 2),
        ],
        "total": 2,
    },
    {
        "name": "Baseline + TX/RX",
        "parts": [
            ("Baseline routing features", 14),
            ("TX/RX", 4),
        ],
        "total": 18,
    },
    {
        "name": "Baseline",
        "parts": [
            ("Baseline routing features", 14),
        ],
        "total": 14,
    },
    {
        "name": "TX/RX-only",
        "parts": [
            ("TX/RX", 4),
        ],
        "total": 4,
    },
]


PART_STYLES = {
    "Baseline routing features": {
        "facecolor": "#4c78a8",
        "hatch": "//",
    },
    "RSSI": {
        "facecolor": "#f58518",
        "hatch": "..",
    },
    "TX/RX": {
        "facecolor": "#54a24b",
        "hatch": "xx",
    },
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    bar_height = 0.62
    y_positions = list(range(len(FEATURE_SETTINGS)))[::-1]

    for y, setting in zip(y_positions, FEATURE_SETTINGS):
        current_x = 0

        for part_name, dimension in setting["parts"]:
            style = PART_STYLES[part_name]

            rectangle = Rectangle(
                (current_x, y - bar_height / 2),
                dimension,
                bar_height,
                facecolor=style["facecolor"],
                edgecolor="black",
                linewidth=0.8,
                hatch=style["hatch"],
            )

            ax.add_patch(rectangle)

            if dimension >= 4:
                label = f"{part_name}\n({dimension})"
            else:
                label = f"{part_name} ({dimension})"

            ax.text(
                current_x + dimension / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
            )

            current_x += dimension

        ax.text(
            setting["total"] + 0.35,
            y,
            f"$d_f={setting['total']}$",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [setting["name"] for setting in FEATURE_SETTINGS]
    )

    ax.set_xlim(0, 22)
    ax.set_ylim(-0.7, len(FEATURE_SETTINGS) - 0.3)

    ax.set_xlabel("Number of features at each time step")
    ax.set_title(
        "Composition and Input Dimension of the Feature Configurations",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.set_xticks([0, 2, 4, 10, 14, 18, 20])
    ax.grid(
        axis="x",
        alpha=0.25,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "feature_configuration_composition.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated:\n{output_path}")


if __name__ == "__main__":
    main()