from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "plots" / "implementation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_node(
    ax,
    x,
    y,
    label,
    node_type="normal",
):
    if node_type == "sink":
        facecolor = "#d9eaf7"
        edgecolor = "#1f4e79"
        radius = 0.34
    elif node_type == "attacker":
        facecolor = "#f4cccc"
        edgecolor = "#a61c00"
        radius = 0.30
    else:
        facecolor = "white"
        edgecolor = "black"
        radius = 0.27

    node = Circle(
        (x, y),
        radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        zorder=3,
    )

    ax.add_patch(node)

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=4,
    )


def draw_route(
    ax,
    start,
    end,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color="#555555",
        shrinkA=18,
        shrinkB=21,
        zorder=1,
    )

    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # --------------------------------------------------------
    # Node positions
    # --------------------------------------------------------

    positions = {
        "Sink": (5.0, 7.0),

        "N1": (2.5, 5.5),
        "N2": (5.0, 5.4),
        "N3": (7.5, 5.5),

        "N4": (1.2, 3.8),
        "N5": (3.0, 3.7),
        "N6": (4.3, 3.6),
        "N7": (5.8, 3.6),
        "N8": (7.0, 3.7),
        "N9": (8.8, 3.8),

        "N10": (2.0, 1.9),
        "N11": (4.0, 1.7),
        "A": (6.0, 1.7),
        "N12": (8.0, 1.9),
    }

    # --------------------------------------------------------
    # RPL parent routes directed towards the sink
    # --------------------------------------------------------

    routes = [
        ("N1", "Sink"),
        ("N2", "Sink"),
        ("N3", "Sink"),

        ("N4", "N1"),
        ("N5", "N1"),
        ("N6", "N2"),
        ("N7", "N2"),
        ("N8", "N3"),
        ("N9", "N3"),

        ("N10", "N4"),
        ("N11", "N6"),
        ("A", "N7"),
        ("N12", "N9"),
    ]

    for child, parent in routes:
        draw_route(
            ax,
            positions[child],
            positions[parent],
        )

    # --------------------------------------------------------
    # Draw nodes
    # --------------------------------------------------------

    for label, (x, y) in positions.items():
        if label == "Sink":
            node_type = "sink"
        elif label == "A":
            node_type = "attacker"
        else:
            node_type = "normal"

        draw_node(
            ax,
            x,
            y,
            label,
            node_type,
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.text(
        5.0,
        7.58,
        "RPL sink / DODAG root",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.text(
        6.0,
        1.18,
        "Attacker node",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.text(
        5.0,
        0.55,
        (
            "Arrows indicate the preferred upward routes "
            "used to forward traffic towards the sink."
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.set_title(
        "Schematic Multi-Hop RPL Network Topology",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "rpl_topology_schematic.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated:\n{output_path}")


if __name__ == "__main__":
    main()