from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch


SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "plots" / "background"
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
    elif node_type == "attacker":
        facecolor = "#f4cccc"
        edgecolor = "#a61c00"
    else:
        facecolor = "white"
        edgecolor = "black"

    node = Circle(
        (x, y),
        0.28,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=3,
    )

    ax.add_patch(node)

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        zorder=4,
    )


def draw_arrow(
    ax,
    start,
    end,
    linestyle="-",
    linewidth=1.4,
    color="black",
    label=None,
    label_offset=(0, 0),
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=17,
        shrinkB=17,
        zorder=1,
    )

    ax.add_patch(arrow)

    if label is not None:
        middle_x = (start[0] + end[0]) / 2
        middle_y = (start[1] + end[1]) / 2

        ax.text(
            middle_x + label_offset[0],
            middle_y + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=8,
        )


def configure_axis(ax, title):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
        pad=8,
    )


def draw_blackhole(ax):
    configure_axis(ax, "(a) Blackhole")

    draw_node(ax, 2.5, 3.55, "Sink", "sink")
    draw_node(ax, 2.5, 2.35, "A", "attacker")
    draw_node(ax, 1.25, 1.15, "N1")
    draw_node(ax, 3.75, 1.15, "N2")

    draw_arrow(ax, (2.5, 2.35), (2.5, 3.55))
    draw_arrow(ax, (1.25, 1.15), (2.5, 2.35))
    draw_arrow(ax, (3.75, 1.15), (2.5, 2.35))

    ax.text(
        3.15,
        2.92,
        "Packets dropped",
        fontsize=8.5,
        ha="left",
    )

    ax.text(
        2.5,
        2.92,
        "×",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center",
    )


def draw_dis_flooding(ax):
    configure_axis(ax, "(b) DIS Flooding")

    draw_node(ax, 2.5, 2.0, "A", "attacker")

    neighbors = [
        (2.5, 3.45, "N1"),
        (1.05, 2.35, "N2"),
        (3.95, 2.35, "N3"),
        (1.55, 0.75, "N4"),
        (3.45, 0.75, "N5"),
    ]

    for x, y, label in neighbors:
        draw_node(ax, x, y, label)

        draw_arrow(
            ax,
            (2.5, 2.0),
            (x, y),
            color="#a61c00",
            linewidth=1.3,
        )

    ax.text(
        2.5,
        1.35,
        "Repeated DIS messages",
        ha="center",
        va="center",
        fontsize=8.5,
    )


def draw_worst_parent(ax):
    configure_axis(ax, "(c) Worst Parent")

    draw_node(ax, 2.5, 3.55, "Sink", "sink")
    draw_node(ax, 1.3, 2.35, "Good")
    draw_node(ax, 3.7, 2.35, "Poor", "attacker")
    draw_node(ax, 2.5, 0.95, "Victim")

    draw_arrow(
        ax,
        (1.3, 2.35),
        (2.5, 3.55),
    )

    draw_arrow(
        ax,
        (3.7, 2.35),
        (2.5, 3.55),
        linestyle="--",
    )

    draw_arrow(
        ax,
        (2.5, 0.95),
        (3.7, 2.35),
        color="#a61c00",
        linewidth=1.8,
        label="Selected route",
        label_offset=(0.35, -0.05),
    )

    ax.plot(
        [2.5, 1.3],
        [0.95, 2.35],
        linestyle=":",
        linewidth=1.5,
        color="black",
    )

    ax.text(
        1.35,
        1.38,
        "Better parent\nnot selected",
        ha="center",
        va="center",
        fontsize=8,
    )


def draw_local_repair(ax):
    configure_axis(ax, "(d) Local Repair")

    draw_node(ax, 2.5, 3.55, "Sink", "sink")
    draw_node(ax, 2.5, 2.25, "A", "attacker")
    draw_node(ax, 1.25, 1.0, "N1")
    draw_node(ax, 3.75, 1.0, "N2")

    draw_arrow(ax, (2.5, 2.25), (2.5, 3.55))
    draw_arrow(ax, (1.25, 1.0), (2.5, 2.25))
    draw_arrow(ax, (3.75, 1.0), (2.5, 2.25))

    repair_arc = Arc(
        (2.5, 2.1),
        2.65,
        2.20,
        angle=0,
        theta1=25,
        theta2=330,
        linewidth=1.8,
        linestyle="--",
        color="#a61c00",
    )
    ax.add_patch(repair_arc)

    arrow_head = FancyArrowPatch(
        (3.70, 1.70),
        (3.84, 2.03),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.6,
        color="#a61c00",
    )
    ax.add_patch(arrow_head)

    ax.text(
        2.5,
        1.55,
        "Repeated route\nreconstruction",
        ha="center",
        va="center",
        fontsize=8.5,
    )


def main():
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
    )

    draw_blackhole(axes[0, 0])
    draw_dis_flooding(axes[0, 1])
    draw_worst_parent(axes[1, 0])
    draw_local_repair(axes[1, 1])

    fig.suptitle(
        "Schematic Effects of the Evaluated RPL-Based Attacks",
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.025,
        (
            "A denotes the attacker. The diagrams illustrate the main "
            "network effect of each attack rather than exact simulation topology."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.055, 1, 0.95)
    )

    output_path = OUTPUT_DIR / "rpl_attack_scenarios.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated:\n{output_path}")


if __name__ == "__main__":
    main()