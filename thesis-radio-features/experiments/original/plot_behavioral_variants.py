from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Output directory
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "results" / "plots" / "methodology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Main figure
# ============================================================

def main() -> None:
    simulation_start = 0
    attack_start = 450
    simulation_end = 900

    time = np.linspace(
        simulation_start,
        simulation_end,
        901,
    )

    attack_phase = time >= attack_start

    # --------------------------------------------------------
    # Base variant:
    # attack is continuously active after the attack starts.
    # --------------------------------------------------------

    base = np.zeros_like(time, dtype=float)
    base[attack_phase] = 1.0

    # --------------------------------------------------------
    # On-off variant:
    # attack activity alternates after the attack starts.
    #
    # The intervals shown here are schematic and do not claim
    # to reproduce an exact attack-specific schedule.
    # --------------------------------------------------------

    on_off = np.zeros_like(time, dtype=float)

    attack_elapsed = (
        time[attack_phase] - attack_start
    )

    schematic_interval = 60

    on_off[attack_phase] = (
        (
            attack_elapsed
            // schematic_interval
        )
        % 2
        == 0
    ).astype(float)

    # --------------------------------------------------------
    # Gradual-change variant:
    # normalized attack intensity gradually increases after
    # the attack starts.
    #
    # This does not represent a specific raw attack parameter.
    # --------------------------------------------------------

    gradual = np.zeros_like(time, dtype=float)

    gradual[attack_phase] = (
        time[attack_phase] - attack_start
    ) / (
        simulation_end - attack_start
    )

    variants = [
        {
            "title": "Base variant",
            "values": base,
            "ylabel": "Attack\nstate",
            "yticks": [0, 1],
            "yticklabels": [
                "Inactive",
                "Active",
            ],
        },
        {
            "title": "On-off variant",
            "values": on_off,
            "ylabel": "Attack\nstate",
            "yticks": [0, 1],
            "yticklabels": [
                "Temporarily off",
                "Active",
            ],
        },
        {
            "title": "Gradual-change variant",
            "values": gradual,
            "ylabel": (
                "Normalized\n"
                "attack intensity"
            ),
            "yticks": [0, 0.5, 1],
            "yticklabels": [
                "No deviation",
                "Intermediate",
                "Maximum",
            ],
        },
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.5, 7.2),
        sharex=True,
    )

    for ax, variant in zip(
        axes,
        variants,
    ):
        ax.plot(
            time,
            variant["values"],
            linewidth=2.2,
        )

        ax.axvline(
            attack_start,
            linestyle="--",
            linewidth=1.3,
        )

        ax.set_title(
            variant["title"],
            fontsize=11,
        )

        ax.set_ylabel(
            variant["ylabel"],
            rotation=0,
            labelpad=48,
            va="center",
        )

        ax.set_yticks(
            variant["yticks"]
        )

        ax.set_yticklabels(
            variant["yticklabels"]
        )

        ax.set_ylim(
            -0.08,
            1.15,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    # Phase labels
    axes[0].text(
        attack_start / 2,
        1.07,
        "Benign phase",
        ha="center",
        va="center",
        fontsize=9,
    )

    axes[0].text(
        (
            attack_start
            + simulation_end
        ) / 2,
        1.07,
        "Attack phase",
        ha="center",
        va="center",
        fontsize=9,
    )

    # Clarification for on-off behavior
    axes[1].text(
        675,
        1.07,
        (
            "Attack remains within the "
            "labeled attack phase"
        ),
        ha="center",
        va="center",
        fontsize=8.5,
    )

    axes[2].set_xlabel(
        "Simulation time (minutes)"
    )

    axes[2].set_xticks(
        [
            simulation_start,
            attack_start,
            simulation_end,
        ]
    )

    axes[2].set_xticklabels(
        [
            "0\nSimulation start",
            "450\nAttack start",
            "900\nSimulation end",
        ]
    )

    fig.suptitle(
        (
            "Behavioral Variants Used "
            "in the Experimental Domains"
        ),
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    png_path = (
        OUTPUT_DIR
        / "behavioral_variants.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / "behavioral_variants.pdf"
    )

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