#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ATTACK = "dis_flooding"
NODE_COUNT = 15
STARTS = (300, 450, 600)
RUNS = range(1, 11)

TOTAL_MINUTES = 900
MS_PER_MINUTE = 60_000

# Preserve the behavioural pattern of the original "oo" DIS Flooding scenario:
# high-intensity DIS flooding for 15 min, then low-intensity for 15 min.
ON_MINUTES = 15
OFF_MINUTES = 15

ON_PERIOD_MS = 1000
OFF_PERIOD_MS = 10000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def replace_client_source(root: ET.Element) -> None:
    found = False

    for motetype in root.findall(".//motetype"):
        source = motetype.find("source")
        commands = motetype.find("commands")

        if source is None or commands is None or source.text is None:
            continue

        if source.text.strip().endswith("udp-client.c"):
            source.text = "[CONFIG_DIR]/../../../../udp-client-validation.c"
            commands.text = "make -j$(CPUS) udp-client-validation.cooja TARGET=cooja"
            found = True

    if not found:
        raise RuntimeError("Could not find udp-client motetype")


def replace_title(root: ET.Element, title: str) -> None:
    element = root.find(".//simulation/title")
    if element is not None:
        element.text = title


def get_script(root: ET.Element) -> ET.Element:
    scripts = root.findall(".//script")

    if len(scripts) != 1 or scripts[0].text is None:
        raise RuntimeError("Expected exactly one non-empty Cooja script")

    return scripts[0]


def prefix_through_attacker(text: str) -> str:
    marker = "var attacker = selectAttacker();"
    position = text.find(marker)

    if position < 0:
        raise RuntimeError("Could not locate attacker-selection marker")

    return text[: position + len(marker)] + "\n\n"


def make_attack_tail(start_minutes: int) -> str:
    start_ms = start_minutes * MS_PER_MINUTE - 5_000
    remaining_minutes = TOTAL_MINUTES - start_minutes

    tail = f'''
/* Controlled on-off validation begins at {start_minutes} minutes */
GENERATE_MSG({start_ms}, "attack_start");
YIELD_THEN_WAIT_UNTIL(msg.equals("attack_start"));

sim.getEventCentral().logEvent(
  "attack",
  "dfa:" + attacker.getID()
);

'''

    current = 0
    cycle = 1

    while current < remaining_minutes:
        on_duration = min(ON_MINUTES, remaining_minutes - current)

        if on_duration > 0:
            tail += f'''
/* Cycle {cycle}: ON / high-intensity DIS flooding */
log.log("Network DIS flooding attack ON from " + attacker.getID() + "!\\n");

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', {ON_PERIOD_MS});
setBool(attacker, 'network_attacks_rpl_dio_reset', true);

GENERATE_MSG({on_duration * MS_PER_MINUTE}, "on_done_{cycle}");
YIELD_THEN_WAIT_UNTIL(msg.equals("on_done_{cycle}"));

'''
            current += on_duration

        if current >= remaining_minutes:
            break

        off_duration = min(OFF_MINUTES, remaining_minutes - current)

        tail += f'''
/* Cycle {cycle}: OFF / low-intensity DIS flooding */
log.log("Network DIS flooding attack OFF from " + attacker.getID() + "!\\n");
sim.getEventCentral().logEvent(
  "stop-attack",
  "dfa:" + attacker.getID()
);

/*
 * Preserve the original oo behaviour:
 * DFA remains enabled, but its period changes from 1000 ms to 10000 ms.
 */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', {OFF_PERIOD_MS});

GENERATE_MSG({off_duration * MS_PER_MINUTE}, "off_done_{cycle}");
YIELD_THEN_WAIT_UNTIL(msg.equals("off_done_{cycle}"));

'''
        current += off_duration
        cycle += 1

    tail += '''
success = true;
log.testOK();
'''

    return tail


def build_scenario(
    source: Path,
    destination: Path,
    start: int,
    run: int,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()

    replace_client_source(root)

    replace_title(
        root,
        f"Controlled on-off validation: {ATTACK}, "
        f"15 nodes, start {start} min, run {run}",
    )

    script = get_script(root)

    script.text = (
        prefix_through_attacker(script.text)
        + make_attack_tail(start)
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    ET.indent(tree, space="  ")

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )


def main() -> None:
    root = project_root()

    source_root = (
        root
        / "applications"
        / "example-attacks"
        / "scenarios"
        / ATTACK
        / str(NODE_COUNT)
        / "oo"
    )

    output_root = (
        root
        / "applications"
        / "example-attacks"
        / "validation_scenarios_onoff"
        / ATTACK
        / str(NODE_COUNT)
    )

    count = 0

    for run in RUNS:
        source = (
            source_root
            / f"{ATTACK}-oo-{NODE_COUNT}-{run}.csc"
        )

        if not source.exists():
            raise FileNotFoundError(source)

        for start in STARTS:
            destination = (
                output_root
                / f"start_{start}"
                / (
                    f"{ATTACK}-oo-{NODE_COUNT}"
                    f"-start-{start}"
                    f"-run-{run:02d}.csc"
                )
            )

            build_scenario(
                source=source,
                destination=destination,
                start=start,
                run=run,
            )

            print(destination.relative_to(root))
            count += 1

    print()
    print(f"Generated {count} controlled on-off DIS Flooding scenarios.")


if __name__ == "__main__":
    main()
