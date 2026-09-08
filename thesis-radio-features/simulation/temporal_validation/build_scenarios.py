#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ATTACKS = ("dis_flooding", "local_repair")
ATTACK_START_MINUTES = (300, 450, 600)
RUNS = tuple(range(1, 11))
NODE_COUNT = 15
TOTAL_MINUTES = 900
MILLISECONDS_PER_MINUTE = 60_000
LOCAL_REPAIR_SETUP_DELAY_MS = 5_000


def project_root_from_script() -> Path:
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
        raise RuntimeError("Could not find the udp-client motetype in the CSC file")


def replace_title(root: ET.Element, title: str) -> None:
    title_element = root.find(".//simulation/title")
    if title_element is not None:
        title_element.text = title


def get_script(root: ET.Element) -> ET.Element:
    scripts = root.findall(".//script")
    if len(scripts) != 1 or scripts[0].text is None:
        raise RuntimeError(f"Expected exactly one non-empty Cooja script, found {len(scripts)}")
    return scripts[0]


def prefix_through_attacker(script_text: str) -> str:
    marker = "var attacker = selectAttacker();"
    position = script_text.find(marker)
    if position < 0:
        raise RuntimeError("Could not locate attacker-selection marker in Cooja script")
    return script_text[: position + len(marker)] + "\n\n"


def prefix_before_attacker(script_text: str) -> str:
    marker = "var attacker = selectAttacker();"
    position = script_text.find(marker)
    if position < 0:
        raise RuntimeError("Could not locate attacker-selection marker in Cooja script")
    return script_text[:position]


def build_attack_tail(attack: str, attack_start_minutes: int) -> str:
    attack_start_ms = attack_start_minutes * MILLISECONDS_PER_MINUTE
    total_ms = TOTAL_MINUTES * MILLISECONDS_PER_MINUTE
    post_attack_ms = total_ms - attack_start_ms

    if attack == "dis_flooding":
        return f'''GENERATE_MSG({attack_start_ms}, "attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("attack"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);

GENERATE_MSG({post_attack_ms}, "done");
YIELD_THEN_WAIT_UNTIL(msg.equals("done"));

success = true;
log.testOK();
'''

    if attack == "local_repair":
        remaining_to_attack_ms = attack_start_ms - LOCAL_REPAIR_SETUP_DELAY_MS
        if remaining_to_attack_ms <= 0:
            raise ValueError("Attack start must exceed the local-repair setup delay")
        return f'''GENERATE_MSG({remaining_to_attack_ms}, "attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("attack"));

log.log("Local Repair attack from " + attacker.getID() + "!\\n");
sim.getEventCentral().logEvent("attack", "localrepair:" + attacker.getID());
setBool(attacker, 'network_attacks_rpl_dio_reset', true);

GENERATE_MSG({post_attack_ms}, "done");
GENERATE_MSG(30000, "dio");
while (true) {{
  YIELD();
  if (msg.equals("dio")) {{
    setBool(attacker, 'network_attacks_local_repair_dio_send', true);
    setBool(attacker, 'network_attacks_rpl_dio_reset', true);
    setBool(attacker, 'network_attacks_rpl_dio_send', true);
    GENERATE_MSG(30000, "dio");
  }} else if (msg.equals("done")) {{
    break;
  }}
}}

success = true;
log.testOK();
'''

    raise ValueError(f"Unsupported attack: {attack}")


def build_benign_tail() -> str:
    total_ms = TOTAL_MINUTES * MILLISECONDS_PER_MINUTE
    return f'''GENERATE_MSG({total_ms}, "done");
YIELD_THEN_WAIT_UNTIL(msg.equals("done"));

success = true;
log.testOK();
'''


def write_tree(tree: ET.ElementTree, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def build_one_attack_scenario(
    source: Path,
    destination: Path,
    attack: str,
    attack_start_minutes: int,
    run: int,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    replace_client_source(root)
    replace_title(root, f"Temporal validation: {attack}, start {attack_start_minutes} min, run {run}")
    script = get_script(root)
    script.text = prefix_through_attacker(script.text) + build_attack_tail(attack, attack_start_minutes)
    write_tree(tree, destination)


def build_one_benign_scenario(source: Path, destination: Path, run: int) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    replace_client_source(root)
    replace_title(root, f"Temporal validation: all benign, run {run}")
    script = get_script(root)
    script.text = prefix_before_attacker(script.text) + build_benign_tail()
    write_tree(tree, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 70 temporal-validation Cooja scenarios")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--force", action="store_true", help="Delete and rebuild validation_scenarios")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    scenario_root = project_root / "applications" / "example-attacks" / "scenarios"
    output_root = project_root / "applications" / "example-attacks" / "validation_scenarios"
    manifest_path = project_root / "temporal_validation" / "scenario_manifest.csv"

    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    for attack in ATTACKS:
        for run in RUNS:
            source = scenario_root / attack / str(NODE_COUNT) / "base" / f"{attack}-base-{NODE_COUNT}-{run}.csc"
            if not source.exists():
                raise FileNotFoundError(source)
            for start_min in ATTACK_START_MINUTES:
                destination = (
                    output_root
                    / attack
                    / str(NODE_COUNT)
                    / f"start_{start_min}"
                    / f"{attack}-base-{NODE_COUNT}-start-{start_min}-run-{run:02d}.csc"
                )
                build_one_attack_scenario(source, destination, attack, start_min, run)
                records.append(
                    {
                        "attack": attack,
                        "node_count": NODE_COUNT,
                        "attack_start_minutes": start_min,
                        "run": run,
                        "scenario_path": destination.relative_to(project_root).as_posix(),
                        "source_scenario": source.relative_to(project_root).as_posix(),
                    }
                )

    for run in RUNS:
        source = scenario_root / "dis_flooding" / str(NODE_COUNT) / "base" / f"dis_flooding-base-{NODE_COUNT}-{run}.csc"
        destination = (
            output_root
            / "benign"
            / str(NODE_COUNT)
            / "all_benign"
            / f"benign-{NODE_COUNT}-run-{run:02d}.csc"
        )
        build_one_benign_scenario(source, destination, run)
        records.append(
            {
                "attack": "benign",
                "node_count": NODE_COUNT,
                "attack_start_minutes": "",
                "run": run,
                "scenario_path": destination.relative_to(project_root).as_posix(),
                "source_scenario": source.relative_to(project_root).as_posix(),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} scenarios")
    print(f"Scenario root: {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
