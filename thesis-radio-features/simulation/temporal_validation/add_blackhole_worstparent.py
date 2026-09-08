#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ATTACKS = ("blackhole", "worst_parent")
STARTS = (300, 450, 600)
RUNS = range(1, 11)
NODE_COUNT = 15
TOTAL_MINUTES = 900
MS_PER_MINUTE = 60_000


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


def attack_tail(attack: str, start_minutes: int) -> str:
    attack_start_ms = start_minutes * MS_PER_MINUTE
    post_attack_ms = (TOTAL_MINUTES - start_minutes) * MS_PER_MINUTE

    if attack == "blackhole":
        setup = '''log.log("Network blackhole attack from " + attacker.getID() + "!\\n");
sim.getEventCentral().logEvent("attack", "blackhole:" + attacker.getID());
setInt16(attacker, 'network_attacks_udp_drop_rate', 100);
setBool(attacker, 'network_attacks_udp_drop_fwd', true);'''
    elif attack == "worst_parent":
        setup = '''log.log("Worst parent attack from " + attacker.getID() + "!\\n");
sim.getEventCentral().logEvent("attack", "wpa:" + attacker.getID());
setBool(attacker, 'network_attacks_worst_parent', true);
setInt16(attacker, 'network_attacks_rpl_dio_fake_rank', 128);
setBool(attacker, 'network_attacks_rpl_dio_reset', true);'''
    else:
        raise ValueError(attack)

    return f'''GENERATE_MSG({attack_start_ms}, "attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("attack"));

{setup}

GENERATE_MSG({post_attack_ms}, "done");
YIELD_THEN_WAIT_UNTIL(msg.equals("done"));

success = true;
log.testOK();
'''


def write_tree(tree: ET.ElementTree, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def build_scenario(
    source: Path,
    destination: Path,
    attack: str,
    start: int,
    run: int,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    replace_client_source(root)
    replace_title(root, f"Temporal validation: {attack}, start {start} min, run {run}")
    script = get_script(root)
    script.text = prefix_through_attacker(script.text) + attack_tail(attack, start)
    write_tree(tree, destination)


def update_manifest(root: Path, new_records: list[dict[str, object]]) -> None:
    manifest = root / "temporal_validation" / "scenario_manifest.csv"
    fields = [
        "attack",
        "node_count",
        "attack_start_minutes",
        "run",
        "scenario_path",
        "source_scenario",
    ]

    existing: list[dict[str, str]] = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))

    replacement_keys = {
        (str(r["attack"]), str(r["attack_start_minutes"]), str(r["run"]))
        for r in new_records
    }

    kept = [
        row for row in existing
        if (row["attack"], row["attack_start_minutes"], row["run"])
        not in replacement_keys
    ]

    all_records = kept + [{k: str(r[k]) for k in fields} for r in new_records]
    all_records.sort(
        key=lambda r: (
            r["attack"],
            int(r["attack_start_minutes"]) if r["attack_start_minutes"] else -1,
            int(r["run"]),
        )
    )

    backup = manifest.with_suffix(".csv.before_blackhole_worstparent")
    if manifest.exists() and not backup.exists():
        shutil.copy2(manifest, backup)

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_records)


def patch_runner(root: Path) -> None:
    runner = root / "temporal_validation" / "run_batch_safe.py"
    if not runner.exists():
        raise FileNotFoundError(runner)

    text = runner.read_text(encoding="utf-8")
    backup = runner.with_suffix(".py.before_blackhole_worstparent")
    if not backup.exists():
        shutil.copy2(runner, backup)

    old_choices = 'choices=["dis_flooding", "local_repair", "benign"]'
    new_choices = 'choices=["dis_flooding", "local_repair", "blackhole", "worst_parent", "benign"]'
    if old_choices in text:
        text = text.replace(old_choices, new_choices)

    text = text.replace("    precompile(root)\n", "")
    runner.write_text(text, encoding="utf-8")


def main() -> None:
    root = project_root()
    scenario_root = root / "applications" / "example-attacks" / "scenarios"
    output_root = root / "applications" / "example-attacks" / "validation_scenarios"

    records: list[dict[str, object]] = []

    for attack in ATTACKS:
        for run in RUNS:
            source = (
                scenario_root
                / attack
                / str(NODE_COUNT)
                / "base"
                / f"{attack}-base-{NODE_COUNT}-{run}.csc"
            )
            if not source.exists():
                raise FileNotFoundError(source)

            for start in STARTS:
                destination = (
                    output_root
                    / attack
                    / str(NODE_COUNT)
                    / f"start_{start}"
                    / f"{attack}-base-{NODE_COUNT}-start-{start}-run-{run:02d}.csc"
                )
                build_scenario(source, destination, attack, start, run)
                records.append(
                    {
                        "attack": attack,
                        "node_count": NODE_COUNT,
                        "attack_start_minutes": start,
                        "run": run,
                        "scenario_path": destination.relative_to(root).as_posix(),
                        "source_scenario": source.relative_to(root).as_posix(),
                    }
                )

    update_manifest(root, records)
    patch_runner(root)

    print("Generated 60 new scenarios")
    print("Added blackhole and worst_parent to scenario_manifest.csv")
    print("Updated run_batch_safe.py attack choices")
    print("Existing 70 scenarios and outputs were not deleted")


if __name__ == "__main__":
    main()
