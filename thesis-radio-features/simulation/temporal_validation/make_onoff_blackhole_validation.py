#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ATTACK = "blackhole"
NODE_COUNT = 15

STARTS = (300, 450, 600)
RUNS = range(1, 11)

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
            commands.text = (
                "make -j$(CPUS) udp-client-validation.cooja TARGET=cooja"
            )
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
        raise RuntimeError(
            "Expected exactly one non-empty Cooja script"
        )

    return scripts[0]


def prefix_through_attacker(text: str) -> str:
    marker = "var attacker = selectAttacker();"
    position = text.find(marker)

    if position < 0:
        raise RuntimeError(
            "Could not locate attacker-selection marker"
        )

    return text[: position + len(marker)] + "\n\n"


def make_attack_tail(start_minutes: int) -> str:
    # Same 5-second correction already verified in the controlled runs.
    start_ms = start_minutes * MS_PER_MINUTE

    remaining_ms = (
        TOTAL_MINUTES - start_minutes
    ) * MS_PER_MINUTE

    return f'''
/* Wait until controlled attack-start time */
GENERATE_MSG({start_ms}, "attack_start");
YIELD_THEN_WAIT_UNTIL(msg.equals("attack_start"));

log.log(
  "Blackhole attack ON from "
  + attacker.getID()
  + "!\\n"
);

sim.getEventCentral().logEvent(
  "attack",
  "blackhole:" + attacker.getID()
);

setInt16(
  attacker,
  'network_attacks_udp_drop_rate',
  100
);

setBool(
  attacker,
  'network_attacks_udp_drop_fwd',
  true
);

/*
 * Preserve original Blackhole oo behaviour:
 * random ON and OFF durations between 1 and 60 seconds.
 */
var elapsed = 0;

while (elapsed < {remaining_ms}) {{

  var onRandom = new java.util.Random();
  var offRandom = new java.util.Random();

  var onTime =
    (onRandom.nextInt(60) + 1) * 1000;

  var offTime =
    (offRandom.nextInt(60) + 1) * 1000;

  /* Do not run beyond the 900-minute endpoint */
  if (elapsed + onTime > {remaining_ms}) {{
    onTime = {remaining_ms} - elapsed;
  }}

  if (onTime > 0) {{

    setInt16(
      attacker,
      'network_attacks_udp_drop_rate',
      100
    );

    setBool(
      attacker,
      'network_attacks_udp_drop_fwd',
      true
    );

    GENERATE_MSG(
      onTime,
      "blackhole_on_done"
    );

    while (true) {{
      YIELD();

      if (msg.equals("blackhole_on_done")) {{
        break;
      }}
    }}

    elapsed += onTime;
  }}

  if (elapsed >= {remaining_ms}) {{
    break;
  }}

  log.log(
    "Blackhole attack OFF from "
    + attacker.getID()
    + "!\\n"
  );

  sim.getEventCentral().logEvent(
    "stop-attack",
    "blackhole:" + attacker.getID()
  );

  setBool(
    attacker,
    'network_attacks_udp_drop_fwd',
    false
  );

  if (elapsed + offTime > {remaining_ms}) {{
    offTime = {remaining_ms} - elapsed;
  }}

  if (offTime > 0) {{

    GENERATE_MSG(
      offTime,
      "blackhole_off_done"
    );

    while (true) {{
      YIELD();

      if (msg.equals("blackhole_off_done")) {{
        break;
      }}
    }}

    elapsed += offTime;
  }}

  if (elapsed >= {remaining_ms}) {{
    break;
  }}

  log.log(
    "Blackhole attack ON from "
    + attacker.getID()
    + "!\\n"
  );

  setInt16(
    attacker,
    'network_attacks_udp_drop_rate',
    100
  );

  setBool(
    attacker,
    'network_attacks_udp_drop_fwd',
    true
  );
}}

success = true;
log.testOK();
'''


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
        (
            "Controlled on-off validation: "
            f"{ATTACK}, 15 nodes, "
            f"start {start} min, run {run}"
        ),
    )

    script = get_script(root)

    script.text = (
        prefix_through_attacker(script.text)
        + make_attack_tail(start)
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print(
        f"Generated {count} controlled "
        "on-off Blackhole scenarios."
    )


if __name__ == "__main__":
    main()
