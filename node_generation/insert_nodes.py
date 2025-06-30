import xml.etree.ElementTree as ET
import json
import os
import ast  # For parsing list input safely
import sys
import random

# Mapping of attack type to point file prefix
ATTACK_PREFIXES = {
    "local_repair": "lr",
    "worst_parent": "wp",
    "blackhole": "bh",
    "dis_flooding": "df"
}

BASE_PATH = "../applications/example-attacks/simulations"
SCENARIO_PATH = "../applications/example-attacks/scenarios"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
VARIANTS = ["base", "oo", "gc"]  # Suffixes to append

def get_attack_prefix(filename):
    for key, prefix in ATTACK_PREFIXES.items():
        if filename.startswith(key):
            return prefix
    raise ValueError(f"Unknown attack prefix in filename: {filename}")

def randomize_positions(csc_file, num_nodes):
    input_filename, input_ext = os.path.splitext(os.path.basename(csc_file))
    parts = input_filename.split("-")
    if len(parts) != 2:
        raise ValueError("Invalid filename format. Expected format: Y-Z.csc")
    
    attack_type = parts[0]
    variant = parts[1]
    
    output_dir = os.path.join(SCENARIO_PATH, attack_type, str(num_nodes), variant)
    os.makedirs(output_dir, exist_ok=True)

    prefix = get_attack_prefix(input_filename)
    points_file = os.path.join(OUTPUT_PATH, f"generated_points-{prefix}-{num_nodes}.json")
    
    with open(points_file, 'r') as f:
        all_positions = json.load(f)

    for i, positions in enumerate(all_positions, start=1):
        tree = ET.parse(csc_file)
        root = tree.getroot()
        motes = root.findall(".//mote")

        # Locate the first mote with ID=1
        first_mote, parent_element = None, None
        for parent in root.iter():
            for mote in parent.findall("mote"):
                id_elem = mote.find(".//interface_config[id]")
                if id_elem is not None and id_elem.find("id") is not None and id_elem.find("id").text == "1":
                    first_mote, parent_element = mote, parent
                    break
            if first_mote is not None:
                break

        if first_mote is None:
            raise ValueError("No mote with ID 1 found in the file.")

        # Get max ID among existing motes
        last_id = max(
            (int(mote.find(".//interface_config[id]/id").text)
             for mote in motes
             if mote.find(".//interface_config[id]/id") is not None),
            default=1
        )

        index = list(parent_element).index(first_mote) + 1
        for j in range(1, num_nodes + 1):
            new_mote = ET.Element("mote")

            pos_iface = ET.Element("interface_config")
            pos_iface.text = "\n org.contikios.cooja.interfaces.Position\n "
            x_elem = ET.SubElement(pos_iface, "x")
            y_elem = ET.SubElement(pos_iface, "y")
            z_elem = ET.SubElement(pos_iface, "z")
            x_elem.text = str(positions[j][0])
            y_elem.text = str(positions[j][1])
            z_elem.text = "0.0"
            new_mote.append(pos_iface)

            id_iface = ET.Element("interface_config")
            id_iface.text = "\n org.contikios.cooja.contikimote.interfaces.ContikiMoteID\n "
            id_elem = ET.SubElement(id_iface, "id")
            id_elem.text = str(last_id + j)
            new_mote.append(id_iface)

            motetype = ET.Element("motetype_identifier")
            motetype.text = "mtype603107969"
            new_mote.append(motetype)

            parent_element.insert(index, new_mote)
            index += 1

        # Randomize attacker ID
        attacker_id = random.randint(2, num_nodes + 1)
        for script in root.findall(".//script"):
            script_text = script.text
            if script_text and "function selectAttacker()" in script_text:
                start = script_text.find("function selectAttacker()")
                end = script_text.find("return attacker;}}", start) + len("return attacker;}}")
                replacement_func = f"function selectAttacker() {{\n  return sim.getMoteWithID({attacker_id});\n}}"
                script.text = script_text[:start] + replacement_func + script_text[end+1:]

        output_file = os.path.join(output_dir, f"{input_filename}-{num_nodes}-{i}{input_ext}")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script.py \"[attack1, attack2, ...]\"")
        sys.exit(1)

    try:
        attack_types = ast.literal_eval(sys.argv[1])
        if not isinstance(attack_types, list):
            raise ValueError
    except Exception:
        print("Error: Invalid attack list. Use format like \"[worst_parent, blackhole]\"")
        sys.exit(1)

    num_nodes = [5, 10, 15, 20]

    # Expand to full filenames with variants like "-base", "-oo", "-gc"
    attack_files = [f"{attack}-{variant}" for attack in attack_types for variant in VARIANTS]

    # Base path where the .csc files are located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.join(SCRIPT_DIR, "../applications/example-attacks/simulations")

    for n in num_nodes:
        for filename in attack_files:
            csc_file_path = os.path.join(BASE_PATH, f"{filename}.csc")
            randomize_positions(csc_file_path, n)