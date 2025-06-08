import xml.etree.ElementTree as ET
import json
import os
import random

def randomize_positions(csc_file, num_nodes):
    input_filename, input_ext = os.path.splitext(os.path.basename(csc_file))
    parts = input_filename.split("-")
    if len(parts) != 2:
        raise ValueError("Invalid filename format. Expected format: Y-Z.csc")
    
    output_dir = os.path.join("../applications/example-attacks/scenarios",parts[0], str(num_nodes), parts[1])
    os.makedirs(output_dir, exist_ok=True)
    
    points_file = f"generated_points-lr-{num_nodes}.json"
    with open(points_file, 'r') as f:
        all_positions = json.load(f)
    
    for i, positions in enumerate(all_positions, start=1):
        tree = ET.parse(csc_file)
        root = tree.getroot()
        motes = root.findall(".//mote")
        
        first_mote = None
        parent_element = None
        for parent in root.iter():
            for mote in parent.findall("mote"):
                id_elem = mote.find(".//interface_config[id]")
                if id_elem is not None and id_elem.find("id") is not None and id_elem.find("id").text == "1":
                    first_mote = mote
                    parent_element = parent
                    break
            if first_mote is not None:
                break
        
        if first_mote is None:
            raise ValueError("No mote with ID 1 found in the file.")
        
        last_id = 0
        for mote in motes:
            id_elem = mote.find(".//interface_config[id]")
            if id_elem is not None and id_elem.find("id") is not None:
                last_id = max(last_id, int(id_elem.find("id").text))
        
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
        random.seed()
        attacker_id = random.randint(2, num_nodes + 1)
        # Find and update the script tag
        scripts = root.findall(".//script")
        for script in scripts:
            script_text = script.text
            if script_text and "function selectAttacker()" in script_text:
                # Find the boundaries of the selectAttacker function
                start = script_text.find("function selectAttacker()")
                end = script_text.find("return attacker;}}", start) + len("return attacker;}}")
                
                # Create the replacement function
                replacement_func = f"function selectAttacker() {{\n  return sim.getMoteWithID({attacker_id});\n}}"
                
                # Replace the entire function
                script.text = script_text[:start] + replacement_func + script_text[end+1:]
        output_file = os.path.join(output_dir, f"{input_filename}-{num_nodes}-{i}{input_ext}")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    num_nodes = [5, 10, 15, 20]
    files = ["../applications/example-attacks/simulations/local_repair-base.csc",
             "../applications/example-attacks/simulations/local_repair-oo.csc",
             "../applications/example-attacks/simulations/local_repair-gc.csc"
             #"../applications/example-attacks/simulations/worst_parent-base.csc",
             #"../applications/example-attacks/simulations/worst_parent-oo.csc",
             #"../applications/example-attacks/simulations/worst_parent-gc.csc"
             ]
    
    for n in num_nodes:
        for f in files:
            randomize_positions(f, n)