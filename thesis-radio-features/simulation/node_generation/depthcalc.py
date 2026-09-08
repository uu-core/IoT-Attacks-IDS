import os
import re

def extract_depth_from_log(log_file):
    with open(log_file, "r") as file:
        lines = file.readlines()

    one_third_index = len(lines) // 3
    first_third_lines = lines[:one_third_index]

    hop_counts = []

    for line in first_third_lines:
        match = re.search(r"HOPCOUNTMSG Data: \d+ from .*?, Hop Count: (\d+)", line)
        if match:
            hop_counts.append(int(match.group(1)))

    return max(hop_counts) if hop_counts else None

def process_all_mote_outputs_for_depth(base_directory):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    base_directory = os.path.join(script_dir, base_directory)  # Convert to absolute path

    if not os.path.isdir(base_directory):
        print(f"Error: The directory '{base_directory}' does not exist.")
        return

    processed_files = 0

    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)

        if os.path.isdir(folder_path):
            log_file = os.path.join(folder_path, "mote-output.log")

            if os.path.exists(log_file):
                depth = extract_depth_from_log(log_file) + 2 #hop count is the number of nodes in betweern the nodes, so we need +2
                if depth is not None:
                    depth_file = os.path.join(folder_path, "depth.txt")
                    with open(depth_file, "w") as df:
                        df.write(str(depth))
                    processed_files += 1

    print(f"Processed {processed_files} mote-output logs and saved depth.txt files.")

if __name__ == "__main__":
    root = "ADD_SCENARIOS_FOLDER"
    types = ["local_repair", "worst_parent"]
    sizes = ["5", "10", "15", "20"]
    variations = ["base", "oo", "gc"]

    for type in types:
        for size in sizes:
            for variation in variations:
                relative_path = os.path.join(root, type, size, variation)
                print(f"Processing: {relative_path}")
                process_all_mote_outputs_for_depth(relative_path)
