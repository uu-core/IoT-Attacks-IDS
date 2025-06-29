#!/usr/bin/env python3
import os
import sys
import ast 
import observableToSink as obs

def process_folders(root_folder):
    """Traverse folders and apply `fun` to each subfolder inside var folders."""
    
    # Loop over each sizeX folder in root_folder
    for size_folder in os.listdir(root_folder):
        size_path = os.path.join(root_folder, size_folder)
        if not os.path.isdir(size_path):
            continue  # Skip files
        
        # Loop over each varY folder inside size_folder
        for var_folder in os.listdir(size_path):
            var_path = os.path.join(size_path, var_folder)
            if not os.path.isdir(var_path):
                continue  # Skip files
            
            # Loop over each subfolder inside varY
            for subfolder in os.listdir(var_path):
                subfolder_path = os.path.join(var_path, subfolder)
                if os.path.isdir(subfolder_path):  
                    obs.MyDataSet(dataAdd=subfolder_path, binSize=60)  # Run function on each folder

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 gen_feat.py \"[folder1, folder2, ...]\"")
        sys.exit(1)

    try:
        # parse the input string to a Python list
        folder_list = ast.literal_eval(sys.argv[1])
        if not isinstance(folder_list, list):
            raise ValueError
    except Exception:
        print("Error: Invalid folder list. Use the format: \"[folder1, folder2]\"")
        sys.exit(1)

    # Get script directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Base path relative to script
    base_path = os.path.join(SCRIPT_DIR, "../applications/example-attacks/scenarios")

    for folder_name in folder_list:
        abs_path = os.path.join(base_path, folder_name)
        process_folders(abs_path)


