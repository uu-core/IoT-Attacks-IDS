#!/bin/bash

# Base paths
BASE_DIR="applications/example-attacks/scenarios"
OUTPUT_BASE_DIR="applications/example-attacks/scenarios_output"
COOJA_PATH="./tools/cooja"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --attack) ATTACK_FILTER="$2"; shift 2 ;;
        --size) SIZE_FILTER="$2"; shift 2 ;;
        --variation) VARIATION_FILTER="$2"; shift 2 ;;
        *) echo "Usage: $0 [--attack ATTACK] [--size SIZE] [--variation VARIATION]"; exit 1 ;;
    esac
done

# Build glob pattern
DIR_PATTERN="$BASE_DIR"
[ -n "$ATTACK_FILTER" ] && DIR_PATTERN="$DIR_PATTERN/$ATTACK_FILTER" || DIR_PATTERN="$DIR_PATTERN/*"
[ -n "$SIZE_FILTER" ] && DIR_PATTERN="$DIR_PATTERN/$SIZE_FILTER" || DIR_PATTERN="$DIR_PATTERN/*"
[ -n "$VARIATION_FILTER" ] && DIR_PATTERN="$DIR_PATTERN/$VARIATION_FILTER" || DIR_PATTERN="$DIR_PATTERN/*"

# Expand all matching directories into a list
DIR_LIST=()
while IFS= read -r -d '' dir; do
    DIR_LIST+=("$dir")
done < <(find $DIR_PATTERN -type d -print0)

# Process each matching directory
for DIR in "${DIR_LIST[@]}"; do
    # Run each .csc file in the directory
    for csc_file in "$DIR"/*.csc; do
        echo "Running $csc_file"
        "$COOJA_PATH/scripts/run-cooja.py" "$csc_file"
    done

    # Move output folder(s)
    for output_dir in "$DIR"/*/; do
        [ -d "$output_dir" ] || continue

        output_base=$(basename "$output_dir")

        # Skip if it's not an output folder (match on naming pattern)
        if [[ "$output_base" =~ ^[a-zA-Z0-9_-]+-[0-9]+-[0-9]+-([0-9]+)-dt-[0-9]+$ ]]; then
            run_id="${BASH_REMATCH[1]}"  # extract the number (e.g., 14)

            # Remove unneeded files
            rm -f "$output_dir/radio-log.pcap" "$output_dir/radio-medium.log"

            # Construct destination path
            REL_PATH="${DIR#$BASE_DIR/}"
            DEST_DIR="$OUTPUT_BASE_DIR/$REL_PATH"
            mkdir -p "$DEST_DIR"

            # Move and rename
            mv "$output_dir" "$DEST_DIR/$run_id"
            echo "Moved output to $DEST_DIR/$run_id"
        fi
    done
done

