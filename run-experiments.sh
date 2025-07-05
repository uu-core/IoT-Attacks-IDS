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
        [ -d "$output_dir" ] || continue  # skip if not a dir
        [ "$(basename "$output_dir")" == "$(basename "$DIR")" ] && continue  # skip the current dir if looped

        # Find the .csc file this output likely corresponds to (assumes only one .csc per output dir)
        csc_file=$(find "$DIR" -maxdepth 1 -name "*.csc" -print | grep "$(basename "$output_dir")" || true)

        if [ -z "$csc_file" ]; then
            # Try fallback: match just on name
            csc_file=$(find "$DIR" -maxdepth 1 -name "*.csc" | head -n 1)
        fi

        if [ -n "$csc_file" ]; then
            CSC_FILENAME=$(basename "$csc_file")
            OUTPUT_NAME=$(echo "$CSC_FILENAME" | awk -F '-' '{print $NF}' | sed 's/\.csc$//')

            # Remove unneeded files
            rm -f "$output_dir/radio-log.pcap" "$output_dir/radio-medium.log"

            # Build destination path
            REL_PATH="${DIR#$BASE_DIR/}"
            DEST_DIR="$OUTPUT_BASE_DIR/$REL_PATH"
            mkdir -p "$DEST_DIR"

            mv "$output_dir" "$DEST_DIR/$OUTPUT_NAME"
            echo "Moved output to $DEST_DIR/$OUTPUT_NAME"
        fi
    done
done

