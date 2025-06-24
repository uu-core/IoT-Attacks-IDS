#!/bin/bash

CURDIR=$(pwd)
SCRIPT_PATH=${0%/*}

if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ] && [ "$SCRIPT_PATH" != "." ]; then
    COOJA_PATH=$SCRIPT_PATH/tools/cooja
else
    COOJA_PATH=./tools/cooja
fi

BASE_DIR="applications/example-attacks/scenarios"
OUTPUT_BASE_DIR="applications/example-attacks/scenarios_output"

# Defaults
ATTACK_FILTER=""
SIZE_FILTER=""
VARIATION_FILTER=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --attack)
            ATTACK_FILTER="$2"
            shift 2
            ;;
        --size)
            SIZE_FILTER="$2"
            shift 2
            ;;
        --variation)
            VARIATION_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--attack ATTACK_NAME] [--size SIZE] [--variation VARIATION_NAME]"
            exit 1
            ;;
    esac
done

# Construct search path
SEARCH_PATH="$BASE_DIR"
[ -n "$ATTACK_FILTER" ] && SEARCH_PATH="$SEARCH_PATH/$ATTACK_FILTER"
[ -n "$SIZE_FILTER" ] && SEARCH_PATH="$SEARCH_PATH/$SIZE_FILTER"
[ -n "$VARIATION_FILTER" ] && SEARCH_PATH="$SEARCH_PATH/$VARIATION_FILTER"

# Validate and run .csc files
if [ ! -d "$SEARCH_PATH" ]; then
    echo "Error: Path does not exist: $SEARCH_PATH"
    exit 1
fi

find "$SEARCH_PATH" -type f -name "*.csc" | while read -r csc_file; do
    echo "Running $csc_file..."
    "$COOJA_PATH/scripts/run-cooja.py" "$csc_file"

    CSC_DIR=$(dirname "$csc_file")
    OUTPUT_FOLDER=$(find "$CSC_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -d "$OUTPUT_FOLDER" ]; then
        # Extract relative path from scenarios/
        REL_PATH="${CSC_DIR#$BASE_DIR/}"

        # Get final number from .csc file name
        CSC_FILENAME=$(basename "$csc_file")
        OUTPUT_NAME=$(echo "$CSC_FILENAME" | awk -F '-' '{print $NF}' | sed 's/\.csc$//')

        # Create destination
        DEST_DIR="$OUTPUT_BASE_DIR/$REL_PATH"
        mkdir -p "$DEST_DIR"

        FINAL_OUTPUT_DIR="$DEST_DIR/$OUTPUT_NAME"
        mv "$OUTPUT_FOLDER" "$FINAL_OUTPUT_DIR"
        echo "Moved output to $FINAL_OUTPUT_DIR"

        # Remove unwanted files
        rm -f "$FINAL_OUTPUT_DIR/radio-log.pcap" "$FINAL_OUTPUT_DIR/radio-medium.log"
    else
        echo "Warning: No output directory found for $csc_file"
    fi
done
