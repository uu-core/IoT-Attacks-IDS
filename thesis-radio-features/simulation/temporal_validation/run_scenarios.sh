#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$PROJECT_ROOT/temporal_validation/scenario_manifest.csv"
OUTPUT_ROOT="$PROJECT_ROOT/applications/example-attacks/validation_outputs"
COOJA_RUNNER="$PROJECT_ROOT/tools/cooja/scripts/run-cooja.py"

ATTACK_FILTER=""
START_FILTER=""
RUN_FILTER=""
FORCE=0

usage() {
  cat <<EOF
Usage: $0 [--attack dis_flooding|local_repair|benign] [--start 300|450|600] [--run 1-10] [--force]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attack) ATTACK_FILTER="$2"; shift 2 ;;
    --start) START_FILTER="$2"; shift 2 ;;
    --run) RUN_FILTER="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 1; }
[[ -x "$COOJA_RUNNER" || -f "$COOJA_RUNNER" ]] || { echo "Missing Cooja runner: $COOJA_RUNNER" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"

TOTAL=$(tail -n +2 "$MANIFEST" | wc -l | tr -d ' ')
INDEX=0

while IFS=, read -r attack node_count attack_start_minutes run scenario_path source_scenario; do
  INDEX=$((INDEX + 1))

  [[ -z "$ATTACK_FILTER" || "$attack" == "$ATTACK_FILTER" ]] || continue
  [[ -z "$RUN_FILTER" || "$run" == "$RUN_FILTER" ]] || continue
  if [[ -n "$START_FILTER" ]]; then
    [[ "$attack" != "benign" && "$attack_start_minutes" == "$START_FILTER" ]] || continue
  fi

  if [[ "$attack" == "benign" ]]; then
    DEST="$OUTPUT_ROOT/benign/all_benign/run_$(printf '%02d' "$run")"
  else
    DEST="$OUTPUT_ROOT/$attack/start_${attack_start_minutes}/run_$(printf '%02d' "$run")"
  fi

  if [[ -d "$DEST" && -f "$DEST/mote-output.log" && "$FORCE" -eq 0 ]]; then
    echo "[$INDEX/$TOTAL] SKIP existing: $DEST"
    continue
  fi

  if [[ "$FORCE" -eq 1 && -d "$DEST" ]]; then
    rm -rf "$DEST"
  fi

  CSC="$PROJECT_ROOT/$scenario_path"
  [[ -f "$CSC" ]] || { echo "Missing scenario: $CSC" >&2; exit 1; }
  SCENARIO_DIR="$(dirname "$CSC")"

  echo "[$INDEX/$TOTAL] RUN $attack start=${attack_start_minutes:-NA} run=$run"
  echo "  $CSC"

  # Remove stale direct child output folders left by an interrupted prior run.
  find "$SCENARIO_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/mote-output.log' ';' -print0 2>/dev/null |
    while IFS= read -r -d '' stale; do
      echo "  Removing stale scenario output: $stale"
      rm -rf "$stale"
    done

  (
    cd "$PROJECT_ROOT"
    COOJA_DISABLE_RADIO_TRACE=1 timeout --signal=TERM --kill-after=30s 60m xvfb-run -a "$COOJA_RUNNER" "$CSC"
  )

  mapfile -d '' GENERATED < <(
    find "$SCENARIO_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/mote-output.log' ';' -print0 2>/dev/null
  )

  if [[ "${#GENERATED[@]}" -ne 1 ]]; then
    echo "Expected one generated output directory, found ${#GENERATED[@]} in $SCENARIO_DIR" >&2
    printf '  %s\n' "${GENERATED[@]:-}"
    exit 1
  fi

  GENERATED_DIR="${GENERATED[0]}"
  rm -f "$GENERATED_DIR/radio-log.pcap" "$GENERATED_DIR/radio-medium.log"
  mkdir -p "$(dirname "$DEST")"
  mv "$GENERATED_DIR" "$DEST"

  cat > "$DEST/run_metadata.txt" <<EOF
attack=$attack
node_count=$node_count
attack_start_minutes=$attack_start_minutes
run=$run
scenario_path=$scenario_path
source_scenario=$source_scenario
EOF

  echo "  Saved: $DEST"
done < <(tail -n +2 "$MANIFEST")

echo "Finished. Outputs: $OUTPUT_ROOT"
