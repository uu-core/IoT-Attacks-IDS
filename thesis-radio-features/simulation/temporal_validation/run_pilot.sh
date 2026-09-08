#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 temporal_validation/build_scenarios.py --force
./temporal_validation/run_scenarios.sh --attack dis_flooding --start 300 --run 1 --force
python3 temporal_validation/generate_features.py \
  --run-dir applications/example-attacks/validation_outputs/dis_flooding/start_300/run_01
python3 temporal_validation/check_pilot.py
