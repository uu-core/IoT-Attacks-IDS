TEMPORAL VALIDATION PATCH
=========================

This patch does not overwrite udp-client.c. It adds udp-client-validation.c and
uses it only in newly generated validation scenarios.

What is corrected
-----------------
1. Attack starts vary across 300, 450, and 600 minutes.
2. Ten all-benign simulations are included.
3. TX/RX are logged as precise raw Energest counters, not truncated whole seconds.
4. Each node's own DATA log is parsed directly, rather than attaching stale energy
   values when a packet later arrives at the sink.
5. TX/RX differences are calculated per node before cross-node mean/std aggregation.
6. RPL cumulative counters and their node-level interval differences are both saved.
7. Bins crossing the attack boundary are marked transition and excluded from the
   main labeled feature file.
8. Existing simulations, source files, and results are not overwritten.

Install
-------
Extract this archive into the existing project root so that these paths appear:

  ~/ids-WPLR/applications/example-attacks/udp-client-validation.c
  ~/ids-WPLR/temporal_validation/

Pilot run first
---------------

  cd ~/ids-WPLR
  ./temporal_validation/run_pilot.sh

The pilot runs one DIS-flooding simulation with attack start at 300 minutes,
generates interval features, and performs consistency checks.

Run the remaining simulations
-----------------------------
After the pilot passes:

  cd ~/ids-WPLR
  ./temporal_validation/run_scenarios.sh
  python3 temporal_validation/generate_features.py

The runner skips the already completed pilot unless --force is supplied.

Outputs
-------
Raw Cooja outputs:

  applications/example-attacks/validation_outputs/

Per-run files:

  node_interval_observations.csv
  features_temporal_validation_all_bins.csv
  features_temporal_validation.csv
  validation_metadata.json

Global feature-generation summary:

  temporal_validation/feature_generation_summary.csv

Useful filtered runs
--------------------

  ./temporal_validation/run_scenarios.sh --attack dis_flooding
  ./temporal_validation/run_scenarios.sh --attack local_repair --start 450
  ./temporal_validation/run_scenarios.sh --attack benign
  ./temporal_validation/run_scenarios.sh --attack dis_flooding --start 300 --run 1 --force
