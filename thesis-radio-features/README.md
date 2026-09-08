# Rethinking Feature Design for RPL-Based IoT Intrusion Detection

This repository contains the simulation, machine-learning, and analysis code associated with the master's thesis "Rethinking Feature Design for RPL-Based IoT Intrusion Detection: Augmenting Routing Metrics with Radio Features" at Uppsala University.

## Overview

The project investigates how radio features affect the performance and generalisation of LSTM-based intrusion detection in RPL-based IoT networks. It compares routing features, RSSI, and cumulative TX/RX measurements, and includes controlled validation experiments designed to examine temporal confounding.

## Repository structure

- `simulation/`: Contiki-NG/Cooja simulation code, attack implementations, data generation, and controlled validation.
- `experiments/original/`: Original feature-comparison and cross-domain experiment scripts.
- `experiments/temporal_diagnostics/`: Diagnostic experiments using cumulative TX/RX, differenced TX/RX, and time-related features.
- `experiments/controlled_validation/`: Reserved for the controlled validation training and analysis pipeline.
- `data/`: Dataset documentation and domain mapping.
- `results/windows/`: Preserved Windows experiment and analysis outputs.
- `legacy/`: Historical experimental implementations, to be documented separately.
- `docs/`: Reproduction instructions, provenance, and methodological notes.

## Upstream simulation dependencies

The simulation work builds on the uu-core/ids-WPLR project.

- ids-WPLR base commit: 5f015edacecd6c05474b3c77977024c8050ff1c2
- Contiki-NG commit: 27cbe8cc67ca1670d5a0fe52156bc91588bc8623
- Cooja commit: 86aaa5e5718ad149d42f61ad7fd08136acdbdcaa

The upstream project, its submodules, and their licenses must be retained and acknowledged. The local simulation modifications and validation scripts are provided as research extensions.

## Reproducibility status

This repository is currently being prepared for release. The original experiment code and historical outputs are preserved. Exact command lines, dependency versions, dataset provenance, and the mapping from published results to individual runs are being documented.

The original and controlled validation pipelines use different experimental designs and must not be treated as interchangeable.

## Citation

Citation details for the thesis and the research group's upstream work will be added before publication.

## License

License information will be added after checking the upstream repositories and the applicable reuse conditions.
