# Rethinking Feature Design for RPL-Based IoT Intrusion Detection

**Master's thesis — Uppsala University**

This directory contains the thesis-specific code for *Rethinking Feature Design for RPL-Based IoT Intrusion Detection: Augmenting Routing Metrics with Radio Features*. The work investigates radio-feature selection for LSTM-based intrusion detection in RPL networks, cross-domain generalisation, temporal confounding, and controlled validation.

The code is maintained on the `yichang-radio-features-thesis` branch of [uu-core/IoT-Attacks-IDS](https://github.com/uu-core/IoT-Attacks-IDS). It extends the research group's work without replacing the upstream implementation.

## Code organisation

| Directory | Contents |
| --- | --- |
| `experiments/original/` | Original feature-comparison training, cross-domain evaluation, tables, and plotting scripts. |
| `experiments/temporal_diagnostics/` | Windows-based cumulative TX/RX, differenced TX/RX, and time-baseline diagnostics. |
| `simulation/applications/` | Modified Contiki-NG application code and simulation scenarios. |
| `simulation/csv_generation/` | CSV and feature-generation source. |
| `simulation/node_generation/` | Node-generation source. |
| `simulation/services/` | Modified attack services. |
| `simulation/temporal_validation/` | Controlled simulation, node-level feature generation, cross-start training, on-off variants, RPL ablations, early detection, and effect-size analysis. |
| `data/domain_details.xlsx` | Original-experiment domain metadata. |

The published directory is a code-only archive. Raw datasets, trained checkpoints, historical metrics, and generated figures are not included. The original and controlled datasets use different preprocessing procedures and must not be treated as interchangeable.

## Original experiments

The original pipeline compares five feature configurations: all retained features, RSSI-only, RPL+TX/RX, RPL-only, and TX/RX-only. Their exact column definitions are in `experiments/original/utils.py`.

The original data consists of aggregated CSV files organised by attack and domain. Supply the dataset separately; the domain mapping is included under `data/`. The path-adapted entry points accept explicit input and output locations.

### Training

Run from the `thesis-radio-features` directory:

```bash
python experiments/original/main.py \
  --data-dir /path/to/attack_data \
  --mapping-file data/domain_details.xlsx \
  --run-dir /path/to/new_output_directory \
  --exp_no 5 \
  --domain dis_flooding_15_base
```

The example selects the TX/RX-only configuration and one domain. Use a fresh output directory. The original trainer writes checkpoints under `saved_models/expN/`, metrics under `results/exp_features_N/`, and logs under `logs/` within the selected run directory.

### Cross-domain evaluation

The evaluator supports `single` and `sweep` modes. The checkpoint directory must contain models trained with the corresponding feature configuration.

```bash
python experiments/original/cross_test.py \
  --mode single \
  --model_exp 5 \
  --model_domain dis_flooding_15_base \
  --test_domain all \
  --data-dir /path/to/attack_data \
  --mapping-file data/domain_details.xlsx \
  --model-dir /path/to/saved_models/exp5 \
  --run-dir /path/to/new_evaluation_output
```

The path adaptation changes input and output configuration only. It does not change the preserved model, preprocessing, or training algorithms. The current original trainer and the final thesis protocol contain historical implementation differences; the current code must not be assumed to reproduce every reported three-seed result without verifying the corresponding historical version and settings.

### Data-discovery verification

A read-only check of the existing dataset discovered 60 domains. Excluding `failing_node` left 48 domains, all with at least 20 CSV files. This verifies data discovery, not full training reproduction.

## Temporal diagnostics

The three Windows-based diagnostics are:

- `experiments/temporal_diagnostics/validate_txrx_time_confound.py`
- `experiments/temporal_diagnostics/validate_relative_time.py`
- `experiments/temporal_diagnostics/validate_time_plus_delta_txrx.py`

They examine cumulative TX/RX, differences calculated from aggregated measurements, and time-related baselines. These are separate from the node-level controlled validation. The published scripts retain their historical behaviour; their full execution from this layout has not been verified.

## Controlled validation

The controlled-validation source is located in `simulation/temporal_validation/`, not in a separate `experiments/controlled_validation/` directory. It includes cross-start-time validation, on-off attack variants, nested and reduced RPL feature sets, TX/RX ablation, early detection, benign-phase false-positive analysis, and effect-size analysis.

### Feature generation

The generators support `--project-root` and optional `--run-dir`. The project root must be a complete compatible simulation checkout with the required raw outputs and upstream dependencies.

Base attacks:

```bash
python simulation/temporal_validation/generate_features.py \
  --project-root /path/to/simulation_project
```

On-off attacks:

```bash
python simulation/temporal_validation/generate_features_onoff.py \
  --project-root /path/to/simulation_project
```

The optional `--run-dir` selects one validation output directory. The generators expect the controlled-validation output layout; they are not interchangeable with the original aggregated-CSV preprocessing.

### Cross-start-time training

The following command-line interfaces were verified in the existing Ubuntu virtual environment:

```bash
python simulation/temporal_validation/train_cross_start_validation_4attacks.py \
  --data-root /path/to/processed_validation_data \
  --output-dir /path/to/new_output_directory
```

```bash
python simulation/temporal_validation/train_cross_start_validation_onoff.py \
  --data-root /path/to/processed_onoff_data \
  --output-dir /path/to/new_output_directory
```

The data root must contain the processed validation data expected by the relevant script. Other controlled-validation analyses have their own input and output arguments. No full controlled training run was performed during repository preparation.

## Simulation environment

The thesis simulation work was developed from [uu-core/ids-WPLR](https://github.com/uu-core/ids-WPLR). The exported `simulation/` directory contains thesis-specific source and configurations; it is not a standalone replacement for the complete upstream checkout. A compatible Contiki-NG and Cooja installation, upstream scripts, and the appropriate scenarios are required.

Recorded source revisions:

| Component | Revision |
| --- | --- |
| Original `ids-WPLR` base | `5f015edacecd6c05474b3c77977024c8050ff1c2` |
| Contiki-NG | `27cbe8cc67ca1670d5a0fe52156bc91588bc8623` |
| Cooja | `86aaa5e5718ad149d42f61ad7fd08136acdbdcaa` |

These revisions document the source environment; they do not constitute a verified clean-install procedure. The relationship between the original simulation checkout and the publication repository must be considered when reconstructing the environment.

## Reproducibility status

Completed checks include the original domain-discovery test, syntax and backup checks for the original path adaptation, and command-line help checks for the controlled feature generators and two cross-start-time trainers.

Full end-to-end reproduction from a clean machine, complete dependency version pinning, and exact historical run provenance remain to be verified. The preparation of this repository did not regenerate or alter the reported thesis results. In particular, the user's historical manual seed runs are not replaced by the current code or by the checks described here.

## Contributions and attribution

The thesis-specific work includes modified simulation and data-generation code, radio-feature comparison, cross-domain evaluation, temporal-confounding diagnostics, and controlled validation. The original research-group code and authors' contributions remain attributed to the upstream projects.

## Citation

Please cite the master's thesis and the research group's original work when using this code. Final bibliographic details will be added after confirmation with the supervisor.

## License

The upstream repository contains a BSD-3-Clause license. The licensing status of thesis-specific additions and any applicable research-group requirements should be confirmed before assigning a separate license. This README does not grant additional rights.
