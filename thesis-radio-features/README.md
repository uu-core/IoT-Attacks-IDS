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

## Expected results

The values below are reference results reported in the thesis, not results generated during repository preparation. They describe the specified historical experiments and should not be treated as pass/fail thresholds for a single new run. Reproduction requires matching the dataset, preprocessing, model implementation, evaluation protocol, and aggregation. The original cumulative experiment and the controlled interval-feature experiment are separate evaluations.

### Original feature comparison

The broad original experiment compares four attacks across 5, 10, 15, and 20 nodes and three behavioural variants. The five feature configurations are All, RSSI-only, RPL+TX/RX, RPL-only, and TX/RX-only.

The thesis reports the following average in-domain F1-scores (Table 5.1):

| Attack | All | RSSI-only | RPL+TX/RX | RPL-only | TX/RX-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blackhole | 0.960 | 0.410 | 0.959 | 0.953 | 0.998 |
| DIS Flooding | 0.985 | 0.422 | 0.986 | 0.982 | 0.993 |
| Local Repair | 0.960 | 0.533 | 0.970 | 0.969 | 0.999 |
| Worst Parent | 0.969 | 0.515 | 0.983 | 0.951 | 0.999 |

These results show strong phase separation by cumulative TX/RX and comparatively weak RSSI-only performance. They do not establish near-perfect attack-specific radio detection, because the original measurements and labels contain temporal structure.

**Expected output files:** The preserved original trainer writes `saved_models/expN/<domain>.pt`, `results/exp_features_N/<domain>/metrics.json`, and timestamped files under `logs/` within the selected run directory. Each metrics JSON contains accuracy, F1, precision, recall, AUC, and a confusion matrix. Table 5.1 is produced by `make_table_5_1.py`; the plotting scripts generate the corresponding comparison and distribution figures. The historical table values are not guaranteed by the current single-run trainer.

### Original cross-domain evaluation

The thesis reports an off-diagonal mean F1-score of approximately 0.853 for cumulative TX/RX-only and 0.390 for RSSI-only, with medians of approximately 0.984 and 0.493, respectively (Table 5.2). The original design therefore favours TX/RX, but performance varies across target attacks, network sizes, and behavioural variants. The result must not be interpreted as universal transferability after temporal confounding is removed.

**Expected output files:** `cross_test.py` writes `results/cross_test/expN/<source_domain>/vs_<target_domain>.json` under the selected output root. The JSON contains classification metrics and a confusion matrix. `plot_cross_domain_summary.py` and `plot_feature_f1_distribution.py` produce the grouped summaries and distribution figures from the evaluation outputs.

### Temporal-confounding diagnostics

The diagnostic experiments on the original dataset report an approximate mean F1-score of 0.966 for absolute time-only features, 0.906 for cumulative TX/RX, and 0.440 for differences calculated from already aggregated TX/RX. Relative time performs almost identically to absolute time, and adding aggregate differences to time does not improve the time-only baseline. These results show that simulation progression explains a substantial part of the original performance. Aggregate differencing is not equivalent to reconstructing node-level interval activity.

**Expected output files in the historical diagnostic layout:**

| Script | Main generated files |
| --- | --- |
| `validate_txrx_time_confound.py` | `results/temporal_validation/temporal_validation_all_runs.csv`, `temporal_validation_summary.csv`, `temporal_validation_by_domain.csv`, `temporal_validation_by_attack.csv`, `temporal_validation_f1.png` |
| `validate_relative_time.py` | `results/relative_time_validation/relative_time_all_runs.csv`, `relative_time_summary.csv`, `absolute_vs_relative_time_summary.csv`, `relative_time_f1.png` |
| `validate_time_plus_delta_txrx.py` | `results/time_plus_delta_validation/time_plus_delta_all_runs.csv`, `time_plus_delta_summary.csv`, `incremental_value_summary.csv`, `incremental_value_by_domain.csv`, `time_plus_delta_f1.png`, `incremental_f1_gain.png` |

These are the filenames recorded in the original scripts and historical output tree. The diagnostic scripts retain their original path assumptions; their complete execution from the published layout has not been verified.

### Controlled validation

The controlled validation uses node-level interval features, held-out attack-start positions, complete-run train/validation/test separation, and genuine 10-step LSTM sequences. The base comparison uses four attacks in a fixed 15-node setting.

**Cross-start-time classification.** The thesis reports the following mean F1-scores for the base and on-off variants:

| Attack | TX/RX base | TX/RX on-off | RPL base | RPL on-off | RPL+TX/RX base | RPL+TX/RX on-off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DIS Flooding | 0.986 | 0.791 | 1.000 | 1.000 | 1.000 | 1.000 |
| Local Repair | 0.516 | 0.063 | 1.000 | 0.916 | 1.000 | 0.914 |
| Blackhole | 0.211 | 0.097 | 0.017 | 0.065 | 0.201 | 0.053 |
| Worst Parent | 0.059 | 0.322 | 0.804 | 0.714 | 0.806 | 0.713 |

The main conclusion is attack- and behaviour-dependent usefulness, not universal superiority of radio features.

**All-benign false positives.** In the final 300 minutes of fully benign runs, the reported false-positive rates are approximately 0.990 for time-only, 0.589 for cumulative TX/RX, and 0.030 for interval TX/RX. The interval representation substantially reduces the late-stage temporal effect.

**Nested RPL and TX/RX ablation.** Adding interval TX/RX to RPL-10 increases the Blackhole mean F1-score from approximately 0.044 to 0.238. The gain remains limited in absolute terms. TX alone is substantially more informative than RX alone for DIS Flooding, Local Repair, and Blackhole. The effect-size analysis supports stronger and more consistent transmission changes than reception changes.

**Early detection.** For Worst Parent, adding interval TX/RX to the complete RPL representation increases the three-minute detection rate from approximately 0.478 to 0.700, while the all-benign false-positive rate increases from approximately 0.006 to 0.051. Early-detection gains must therefore be considered together with false-positive costs.

**Expected generated files:** The feature generators write per-run node-level and minute-level CSVs, including `node_interval_observations.csv`, `features_temporal_validation_all_bins.csv`, `features_temporal_validation.csv`, `validation_metadata.json`, and `feature_generation_summary.csv`. The exact set depends on the generator and processing stage.

The cross-start-time trainers write the following files under the selected `--output-dir`:

```text
cross_start_all_runs.csv
cross_start_summary.csv
cross_start_overall.csv
cross_start_f1.png
```

The nested RPL, reduced RPL, and TX/RX ablation scripts follow the same detailed-run/summary/overall/figure pattern, including `nested_rpl_all_runs.csv`, `nested_rpl_summary.csv`, `nested_rpl_overall.csv`, `nested_rpl_f1.png`, `reduced_rpl_all_runs.csv`, `reduced_rpl_summary.csv`, `reduced_rpl_overall.csv`, `tx_rx_ablation_all_runs.csv`, `tx_rx_ablation_summary.csv`, `tx_rx_ablation_overall.csv`, and `tx_rx_ablation_f1.png`. Early-detection, all-benign, and effect-size scripts produce separate analysis outputs; their exact filenames and optional arguments are defined in their respective source files.

### Interpreting a new run

A successful execution should produce the expected output schema and finite metrics, but numerical agreement with the thesis requires the matching historical protocol and aggregation. The published code-only branch does not include the raw datasets or historical result files. No new experiment was run to generate the reference values above. The thesis itself is the authoritative source for the complete results and methodological limitations.

## Reproducibility status

Completed checks include the original domain-discovery test, syntax and backup checks for the original path adaptation, and command-line help checks for the controlled feature generators and two cross-start-time trainers.

Full end-to-end reproduction from a clean machine, complete dependency version pinning, and exact historical run provenance remain to be verified. The preparation of this repository did not regenerate or alter the reported thesis results. In particular, the user's historical manual seed runs are not replaced by the current code or by the checks described here.

## Contributions and attribution

The thesis-specific work includes modified simulation and data-generation code, radio-feature comparison, cross-domain evaluation, temporal-confounding diagnostics, and controlled validation. The original research-group code and authors' contributions remain attributed to the upstream projects.
