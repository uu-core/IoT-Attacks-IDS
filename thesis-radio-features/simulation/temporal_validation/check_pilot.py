#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one generated temporal-validation run")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Default: dis_flooding/start_300/run_01",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_dir = args.run_dir or (
        project_root
        / "applications"
        / "example-attacks"
        / "validation_outputs"
        / "dis_flooding"
        / "start_300"
        / "run_01"
    )
    run_dir = run_dir.resolve()

    metadata_path = run_dir / "validation_metadata.json"
    features_path = run_dir / "features_temporal_validation.csv"
    nodes_path = run_dir / "node_interval_observations.csv"

    for path in (metadata_path, features_path, nodes_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text())
    features = pd.read_csv(features_path)
    node_data = pd.read_csv(nodes_path)

    checks = {
        "parsed 15 client nodes": metadata["parsed_nodes"] == 15,
        "attack event is exactly 300 minutes after steady state": metadata["attack_time_relative_us"] == 300 * 60_000_000,
        "simulation covers approximately 900 minutes": 899 * 60_000_000 <= metadata["total_time_relative_us"] <= 901 * 60_000_000,
        "minute feature file has at least 895 labeled bins": len(features) >= 895,
        "both benign and attack labels exist": set(features["label"].dropna().astype(int).unique()) == {0, 1},
        "TX interval values exist": node_data["delta_txraw"].notna().sum() > 0,
        "RX interval values exist": node_data["delta_rxraw"].notna().sum() > 0,
        "no TX counter reset": metadata["negative_tx_differences"] == 0,
        "no RX counter reset": metadata["negative_rx_differences"] == 0,
    }

    print(f"Run: {run_dir}")
    print(f"Feature rows: {len(features)}")
    print(f"Observed nodes: {metadata['parsed_nodes']}")
    print(f"Mean nodes per minute: {features['nodes_observed'].mean():.2f}")
    print(f"Minimum nodes in a non-empty minute: {features.loc[features['nodes_observed'] > 0, 'nodes_observed'].min()}")
    print(f"Mean interval TX raw ticks: {node_data['delta_txraw'].mean():.3f}")
    print(f"Mean interval RX raw ticks: {node_data['delta_rxraw'].mean():.3f}")
    print("")

    failed = []
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            failed.append(name)

    if failed:
        raise SystemExit(f"Pilot validation failed {len(failed)} check(s)")

    print("\nPilot validation passed. The remaining 69 simulations can be run.")


if __name__ == "__main__":
    main()
