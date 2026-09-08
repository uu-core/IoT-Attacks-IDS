#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MICROSECONDS_PER_MINUTE = 60_000_000
BIN_SIZE_US = MICROSECONDS_PER_MINUTE
TOTAL_MINUTES = 900

RPL_COUNTERS = ["disr", "diss", "dior", "dios", "diar", "tots"]
DATA_FIELDS = ["sq", "rank", "ver", *RPL_COUNTERS, "txraw", "rxraw"]

KEY_VALUE_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_]*):\s*(-?\d+)")


@dataclass
class RunMetadata:
    attack: str
    attack_start_minutes_expected: int | None
    run: int
    steady_time_us: int
    attack_time_us: int | None
    stop_time_us: int
    attack_time_relative_us: int | None
    total_time_relative_us: int
    parsed_nodes: int
    parsed_data_rows: int
    negative_tx_differences: int
    negative_rx_differences: int


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_run_path(run_dir: Path, output_root: Path) -> tuple[str, int | None, int]:
    relative = run_dir.relative_to(output_root)
    parts = relative.parts
    if len(parts) != 3:
        raise ValueError(f"Unexpected validation output path: {run_dir}")

    attack = parts[0]
    start_part = parts[1]
    run_match = re.fullmatch(r"run_(\d+)", parts[2])
    if not run_match:
        raise ValueError(f"Cannot parse run number from {run_dir}")
    run = int(run_match.group(1))

    if attack == "benign":
        return attack, None, run

    start_match = re.fullmatch(r"start_(\d+)", start_part)
    if not start_match:
        raise ValueError(f"Cannot parse attack start from {run_dir}")
    return attack, int(start_match.group(1)), run


def first_integer(text: str) -> int:
    match = re.match(r"\s*(\d+)", text)
    if not match:
        raise ValueError(f"No leading integer timestamp in line: {text!r}")
    return int(match.group(1))


def read_steady_time(script_log: Path, events_log: Path) -> int:
    if events_log.exists():
        for line in events_log.read_text(errors="replace").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3 and fields[1].strip().lower() == "network" and fields[2].strip().lower() == "steady-state":
                return first_integer(fields[0])

    for line in script_log.read_text(errors="replace").splitlines():
        if "network steady state!" in line.lower():
            return first_integer(line)

    raise RuntimeError(f"Could not find steady-state time in {script_log} or {events_log}")


def read_attack_time(events_log: Path) -> int | None:
    if not events_log.exists():
        return None
    for line in events_log.read_text(errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1].strip().lower() == "attack":
            return first_integer(fields[0])
    return None


def read_stop_time(script_log: Path) -> int:
    for line in script_log.read_text(errors="replace").splitlines():
        if "TEST OK" in line:
            return first_integer(line)
    raise RuntimeError(f"Could not find TEST OK in {script_log}")


def parse_data_message(message: str) -> dict[str, int] | None:
    if "DATA:" not in message:
        return None
    values = {key.lower(): int(value) for key, value in KEY_VALUE_PATTERN.findall(message)}
    missing = [field for field in DATA_FIELDS if field not in values]
    if missing:
        return None
    return {field: values[field] for field in DATA_FIELDS}


def read_node_observations(mote_log: Path) -> pd.DataFrame:
    log = pd.read_csv(mote_log, sep="\t", dtype={"message": str}, low_memory=False)
    required = {"# time", "mote", "message"}
    missing = required - set(log.columns)
    if missing:
        raise KeyError(f"Missing columns in {mote_log}: {sorted(missing)}")

    log = log.rename(columns={"# time": "time_us_raw"})

    records: list[dict[str, int]] = []
    for row in log[["time_us_raw", "mote", "message"]].itertuples(index=False):
        if pd.isna(row.mote) or pd.isna(row.time_us_raw):
            continue
        node = int(float(row.mote))
        if node == 1:
            continue

        parsed = parse_data_message(str(row.message))
        if parsed is None:
            continue

        records.append({"time_us": int(float(row.time_us_raw)), "node": node, **parsed})

    if not records:
        raise RuntimeError(f"No validation DATA rows found in {mote_log}. Was udp-client-validation.c compiled?")

    observations = pd.DataFrame.from_records(records)
    observations = observations.sort_values(["node", "time_us", "sq"]).drop_duplicates(
        subset=["node", "time_us", "sq"], keep="last"
    )
    return observations.reset_index(drop=True)


def add_node_level_differences(observations: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    result = observations.copy()
    cumulative_columns = [*RPL_COUNTERS, "txraw", "rxraw"]

    for column in cumulative_columns:
        delta_name = f"delta_{column}"
        result[delta_name] = result.groupby("node", sort=False)[column].diff()

    negative_tx = int((result["delta_txraw"] < 0).sum())
    negative_rx = int((result["delta_rxraw"] < 0).sum())

    for column in cumulative_columns:
        delta_name = f"delta_{column}"
        result.loc[result[delta_name] < 0, delta_name] = np.nan

    return result, negative_tx, negative_rx


def select_last_node_observation_per_bin(observations: pd.DataFrame, steady_time_us: int, total_relative_us: int) -> pd.DataFrame:
    data = observations.copy()
    data["relative_time_us"] = data["time_us"] - steady_time_us
    data = data[(data["relative_time_us"] >= 0) & (data["relative_time_us"] < total_relative_us)].copy()
    data["minute_bin"] = np.floor(data["relative_time_us"] / BIN_SIZE_US).astype(int)
    data = data.sort_values(["minute_bin", "node", "time_us"])
    data = data.groupby(["minute_bin", "node"], as_index=False).tail(1)
    return data.reset_index(drop=True)


def mean_std_features(node_bin_data: pd.DataFrame, source_columns: Iterable[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for column in source_columns:
        values = pd.to_numeric(node_bin_data[column], errors="coerce")
        output[f"{column}_mean"] = float(values.mean()) if values.notna().any() else math.nan
        output[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() >= 2 else 0.0
    return output


def build_minute_features(
    node_observations: pd.DataFrame,
    attack_time_relative_us: int | None,
    total_relative_us: int,
) -> pd.DataFrame:
    feature_columns = [
        "rank",
        *RPL_COUNTERS,
        *[f"delta_{column}" for column in RPL_COUNTERS],
        "txraw",
        "rxraw",
        "delta_txraw",
        "delta_rxraw",
    ]

    rows: list[dict[str, float | int | str]] = []
    total_bins = int(math.ceil(total_relative_us / BIN_SIZE_US))

    for minute_bin in range(total_bins):
        group = node_observations[node_observations["minute_bin"] == minute_bin]
        bin_start_us = minute_bin * BIN_SIZE_US
        bin_end_us = min((minute_bin + 1) * BIN_SIZE_US, total_relative_us)

        if attack_time_relative_us is None:
            label = 0
            phase = "benign"
        elif bin_end_us <= attack_time_relative_us:
            label = 0
            phase = "benign"
        elif bin_start_us >= attack_time_relative_us:
            label = 1
            phase = "attack"
        else:
            label = -1
            phase = "transition"

        row: dict[str, float | int | str] = {
            "minute_bin": minute_bin,
            "minute_start": minute_bin,
            "bin_start_us": bin_start_us,
            "bin_end_us": bin_end_us,
            "nodes_observed": int(group["node"].nunique()),
            "label": label,
            "phase": phase,
        }
        row.update(mean_std_features(group, feature_columns))
        rows.append(row)

    return pd.DataFrame(rows)


def process_run(run_dir: Path, output_root: Path) -> RunMetadata:
    attack, expected_start_min, run = parse_run_path(run_dir, output_root)
    mote_log = run_dir / "mote-output.log"
    script_log = run_dir / "script.log"
    events_log = run_dir / "events.log"

    if not mote_log.exists() or not script_log.exists():
        raise FileNotFoundError(f"Missing required logs in {run_dir}")

    steady_time_us = read_steady_time(script_log, events_log)
    attack_time_us = read_attack_time(events_log)
    stop_time_us = read_stop_time(script_log)
    total_relative_us = stop_time_us - steady_time_us
    attack_relative_us = None if attack_time_us is None else attack_time_us - steady_time_us

    if attack == "benign" and attack_time_us is not None:
        raise RuntimeError(f"Benign run unexpectedly contains an attack event: {run_dir}")
    if attack != "benign" and attack_time_us is None:
        raise RuntimeError(f"Attack run contains no attack event: {run_dir}")

    if expected_start_min is not None and attack_relative_us is not None:
        expected_us = expected_start_min * MICROSECONDS_PER_MINUTE
        if abs(attack_relative_us - expected_us) > 1_000:
            raise RuntimeError(
                f"Attack event mismatch in {run_dir}: expected {expected_us}, observed {attack_relative_us}"
            )

    observations = read_node_observations(mote_log)
    observations, negative_tx, negative_rx = add_node_level_differences(observations)
    observations = select_last_node_observation_per_bin(observations, steady_time_us, total_relative_us)

    minute_features = build_minute_features(observations, attack_relative_us, total_relative_us)
    minute_features.insert(0, "run", run)
    minute_features.insert(0, "attack_start_minutes", expected_start_min if expected_start_min is not None else np.nan)
    minute_features.insert(0, "attack", attack)

    observations.to_csv(run_dir / "node_interval_observations.csv", index=False)
    minute_features.to_csv(run_dir / "features_temporal_validation_all_bins.csv", index=False)
    minute_features[minute_features["label"] >= 0].to_csv(
        run_dir / "features_temporal_validation.csv", index=False
    )

    metadata = RunMetadata(
        attack=attack,
        attack_start_minutes_expected=expected_start_min,
        run=run,
        steady_time_us=steady_time_us,
        attack_time_us=attack_time_us,
        stop_time_us=stop_time_us,
        attack_time_relative_us=attack_relative_us,
        total_time_relative_us=total_relative_us,
        parsed_nodes=int(observations["node"].nunique()),
        parsed_data_rows=int(len(observations)),
        negative_tx_differences=negative_tx,
        negative_rx_differences=negative_rx,
    )
    (run_dir / "validation_metadata.json").write_text(json.dumps(asdict(metadata), indent=2))
    return metadata


def discover_runs(output_root: Path) -> list[Path]:
    return sorted(path.parent for path in output_root.rglob("mote-output.log"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate node-level interval and minute-level validation features")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--run-dir", type=Path, default=None, help="Process only one validation output directory")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = project_root / "applications" / "example-attacks" / "validation_outputs"
    run_dirs = [args.run_dir.resolve()] if args.run_dir else discover_runs(output_root)
    if not run_dirs:
        raise RuntimeError(f"No validation outputs found under {output_root}")

    records = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir}")
        metadata = process_run(run_dir, output_root)
        records.append(asdict(metadata))

    summary = pd.DataFrame(records)
    summary_path = project_root / "temporal_validation" / "feature_generation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Processed {len(records)} runs")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
