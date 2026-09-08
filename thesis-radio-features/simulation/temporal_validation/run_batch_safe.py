#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

POLL_SECONDS = 5
DEFAULT_TIMEOUT_MINUTES = 50


@dataclass(frozen=True)
class Job:
    attack: str
    node_count: int
    attack_start_minutes: int | None
    run: int
    scenario_path: str
    source_scenario: str

    @property
    def group(self) -> tuple[str, int | None]:
        return self.attack, self.attack_start_minutes


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal-validation Cooja scenarios safely, recover after TEST OK, and resume completed runs."
    )
    parser.add_argument("--workers", type=int, default=2, help="Parallel scenario groups; default: 2")
    parser.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--attack", choices=["dis_flooding", "local_repair", "blackhole", "worst_parent", "benign"])
    parser.add_argument("--start", type=int, choices=[300, 450, 600])
    parser.add_argument("--run", type=int, choices=range(1, 11))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_manifest(root: Path, args: argparse.Namespace) -> list[Job]:
    path = root / "temporal_validation" / "scenario_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    jobs: list[Job] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            attack = row["attack"]
            start_text = row["attack_start_minutes"].strip()
            start = int(start_text) if start_text else None
            run = int(row["run"])

            if args.attack and attack != args.attack:
                continue
            if args.start is not None and start != args.start:
                continue
            if args.run is not None and run != args.run:
                continue

            jobs.append(
                Job(
                    attack=attack,
                    node_count=int(row["node_count"]),
                    attack_start_minutes=start,
                    run=run,
                    scenario_path=row["scenario_path"],
                    source_scenario=row["source_scenario"],
                )
            )
    return jobs


def destination(root: Path, job: Job) -> Path:
    output = root / "applications" / "example-attacks" / "validation_outputs"
    if job.attack == "benign":
        return output / "benign" / "all_benign" / f"run_{job.run:02d}"
    return output / job.attack / f"start_{job.attack_start_minutes}" / f"run_{job.run:02d}"


def is_complete(dest: Path) -> bool:
    required = [
        dest / "mote-output.log",
        dest / "script.log",
        dest / "features_temporal_validation.csv",
        dest / "validation_metadata.json",
    ]
    return all(path.exists() for path in required)


def kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 8
    while process.poll() is None and time.time() < deadline:
        time.sleep(0.5)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def contains_test_ok(script_log: Path) -> bool:
    if not script_log.exists():
        return False
    try:
        with script_log.open("rb") as handle:
            handle.seek(max(0, script_log.stat().st_size - 8192))
            return b"TEST OK" in handle.read()
    except OSError:
        return False


def generated_directories(scenario_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in scenario_dir.iterdir():
        if path.is_dir() and (path / "mote-output.log").exists():
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def remove_stale_outputs(scenario_dir: Path) -> None:
    for path in generated_directories(scenario_dir):
        shutil.rmtree(path, ignore_errors=True)


def write_metadata(dest: Path, job: Job) -> None:
    text = (
        f"attack={job.attack}\n"
        f"node_count={job.node_count}\n"
        f"attack_start_minutes={'' if job.attack_start_minutes is None else job.attack_start_minutes}\n"
        f"run={job.run}\n"
        f"scenario_path={job.scenario_path}\n"
        f"source_scenario={job.source_scenario}\n"
    )
    (dest / "run_metadata.txt").write_text(text, encoding="utf-8")


def generate_features(root: Path, dest: Path, runner_log: Path) -> None:
    command = [
        sys.executable,
        str(root / "temporal_validation" / "generate_features.py"),
        "--run-dir",
        str(dest),
    ]
    with runner_log.open("ab") as handle:
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Feature generation failed for {dest}; see {runner_log}")


def run_one(root: Path, job: Job, timeout_minutes: int, force: bool) -> str:
    dest = destination(root, job)
    label = f"{job.attack} start={job.attack_start_minutes or 'NA'} run={job.run:02d}"

    if is_complete(dest) and not force:
        return f"SKIP {label}"

    if force and dest.exists():
        shutil.rmtree(dest)

    csc = root / job.scenario_path
    if not csc.exists():
        raise FileNotFoundError(csc)
    scenario_dir = csc.parent
    remove_stale_outputs(scenario_dir)

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_log = dest.parent / f"runner-run_{job.run:02d}.log"
    if temp_log.exists():
        temp_log.unlink()

    env = os.environ.copy()
    env["COOJA_DISABLE_RADIO_TRACE"] = "1"
    command = [
        "xvfb-run",
        "-a",
        str(root / "tools" / "cooja" / "scripts" / "run-cooja.py"),
        str(csc),
    ]

    with temp_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.time() + timeout_minutes * 60
    generated: Path | None = None
    success = False

    try:
        while time.time() < deadline:
            dirs = generated_directories(scenario_dir)
            if dirs:
                generated = dirs[-1]
                if contains_test_ok(generated / "script.log"):
                    success = True
                    break

            if process.poll() is not None:
                time.sleep(2)
                dirs = generated_directories(scenario_dir)
                if dirs:
                    generated = dirs[-1]
                    success = contains_test_ok(generated / "script.log")
                break
            time.sleep(POLL_SECONDS)
    finally:
        kill_process_group(process)

    if not success or generated is None:
        raise RuntimeError(f"Simulation failed or timed out: {label}; see {temp_log}")

    for name in ("radio-log.pcap", "radio-medium.log"):
        try:
            (generated / name).unlink()
        except FileNotFoundError:
            pass

    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(generated), str(dest))
    write_metadata(dest, job)
    shutil.move(str(temp_log), str(dest / "runner.log"))
    generate_features(root, dest, dest / "runner.log")

    return f"DONE {label}"


def run_group(root: Path, jobs: list[Job], timeout_minutes: int, force: bool) -> list[str]:
    results: list[str] = []
    for job in sorted(jobs, key=lambda item: item.run):
        message = run_one(root, job, timeout_minutes, force)
        print(message, flush=True)
        results.append(message)
    return results


def stop_old_validation_processes() -> None:
    subprocess.run(["pkill", "-f", "run-cooja.py.*validation_scenarios"], check=False)
    subprocess.run(["pkill", "-f", "gradlew.*validation_scenarios"], check=False)
    time.sleep(2)


def precompile(root: Path) -> None:
    app_dir = root / "applications" / "example-attacks"
    jobs = max(1, min(4, os.cpu_count() or 1))
    command = ["make", f"-j{jobs}", "udp-client-validation.cooja", "TARGET=cooja"]
    print("Precompiling validation mote...", flush=True)
    subprocess.run(command, cwd=app_dir, check=True)


def rebuild_global_summary(root: Path) -> None:
    command = [sys.executable, str(root / "temporal_validation" / "generate_features.py")]
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    root = project_root()
    stop_old_validation_processes()

    jobs = read_manifest(root, args)
    if not jobs:
        raise SystemExit("No jobs matched the filters")

    groups: dict[tuple[str, int | None], list[Job]] = {}
    for job in jobs:
        groups.setdefault(job.group, []).append(job)

    workers = max(1, min(args.workers, len(groups)))
    print(f"Jobs: {len(jobs)}; groups: {len(groups)}; workers: {workers}", flush=True)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_group, root, group_jobs, args.timeout_minutes, args.force): group
            for group, group_jobs in groups.items()
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                future.result()
            except Exception as exc:
                message = f"FAILED group {group}: {exc}"
                print(message, file=sys.stderr, flush=True)
                failures.append(message)

    rebuild_global_summary(root)

    if failures:
        raise SystemExit("\n".join(failures))

    print("All selected simulations completed and features were generated.")


if __name__ == "__main__":
    main()
