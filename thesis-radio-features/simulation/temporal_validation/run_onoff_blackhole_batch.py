#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ATTACK = "blackhole"
STARTS = (300, 450, 600)
RUNS = range(1, 11)

POLL_SECONDS = 5
TIMEOUT_MINUTES = 50


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def kill_process(process):
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    time.sleep(3)

    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def has_test_ok(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        return "TEST OK" in path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return False


def generated_dirs(scenario_dir: Path) -> list[Path]:
    result = []

    for p in scenario_dir.iterdir():
        if p.is_dir() and (p / "mote-output.log").exists():
            result.append(p)

    return sorted(
        result,
        key=lambda p: p.stat().st_mtime,
    )


def output_dir(project: Path, start: int, run: int) -> Path:
    return (
        project
        / "applications"
        / "example-attacks"
        / "validation_outputs_onoff"
        / ATTACK
        / f"start_{start}"
        / f"run_{run:02d}"
    )


def scenario_file(project: Path, start: int, run: int) -> Path:
    return (
        project
        / "applications"
        / "example-attacks"
        / "validation_scenarios_onoff"
        / ATTACK
        / "15"
        / f"start_{start}"
        / f"{ATTACK}-oo-15-start-{start}-run-{run:02d}.csc"
    )


def complete(dest: Path) -> bool:
    return (
        (dest / "mote-output.log").exists()
        and (dest / "script.log").exists()
        and (dest / "features_temporal_validation.csv").exists()
    )


def generate_features(project: Path, dest: Path) -> None:
    log = dest / "runner.log"

    command = [
        sys.executable,
        str(
            project
            / "temporal_validation"
            / "generate_features_onoff.py"
        ),
        "--run-dir",
        str(dest),
    ]

    with log.open("ab") as handle:
        result = subprocess.run(
            command,
            cwd=project,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Feature generation failed: {dest}"
        )


def run_one(start: int, run: int) -> str:
    project = root()

    csc = scenario_file(project, start, run)
    dest = output_dir(project, start, run)

    label = f"Blackhole on-off start={start} run={run:02d}"

    if complete(dest):
        return f"SKIP {label}"

    if not csc.exists():
        raise FileNotFoundError(csc)

    scenario_dir = csc.parent

    for p in generated_dirs(scenario_dir):
        shutil.rmtree(p, ignore_errors=True)

    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner_temp = (
        dest.parent
        / f"runner-run_{run:02d}.log"
    )

    if runner_temp.exists():
        runner_temp.unlink()

    env = os.environ.copy()
    env["COOJA_DISABLE_RADIO_TRACE"] = "1"

    command = [
        "xvfb-run",
        "-a",
        str(
            project
            / "tools"
            / "cooja"
            / "scripts"
            / "run-cooja.py"
        ),
        str(csc),
    ]

    with runner_temp.open("wb") as handle:
        process = subprocess.Popen(
            command,
            cwd=project,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = (
        time.time()
        + TIMEOUT_MINUTES * 60
    )

    generated = None
    success = False

    try:
        while time.time() < deadline:
            dirs = generated_dirs(
                scenario_dir
            )

            if dirs:
                generated = dirs[-1]

                if has_test_ok(
                    generated / "script.log"
                ):
                    success = True
                    break

            if process.poll() is not None:
                time.sleep(2)

                dirs = generated_dirs(
                    scenario_dir
                )

                if dirs:
                    generated = dirs[-1]

                    success = has_test_ok(
                        generated / "script.log"
                    )

                break

            time.sleep(POLL_SECONDS)

    finally:
        kill_process(process)

    if not success or generated is None:
        raise RuntimeError(
            f"Simulation failed or timed out: {label}"
        )

    for name in (
        "radio-log.pcap",
        "radio-medium.log",
    ):
        try:
            (generated / name).unlink()
        except FileNotFoundError:
            pass

    if dest.exists():
        shutil.rmtree(dest)

    shutil.move(
        str(generated),
        str(dest),
    )

    metadata = (
        f"attack={ATTACK}\n"
        f"variant=on-off\n"
        f"node_count=15\n"
        f"attack_start_minutes={start}\n"
        f"run={run}\n"
        f"scenario_path={csc.relative_to(project)}\n"
    )

    (dest / "run_metadata.txt").write_text(
        metadata,
        encoding="utf-8",
    )

    shutil.move(
        str(runner_temp),
        str(dest / "runner.log"),
    )

    generate_features(
        project,
        dest,
    )

    return f"DONE {label}"


def run_start(start: int) -> list[str]:
    results = []

    for run in RUNS:
        message = run_one(
            start,
            run,
        )

        print(
            message,
            flush=True,
        )

        results.append(message)

    return results


def main() -> None:
    print(
        "Running 30 Blackhole on-off controlled simulations...",
        flush=True,
    )

    failures = []

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        futures = {
            executor.submit(
                run_start,
                start,
            ): start
            for start in STARTS
        }

        for future in as_completed(
            futures
        ):
            start = futures[future]

            try:
                future.result()

            except Exception as exc:
                message = (
                    f"FAILED start={start}: {exc}"
                )

                print(
                    message,
                    file=sys.stderr,
                    flush=True,
                )

                failures.append(
                    message
                )

    if failures:
        print()
        print("Some groups failed:")

        for message in failures:
            print(message)

        raise SystemExit(1)

    print()
    print(
        "All 30 Blackhole on-off runs completed."
    )


if __name__ == "__main__":
    main()
