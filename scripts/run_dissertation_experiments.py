#!/usr/bin/env python
"""Single reproducible entry point for Greedy, exact-DP, and deep experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[
        "prepare", "tests", "greedy", "dp", "neural", "plots", "audit",
        "core", "all",
    ], default="all")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dissertation_main"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _environment() -> dict:
    packages = {}
    try:
        import numpy
        import pandas
        import torch
        packages = {
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    except Exception as error:  # environment capture must not hide pipeline errors
        packages = {"environment_error": repr(error)}
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "git_revision": _git_revision(),
        "packages": packages, "pip_freeze": freeze.stdout.splitlines(),
    }


def _commands(args) -> dict[str, list[str]]:
    python = sys.executable
    output = args.output_dir
    quick = ["--quick"] if args.quick else []
    resume = ["--resume"] if args.resume else []
    return {
        "prepare": [python, "scripts/prepare_dissertation_data.py",
                    "--output-dir", str(output)],
        "tests": [python, "-m", "pytest", "-q"],
        "greedy": [python, "scripts/eval_greedy.py",
                   "--output-dir", str(output / "greedy"), *quick],
        "dp": [python, "scripts/evaluate_dp_stage.py",
               "--output-dir", str(output / "dp"), *quick],
        "neural": [python, "scripts/evaluate_neural_distance_stage.py",
                   "--output-dir", str(output / "neural"),
                   "--device", args.device, *quick, *resume],
        "plots": [python, "scripts/plot_neural_distance_diagnostics.py",
                  str(output / "neural")],
        "audit": [python, "scripts/audit_dissertation_results.py",
                  "--results-root", str(output),
                  *(["--allow-incomplete"] if args.quick else [])],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2), encoding="utf-8")
    requested = {
        "core": ["prepare", "tests", "greedy", "dp", "neural"],
        "all": ["prepare", "tests", "greedy", "dp", "neural", "plots", "audit"],
    }.get(args.stage, [args.stage])
    commands = _commands(args)
    status_path = args.output_dir / "pipeline_status.csv"
    existing = []
    if args.resume and status_path.is_file():
        existing = list(csv.DictReader(status_path.open(encoding="utf-8")))
    completed = {row["stage"] for row in existing if row["status"] == "success"}
    rows = existing[:]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

    for stage in requested:
        if args.resume and stage in completed:
            print(f"[resume] {stage}: already complete")
            continue
        command = commands[stage]
        print(f"\n[{stage}] {subprocess.list2cmdline(command)}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, env=environment, check=False)
        row = {
            "stage": stage,
            "status": "success" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "command": subprocess.list2cmdline(command),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows = [old for old in rows if old["stage"] != stage] + [row]
        with status_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        if result.returncode:
            raise SystemExit(f"Stage {stage!r} failed; inspect {status_path}")

    state = {
        "phase": "complete", "requested_stages": requested,
        "quick": args.quick, "resume": args.resume,
        "methodology": {
            "basic": "adjacent bottom-up Greedy with handcrafted distances",
            "intermediate": "exact ordered DP under a fixed additive objective",
            "advanced": "learned temporal Siamese distance with Greedy and exact DP",
        },
        "rl_status": (
            "The reviewed archive's RL-v2 is not silently substituted for the "
            "current temporal model; its frozen evidence showed no F1 advantage "
            "over exact DP and it remains an explicitly exploratory ablation."
        ),
    }
    (args.output_dir / "pipeline_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
