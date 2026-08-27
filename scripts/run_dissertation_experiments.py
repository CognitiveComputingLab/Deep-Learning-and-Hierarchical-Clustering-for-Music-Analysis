#!/usr/bin/env python
"""Single reproducible entry point for the dissertation experiment pipeline."""

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
    parser.add_argument(
        "--stage",
        choices=["prepare", "greedy", "dp", "deep", "visualize", "oof", "external", "core", "all"],
        default="all")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dissertation_main"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260827, 20260828, 20260829])
    return parser.parse_args()


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _environment():
    try:
        import numpy
        import pandas
        import torch
        packages = {
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    except Exception as error:
        packages = {"environment_error": repr(error)}
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "git_revision": _git_revision(),
        "packages": packages,
        "pip_freeze": freeze.stdout.splitlines(),
    }


def _commands(args):
    python = sys.executable
    output = args.output_dir
    common = ["--quick"] if args.quick else []
    seeds = [str(value) for value in args.seeds[:1] if args.quick] or [
        str(value) for value in args.seeds]
    return {
        "prepare": [python, "scripts/prepare_dissertation_data.py",
                    "--output-dir", str(output)],
        "greedy": [python, "scripts/eval_greedy.py",
                   "--output-dir", str(output / "greedy"), *common],
        "dp": [python, "scripts/evaluate_optimized_stage.py",
               "--output-dir", str(output / "dp"), *common],
        "deep": [python, "scripts/train_deep_clustering.py",
                 "--output-dir", str(output / "deep"),
                 "--device", args.device, "--seeds", *seeds,
                 *(["--resume"] if args.resume else []), *common],
        "visualize": [python, "scripts/visualize_pitchscape_comparison.py",
                      "--checkpoint-dir", str(output / "deep"),
                      "--output-dir", str(output / "pitch_scapes"),
                      "--device", args.device],
        "oof": [python, "scripts/evaluate_neural_corpus.py",
                "--output-dir", str(output / "neural_oof"),
                "--device", args.device, "--seeds", *seeds, *common],
        "external": [python, "scripts/evaluate_external_corpora.py",
                     "--manifest", str(output / "data_manifest.csv"),
                     "--checkpoint-dir", str(output / "deep"),
                     "--output-dir", str(output / "external"),
                     "--device", args.device, *common],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2), encoding="utf-8")
    requested = {
        "core": ["prepare", "greedy", "dp", "deep"],
        "all": ["prepare", "greedy", "dp", "deep", "visualize", "oof", "external"],
    }.get(args.stage, [args.stage])
    if args.quick and "visualize" in requested:
        if args.stage == "visualize":
            raise SystemExit(
                "Pitch Scape dissertation figures require a completed non-quick "
                "three-seed deep run.")
        requested = [stage for stage in requested if stage != "visualize"]
    commands = _commands(args)
    status_path = args.output_dir / "pipeline_status.csv"
    existing = []
    if args.resume and status_path.exists():
        existing = list(csv.DictReader(status_path.open(encoding="utf-8")))
    completed = {row["stage"] for row in existing if row["status"] == "success"}
    rows = existing[:]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
    failed = []
    for stage in requested:
        if args.resume and stage in completed:
            print(f"[resume] {stage}: already complete")
            continue
        command = commands[stage]
        print(f"\n[{stage}] {' '.join(command)}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                                check=False)
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
            writer.writeheader(); writer.writerows(rows)
        if result.returncode:
            failed.append(stage)
            if stage != "external":
                raise SystemExit(
                    f"Stage {stage!r} failed; inspect {status_path}")
    state = {
        "phase": "complete" if not failed else "complete_with_external_exclusions",
        "requested_stages": requested, "failed_stages": failed,
        "quick": args.quick, "resume": args.resume,
    }
    (args.output_dir / "pipeline_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8")
    if failed:
        print("Core experiments completed; external evaluation remains explicitly excluded.")


if __name__ == "__main__":
    main()
