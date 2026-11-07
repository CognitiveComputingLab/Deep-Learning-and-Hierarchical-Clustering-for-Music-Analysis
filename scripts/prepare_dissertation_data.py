#!/usr/bin/env python
"""Freeze a checksum manifest of the paired ABC/DCML evaluation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_manifest import build_abc_manifest, manifest_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dissertation_main"))
    parser.add_argument("--expected-pieces", type=int, default=70)
    parser.add_argument("--expected-works", type=int, default=16)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_abc_manifest(args.data_root)
    summary = manifest_summary(frame)
    summary.update({
        "expected_pieces": args.expected_pieces,
        "expected_works": args.expected_works,
        "complete": (
            summary["included_pieces"] == args.expected_pieces
            and summary["included_works"] == args.expected_works
            and summary["excluded_pieces"] == 0
        ),
        "scope": "paired ABC note TSV and DCML harmony TSV only",
    })
    frame.to_csv(args.output_dir / "data_manifest.csv", index=False)
    (args.output_dir / "data_manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["complete"] and not args.allow_incomplete:
        raise SystemExit(
            "Corpus does not match the preregistered 70-piece/16-work scope; "
            "inspect data_manifest.csv or pass --allow-incomplete for diagnostics.")


if __name__ == "__main__":
    main()
