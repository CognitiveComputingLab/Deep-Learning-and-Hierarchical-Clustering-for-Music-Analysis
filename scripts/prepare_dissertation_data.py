#!/usr/bin/env python
"""Build the immutable data/checksum and explicit exclusion audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_manifest import build_data_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dissertation_main"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_data_manifest(PROJECT_ROOT)
    manifest.to_csv(args.output_dir / "data_manifest.csv", index=False)
    manifest[manifest.status != "included"].to_csv(
        args.output_dir / "exclusion_audit.csv", index=False)
    summary = (manifest.groupby(["dataset", "status"], as_index=False)
               .piece_id.count().rename(columns={"piece_id": "n_pieces"}))
    summary.to_csv(args.output_dir / "data_manifest_summary.csv", index=False)
    (args.output_dir / "data_revisions.json").write_text(json.dumps({
        dataset: {
            "annotation_revision": group.annotation_revision.iloc[0],
            "score_revision": group.score_revision.iloc[0],
        } for dataset, group in manifest.groupby("dataset")
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
