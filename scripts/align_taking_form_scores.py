#!/usr/bin/env python
"""Align converted Taking Form hierarchies to official DCML score timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_manifest import sha256_file
from score_alignment import align_piece_to_dcml_score, taking_form_score_key
from shared_annotation_format import SharedPiece
from shared_evaluation import algorithm_readiness


SOURCES = {
    "Taking_Form_Beethoven": "beethoven_piano_sonatas",
    "Taking_Form_Mozart": "mozart_piano_sonatas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path,
                        default=Path("data/shared_annotations_v1"))
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/shared_annotations_v1_aligned"))
    parser.add_argument("--dataset", choices=["beethoven", "mozart", "both"],
                        default="both")
    parser.add_argument("--piece", action="append")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = set(SOURCES)
    if args.dataset == "beethoven":
        wanted = {"Taking_Form_Beethoven"}
    elif args.dataset == "mozart":
        wanted = {"Taking_Form_Mozart"}
    selected = set(args.piece or [])
    jobs = []
    for dataset in sorted(wanted):
        paths = sorted((args.shared_root / dataset).glob("*.json"))
        if selected:
            paths = [path for path in paths if path.stem in selected]
        if args.quick and paths:
            paths = paths[:1]
        jobs.extend((dataset, path) for path in paths)
    if not jobs:
        raise SystemExit("No converted Taking Form JSON files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset, path in jobs:
        piece = SharedPiece.read_json(path)
        try:
            score_piece = taking_form_score_key(piece.piece_id, dataset)
            score_root = args.external_root / SOURCES[dataset]
            measures = score_root / "measures" / f"{score_piece}.measures.tsv"
            notes = score_root / "notes" / f"{score_piece}.notes.tsv"
            if not measures.exists() or not notes.exists():
                missing = []
                if not measures.exists():
                    missing.append(str(measures))
                if not notes.exists():
                    missing.append(str(notes))
                raise FileNotFoundError("; ".join(missing))
            aligned, diagnostics = align_piece_to_dcml_score(
                piece, measures, notes, score_piece=score_piece)
            target = args.output_dir / dataset / path.name
            aligned.write_json(target)
            rows.append({
                "dataset": dataset,
                "piece": piece.piece_id,
                "score_piece": score_piece,
                "status": "success",
                "message": "",
                **diagnostics.to_dict(),
                **algorithm_readiness(aligned),
                "output_path": str(target),
                "output_sha256": sha256_file(target),
            })
        except Exception as error:
            rows.append({
                "dataset": dataset,
                "piece": piece.piece_id,
                "score_piece": "",
                "status": "failed" if not isinstance(error, FileNotFoundError) else "missing_score",
                "message": repr(error),
            })
            if args.strict:
                raise
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "alignment_manifest.csv", index=False)
    summary = {
        "requested": len(jobs),
        "aligned": int((frame.status == "success").sum()),
        "missing_score": int((frame.status == "missing_score").sum()),
        "failed": int((frame.status == "failed").sum()),
        "pitch_clustering_ready": int(
            frame.get("pitch_clustering_ready", pd.Series(dtype=bool)).fillna(False).sum()),
        "warning": (
            "File-name and coordinate alignment is necessary but not sufficient: "
            "edition and repeat semantics must be treated as validity checks."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["failed"] or summary["missing_score"]:
        print("Partial alignment: inspect alignment_manifest.csv for rejected or missing scores.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
