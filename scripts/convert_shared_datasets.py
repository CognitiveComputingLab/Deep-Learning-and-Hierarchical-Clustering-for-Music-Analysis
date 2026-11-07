#!/usr/bin/env python
"""Convert structural corpora to the versioned shared annotation format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_manifest import sha256_file
from dataset_converters import convert_dcml_piece, convert_dezrann, convert_taking_form
from shared_annotation_format import SCHEMA_VERSION, validate_piece
from shared_evaluation import algorithm_readiness


DATASET_CHOICES = ("abc", "taking_form", "fugue", "mozart", "winterreise")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES,
                        default=list(DATASET_CHOICES))
    parser.add_argument("--abc-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--taking-form-root", type=Path,
                        default=Path("external/Taking-Form"))
    parser.add_argument("--algomus-root", type=Path,
                        default=Path("external/algomus-data"))
    parser.add_argument("--winterreise-root", type=Path,
                        default=Path("external/schubert_winterreise"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/shared_annotations_v1"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="stop at the first unavailable dataset or failed piece")
    return parser.parse_args()


def _dcml_pairs(root: Path):
    notes = {path.name.removesuffix(".notes.tsv"): path
             for path in (root / "notes").glob("*.notes.tsv")}
    harmonies = {path.name.removesuffix(".harmonies.tsv"): path
                 for path in (root / "harmonies").glob("*.harmonies.tsv")}
    return [(piece, notes[piece], harmonies[piece])
            for piece in sorted(set(notes) & set(harmonies))]


def _jobs(args):
    jobs = []
    statuses = []
    if "abc" in args.datasets:
        pairs = _dcml_pairs(args.abc_root)
        statuses.append({"dataset": "ABC_DCML", "available": bool(pairs),
                         "source_root": str(args.abc_root), "discovered": len(pairs),
                         "note": "flat local-key partition plus pedal/cadence/phrase events"})
        jobs.extend(("ABC_DCML", piece, lambda n=n, h=h: convert_dcml_piece(n, h))
                    for piece, n, h in pairs)
    if "taking_form" in args.datasets:
        for subdir, dataset in (("Beethoven_Sonatas", "Taking_Form_Beethoven"),
                                ("Mozart_Sonatas", "Taking_Form_Mozart")):
            paths = sorted((args.taking_form_root / "corpus" / subdir).glob("*.csv"))
            statuses.append({"dataset": dataset, "available": bool(paths),
                             "source_root": str(args.taking_form_root / "corpus" / subdir),
                             "discovered": len(paths),
                             "note": "explicit coarse-to-fine form hierarchy in measure/beat time"})
            jobs.extend((dataset, path.stem,
                         lambda p=path, d=dataset: convert_taking_form(p, dataset=d))
                        for path in paths)
    if "fugue" in args.datasets:
        paths = sorted((args.algomus_root / "fugues" / "bach-wtc-i").glob("*-ref.dez"))
        statuses.append({"dataset": "Algomus_Fugue", "available": bool(paths),
                         "source_root": str(args.algomus_root / "fugues" / "bach-wtc-i"),
                         "discovered": len(paths),
                         "note": "overlap-aware subjects/countersubjects plus cadence/pedal events"})
        jobs.extend(("Algomus_Fugue", path.stem,
                     lambda p=path: convert_dezrann(
                         p, dataset="Algomus_Fugue", profile="fugue"))
                    for path in paths)
    if "mozart" in args.datasets:
        paths = sorted((args.algomus_root / "quartets" / "mozart").glob("*-ref.dez"))
        statuses.append({"dataset": "Algomus_Mozart_Quartets", "available": bool(paths),
                         "source_root": str(args.algomus_root / "quartets" / "mozart"),
                         "discovered": len(paths),
                         "note": "numbered top.* form hierarchy plus tonality/cadence/pedal layers"})
        jobs.extend(("Algomus_Mozart_Quartets", path.stem,
                     lambda p=path: convert_dezrann(
                         p, dataset="Algomus_Mozart_Quartets", profile="mozart_sonata"))
                    for path in paths)
    if "winterreise" in args.datasets:
        pairs = _dcml_pairs(args.winterreise_root)
        statuses.append({"dataset": "Schubert_Winterreise_DCML",
                         "available": bool(pairs), "source_root": str(args.winterreise_root),
                         "discovered": len(pairs),
                         "note": ("DCML symbolic notes/harmonies supported directly; original "
                                  "multimodal structural CSV requires a separately downloaded release")})
        jobs.extend(("Schubert_Winterreise_DCML", piece,
                     lambda n=n, h=h: convert_dcml_piece(
                         n, h, dataset="Schubert_Winterreise_DCML"))
                    for piece, n, h in pairs)
    if args.quick:
        first = {}
        for job in jobs:
            first.setdefault(job[0], job)
        jobs = list(first.values())
    return jobs, statuses


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs, statuses = _jobs(args)
    unavailable = [row for row in statuses if not row["available"]]
    if args.strict and unavailable:
        raise SystemExit("Unavailable datasets: " + ", ".join(row["dataset"] for row in unavailable))
    manifest, errors = [], []
    for dataset, source_id, converter in jobs:
        try:
            piece = converter()
            validation = validate_piece(piece, raise_on_error=False)
            if validation:
                raise ValueError("; ".join(validation))
            readiness = algorithm_readiness(piece)
            safe_piece = re.sub(r"[^A-Za-z0-9_.-]+", "_", piece.piece_id)
            target = args.output_dir / dataset / f"{safe_piece}.json"
            piece.write_json(target)
            manifest.append({
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset, "piece": piece.piece_id, "work": piece.work_id,
                "coordinate_system": piece.coordinate_system,
                "duration_qb": piece.duration_qb,
                "annotations": len(piece.annotations),
                "hierarchical_segments": sum(
                    item.kind == "hierarchical_segment" for item in piece.annotations),
                "partition_segments": sum(
                    item.kind == "partition_segment" for item in piece.annotations),
                "overlapping_spans": sum(item.kind == "span" for item in piece.annotations),
                "point_events": sum(item.kind == "point" for item in piece.annotations),
                **readiness,
                "output_path": str(target),
                "output_sha256": sha256_file(target),
                "status": "success", "error": "",
            })
        except Exception as error:
            row = {"dataset": dataset, "piece": source_id,
                   "status": "failed", "error": repr(error)}
            errors.append(row)
            print(f"[failed] {dataset}/{source_id}: {error}")
            if args.strict:
                raise
    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(args.output_dir / "manifest.csv", index=False)
    pd.DataFrame(statuses).to_csv(args.output_dir / "dataset_status.csv", index=False)
    pd.DataFrame(errors, columns=["dataset", "piece", "status", "error"]).to_csv(
        args.output_dir / "conversion_errors.csv", index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "requested_datasets": args.datasets,
        "quick": args.quick,
        "converted_pieces": len(manifest), "failed_pieces": len(errors),
        "available_datasets": [row["dataset"] for row in statuses if row["available"]],
        "unavailable_datasets": [row["dataset"] for row in statuses if not row["available"]],
        "algorithm_ready_pieces": int(manifest_frame.pitch_clustering_ready.sum())
        if not manifest_frame.empty else 0,
        "hierarchical_reference_pieces": int(
            manifest_frame.hierarchical_evaluation_ready.sum())
        if not manifest_frame.empty else 0,
        "warning": (
            "Annotation conversion does not create missing score alignments or turn "
            "overlapping patterns into trees."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(f"{len(errors)} conversions failed; inspect conversion_errors.csv")


if __name__ == "__main__":
    main()
