#!/usr/bin/env python
"""Build an auditable dissertation code and evidence package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPTS = [
    "eval_greedy.py", "evaluate_boundary_aware_stage.py", "evaluate_dp_stage.py",
    "train_parametric_distance.py", "evaluate_neural_distance_stage.py",
    "run_dissertation_experiments.py", "prepare_dissertation_data.py",
    "audit_dissertation_results.py", "plot_neural_distance_diagnostics.py",
    "dptree_keyprofile.py", "dptree_circle_of_fifths.py", "dptree_weighted.py",
    "build_paper_release.py", "render_paper_pitchscapes.py",
    "render_parametric_case_pitchscapes.py",
    'aggregate_neural_shards.py', 'render_formal_neural_pitchscapes.py',
    'render_formal_case_pitchscapes.py',
]
CORE_SRC = [
    "greedy_clustering.py", "greedy_evaluation.py", "dp_clustering.py",
    "dp_stage_evaluation.py", "ordered_affinity.py", "parametric_distances.py",
    "neural_distance.py", "boundary_contrast.py", "data_manifest.py",
    "score_alignment.py", "dp_pitchscape.py", "dataset_converters.py",
    "hierarchical_evaluation.py", "hierarchical_parametric.py",
    "shared_annotation_format.py", "shared_evaluation.py",
]
DOCS = ["README.md", "PAPER_REPOSITORY.md", "ADVANCED_STAGE.md", "DATASETS.md", "NOTES.md"]
RESULTS = {
    "basic_greedy": Path("results/dissertation_main/greedy"),
    "intermediate_dp": Path("results/dissertation_main/dp"),
    "intermediate_parametric_supporting/beethoven_to_mozart": Path(
        "results/parametric_hierarchy_beethoven_to_mozart"),
    "intermediate_parametric_supporting/mozart_to_beethoven": Path(
        "results/parametric_hierarchy_mozart_to_beethoven"),
}
FIGURES = {
    "greedy/op95_key_profile_greedy.png": Path(
        "results/paper_pitchscapes/n11op95_01_greedy_greedy_key_profile.png"),
    "greedy/op95_circle_of_fifths_greedy.png": Path(
        "results/paper_pitchscapes/n11op95_01_greedy_greedy_circle_of_fifths.png"),
    "greedy/op95_tonic_weighted_greedy.png": Path(
        "results/paper_pitchscapes/n11op95_01_greedy_greedy_tonic_weighted.png"),
    "dp/op95_key_profile_dp.png": Path(
        "results/paper_pitchscapes/n11op95_01_dp_dp_key_profile.png"),
    "dp/op95_circle_of_fifths_dp.png": Path(
        "results/paper_pitchscapes/n11op95_01_dp_dp_circle_of_fifths.png"),
    "dp/op95_tonic_weighted_dp.png": Path(
        "results/paper_pitchscapes/n11op95_01_dp_dp_tonic_weighted.png"),
    'neural/op95_neural_greedy.png': Path(
        'results/paper_pitchscapes/n11op95_01_advanced_neural_greedy.png'),
    'neural/op95_neural_dp.png': Path(
        'results/paper_pitchscapes/n11op95_01_advanced_neural_dp.png'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_release")
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "results/dissertation_main")
    parser.add_argument("--force", action="store_true",
                        help="Refresh the generated output directory after checking its path.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def coverage(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    works = {row.get("work") for row in rows if row.get("work")}
    pieces = {row.get("piece") for row in rows if row.get("piece")}
    statuses = {row.get("status") for row in rows if row.get("status")}
    return {"exists": True, "rows": len(rows), "works": len(works),
            "pieces": len(pieces), "statuses": sorted(statuses)}


def advanced_status(results_root: Path) -> tuple[str, dict]:
    neural = results_root / "neural"
    audit_path = results_root / "final_evidence" / "final_audit.json"
    reasons = []
    audit = {}
    if not audit_path.is_file():
        reasons.append("final_evidence/final_audit.json is missing")
    else:
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reasons.append("final_audit.json is invalid JSON")
    if audit.get("ready_for_dissertation") is not True:
        reasons.append("final audit is not ready_for_dissertation=true")
    result_coverage = coverage(neural / "held_out_per_work.csv")
    if not result_coverage.get("exists"):
        reasons.append("neural/held_out_per_work.csv is missing")
    elif result_coverage.get("works") != 16:
        reasons.append(f"neural results cover {result_coverage['works']} works, expected 16")
    status = "complete" if not reasons else "pending"
    return status, {"status": status, "reasons": reasons, "audit": audit,
                    "coverage": result_coverage}


def validate_output_path(output: Path) -> Path:
    """Restrict destructive refreshes to a dedicated project subdirectory."""
    resolved = output.resolve()
    protected = {
        ROOT.resolve(),
        *((ROOT / name).resolve()
          for name in ("scripts", "src", "tests", "results", "external")),
    }
    if ROOT.resolve() not in resolved.parents or resolved in protected:
        raise ValueError(
            "--output-dir must be a dedicated subdirectory inside the project; "
            f"refusing unsafe path: {resolved}")
    return resolved


def build(args):
    output = validate_output_path(args.output_dir)
    results_root = args.results_root.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise RuntimeError(
                f"Output directory is non-empty: {output}; pass --force to refresh it")
        # This is restricted to the explicitly named generated release directory.
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT), "results_root": str(results_root),
        "files": [], "results": {}, "excluded": {
            "quick_results": "diagnostic only; omitted from primary evidence",
            "single_piece_results": "not representative of the corpus",
            "legacy_exploration": "omitted; retained in source project",
            "raw_external_data": "use DATASETS.md and data manifest; check licence",
        },
    }
    for filename in CORE_SCRIPTS:
        copy_file(ROOT / "scripts" / filename, output / "code/scripts" / filename)
    for filename in CORE_SRC:
        copy_file(ROOT / "src" / filename, output / "code/src" / filename)
    for source in sorted((ROOT / "tests").glob("test_*.py")):
        copy_file(source, output / "tests" / source.name)
    (output / "pytest.ini").write_text(
        "[pytest]\npythonpath =\n    code\n    code/src\n    code/scripts\n",
        encoding="utf-8")
    for filename in DOCS:
        if (ROOT / filename).is_file():
            copy_file(ROOT / filename, output / filename)

    for label, relative in RESULTS.items():
        source = ROOT / relative
        if source.is_dir():
            copy_tree(source, output / "results" / label)
            if label == "basic_greedy":
                metric = "boundary_per_piece.csv"
            elif label == "intermediate_dp":
                metric = "per_work_results.csv"
            else:
                metric = "held_out_per_work.csv"
            manifest["results"][label] = {
                "source": str(source),
                "status": "supporting" if "supporting" in label else "complete",
                "coverage": coverage(source / metric),
            }

    manifest_dir = output / "data_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("data_manifest.csv", "data_manifest_summary.json"):
        source = results_root / filename
        if source.is_file():
            copy_file(source, manifest_dir / filename)

    adv_status, detail = advanced_status(results_root)
    manifest["advanced"] = detail
    adv_destination = output / "results/advanced_neural"
    if adv_status == "complete":
        copy_tree(results_root / "neural", adv_destination)
    else:
        adv_destination.mkdir(parents=True, exist_ok=True)
        (adv_destination / "RESULT_STATUS.md").write_text(
            "# Advanced result status\n\n"
            "The formal 16-work Advanced result is pending and is not a main "
            "dissertation result. Quick runs are diagnostic only.\n\n"
            + "\n".join(f"- {reason}" for reason in detail["reasons"]) + "\n",
            encoding="utf-8")
        neural_figures = output / "figures/neural"
        neural_figures.mkdir(parents=True, exist_ok=True)
        (neural_figures / "RESULT_STATUS.md").write_text(
            "# Neural pitch-scape status\n\n"
            "Neural representative figures are pending the complete 16-work "
            "Advanced run and are not included as primary paper figures.\n",
            encoding="utf-8")
    for relative, source_relative in FIGURES.items():
        source = ROOT / source_relative
        if source.is_file():
            copy_file(source, output / "figures" / relative)
            source_metadata = source.with_suffix(".json")
            if source_metadata.is_file():
                copy_file(source_metadata,
                          (output / "figures" / relative).with_suffix(".json"))
    parametric_cases = ROOT / "results/paper_parametric_cases"
    formal_cases = ROOT / 'results/paper_pitchscapes/cases'
    if formal_cases.is_dir():
        copy_tree(formal_cases, output / 'figures/formal_held_out_cases')
    if parametric_cases.is_dir():
        copy_tree(parametric_cases, output / "figures/dp_parametric_cases")
    (output / "figures/FIGURE_INDEX.md").write_text(
        "# Paper figure index\n\n"
        "## Handcrafted-distance comparison (ABC/DCML, Op.95)\n\n"
        "The greedy and dp directories contain separate key-profile, "
        "circle-of-fifths and tonic-weighted trees using the same piece and "
        "8-qb leaves. These figures compare search and distance-function "
        "effects; DCML local-key labels are not used to construct the trees.\n\n"
        "## Trained parametric-distance cases (Taking Form)\n\n"
        "The dp_parametric_cases directory contains separate Greedy and "
        "exact-DP images and paired overview images for three deterministically "
        "selected held-out cases. See CASE_STUDIES.md and case_selection.csv. "
        "Gold markers are expert level-1 boundaries used only for visual "
        "reference and evaluation.\n\n"
        "## Neural-distance comparison\n\n"
        "Formal neural figures remain pending until the audited 16-work "
        "Advanced experiment is complete. Quick checkpoints are intentionally "
        "excluded.\n",
        encoding="utf-8")

    release_readme = output / "README.md"
    if adv_status == 'complete':
        (output / 'figures/FIGURE_INDEX.md').write_text(
            '# Paper figure index\n\n'
            '## Formal ABC/DCML comparisons\n\n'
            'The greedy, dp and neural directories contain the Op.95 '
            'comparison. The formal_held_out_cases directory contains '
            'separate key-profile+DP, Neural+Greedy and Neural+DP figures '
            'reconstructed from audited outer-fold models. Case selection '
            'is post-hoc illustration and does not replace corpus-level '
            'statistics. Gold-coloured triangles denote predicted '
            'boundaries, not ground truth.\n\n'
            '## Trained parametric-distance cases\n\n'
            'The dp_parametric_cases directory contains supplementary '
            'Taking Form comparisons under their separately documented '
            'evaluation protocol.\n',
            encoding='utf-8')
    # Always start from the source README so repeated builds are idempotent.
    existing = (ROOT / "README.md").read_text(encoding="utf-8")
    release_readme.write_text(
        existing + "\n\n## Paper release status\n\n"
        f"Advanced status: **{adv_status}**.\n\n"
        "Boundary F1 is the primary metric. DCML local-key annotations are "
        "flat boundaries, not hierarchical ground truth. Advanced is deep "
        "Siamese metric learning plus exact DP, not reinforcement learning.\n",
        encoding="utf-8")
    manifest_path = output / "RELEASE_MANIFEST.json"
    for path in sorted(output.rglob("*")):
        if path == manifest_path or not path.is_file():
            continue
        manifest["files"].append({"path": path.relative_to(output).as_posix(),
                                  "sha256": sha256_file(path),
                                  "bytes": path.stat().st_size})
    manifest["release_status"] = "complete" if adv_status == "complete" else "advanced_pending"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    args = parse_args()
    manifest = build(args)
    print(f"Paper release: {args.output_dir.resolve()}")
    print(f"Release status: {manifest['release_status']}")
    print(f"Files: {len(manifest['files'])}")
    for reason in manifest["advanced"]["reasons"]:
        print(f"Advanced pending: {reason}")
