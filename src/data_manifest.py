"""Deterministic source discovery, pairing, and checksum audit."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pandas as pd

from taking_form_loader import read_taking_form_events
from corpus_loaders import read_fugue_dez


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _score_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted({*root.rglob("*.krn"), *root.rglob("*.musicxml"),
                   *root.rglob("*.mxl"), *root.rglob("*.xml")})


def _taking_key(path: Path) -> tuple[int, int] | None:
    match = re.search(r"sonata\s*0*(\d+).*?movt\s*0*(\d+)", path.stem, re.I)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _taking_score_key(path: Path) -> tuple[int, int] | None:
    name = path.stem.lower()
    patterns = [
        r"sonata[-_ ]*0*(\d+)[^\d]+(?:movement|movt|m)?[-_ ]*0*(\d+)",
        r"(?:^|[^\d])0*(\d+)[-_]0*(\d+)(?:[^\d]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _fugue_key(path: Path) -> int | None:
    match = re.match(r"0*(\d+)-bwv\d+-ref$", path.stem, re.I)
    return int(match.group(1)) if match else None


def _fugue_score_key(path: Path) -> int | None:
    name = path.stem.lower()
    patterns = [
        r"(?:fugue|fug|f)[-_ ]*0*(\d+)",
        r"wtc[-_ ]*(?:i|1)[-_ ]*(?:fugue|fug|f)?[-_ ]*0*(\d+)",
        r"0*(\d+)[-_ ]*(?:fugue|fug|f)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 24:
                return value
    return None


def _paired_rows(annotation_paths, score_paths, annotation_key, score_key,
                 dataset, annotation_validator):
    score_index: dict[object, list[Path]] = {}
    for score in score_paths:
        key = score_key(score)
        if key is not None:
            score_index.setdefault(key, []).append(score)
    rows = []
    for annotation in sorted(annotation_paths):
        key = annotation_key(annotation)
        matches = score_index.get(key, [])
        status = "included"
        reason = ""
        try:
            annotation_validator(annotation)
        except Exception as error:
            status, reason = "excluded", f"annotation_invalid: {error}"
        if key is None:
            status, reason = "excluded", "annotation_filename_unrecognised"
        elif not matches:
            status, reason = "excluded", "paired_score_missing"
        elif len(matches) > 1:
            status, reason = "excluded", "paired_score_ambiguous"
        score = matches[0] if len(matches) == 1 else None
        rows.append({
            "dataset": dataset,
            "work_id": str(key[0] if isinstance(key, tuple) else key),
            "piece_id": annotation.stem,
            "annotation_path": str(annotation), "score_path": str(score or ""),
            "status": status, "reason": reason,
            "annotation_sha256": sha256_file(annotation),
            "score_sha256": sha256_file(score) if score else "",
        })
    return rows


def build_data_manifest(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    rows = []
    abc = root / "external" / "ABC"
    notes = {path.name.replace(".notes.tsv", ""): path
             for path in (abc / "notes").glob("*.notes.tsv")}
    harmony = {path.name.replace(".harmonies.tsv", ""): path
               for path in (abc / "harmonies").glob("*.harmonies.tsv")}
    for piece in sorted(set(notes) | set(harmony)):
        complete = piece in notes and piece in harmony
        rows.append({
            "dataset": "abc", "work_id": piece.rsplit("_", 1)[0],
            "piece_id": piece, "annotation_path": str(harmony.get(piece, "")),
            "score_path": str(notes.get(piece, "")),
            "status": "included" if complete else "excluded",
            "reason": "" if complete else "notes_or_harmony_missing",
            "annotation_sha256": sha256_file(harmony[piece]) if piece in harmony else "",
            "score_sha256": sha256_file(notes[piece]) if piece in notes else "",
        })

    taking_root = root / "external" / "Taking-Form"
    taking_annotations = list(
        (taking_root / "corpus" / "Beethoven_Sonatas").glob("*.csv"))
    taking_scores_root = root / "external" / "beethoven-piano-sonatas"
    rows.extend(_paired_rows(
        taking_annotations, _score_files(taking_scores_root),
        _taking_key, _taking_score_key, "taking_form",
        lambda path: read_taking_form_events(path)))

    fugue_root = root / "external" / "algomus-data" / "fugues" / "bach-wtc-i"
    fugue_annotations = list(fugue_root.glob("*-ref.dez"))
    fugue_scores_root = root / "external" / "bach-wtc"
    rows.extend(_paired_rows(
        fugue_annotations, _score_files(fugue_scores_root),
        _fugue_key, _fugue_score_key, "algomus_fugue",
        lambda path: read_fugue_dez(path)))
    frame = pd.DataFrame(rows)
    revisions = {
        "abc": git_revision(abc),
        "taking_form": git_revision(taking_root),
        "algomus_fugue": git_revision(root / "external" / "algomus-data"),
    }
    frame["annotation_revision"] = frame.dataset.map(revisions).fillna("")
    frame["score_revision"] = frame.dataset.map({
        "abc": revisions["abc"],
        "taking_form": git_revision(taking_scores_root),
        "algomus_fugue": git_revision(fugue_scores_root),
    }).fillna("")
    return frame
