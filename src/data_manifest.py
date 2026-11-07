"""Deterministic ABC/DCML discovery and checksum audit.

The manifest is deliberately independent of model code.  It records every
notes/harmony stem, including incomplete pairs, so exclusions cannot silently
change between dissertation runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def work_id(piece: str) -> str:
    """Return the quartet/work identifier used by grouped evaluation."""
    return piece.rsplit("_", 1)[0]


def build_abc_manifest(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root).resolve()
    notes_root = root / "notes"
    harmony_root = root / "harmonies"
    notes = {
        path.name.removesuffix(".notes.tsv"): path
        for path in notes_root.glob("*.notes.tsv")
    }
    harmonies = {
        path.name.removesuffix(".harmonies.tsv"): path
        for path in harmony_root.glob("*.harmonies.tsv")
    }
    rows = []
    for piece in sorted(set(notes) | set(harmonies)):
        note_path = notes.get(piece)
        harmony_path = harmonies.get(piece)
        complete = note_path is not None and harmony_path is not None
        rows.append({
            "dataset": "ABC_DCML",
            "work": work_id(piece),
            "piece": piece,
            "notes_path": str(note_path or ""),
            "harmonies_path": str(harmony_path or ""),
            "status": "included" if complete else "excluded",
            "reason": "" if complete else "notes_or_harmonies_missing",
            "notes_sha256": sha256_file(note_path) if note_path else "",
            "harmonies_sha256": sha256_file(harmony_path) if harmony_path else "",
        })
    return pd.DataFrame(rows, columns=[
        "dataset", "work", "piece", "notes_path", "harmonies_path",
        "status", "reason", "notes_sha256", "harmonies_sha256",
    ])


def manifest_summary(frame: pd.DataFrame) -> dict[str, int]:
    included = frame[frame.status == "included"] if not frame.empty else frame
    return {
        "discovered_pieces": int(len(frame)),
        "included_pieces": int(len(included)),
        "excluded_pieces": int((frame.status == "excluded").sum()) if not frame.empty else 0,
        "included_works": int(included.work.nunique()) if not included.empty else 0,
    }
