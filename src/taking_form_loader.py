"""Validated loader for Taking Form's hierarchical tabular annotations.

Comparator rows such as ``9-16=1-8`` copy the source interval's annotation
events into a fully written-out destination interval.  ``Repeat:`` rows denote
score repeat signs and are audited but do not invent extra notated score time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


@dataclass
class FormNode:
    label: str
    level: int
    start_measure: int
    end_measure: Optional[int] = None
    children: List["FormNode"] = field(default_factory=list)

    def add_child(self, child: "FormNode") -> None:
        self.children.append(child)

    def pretty_print(self, indent: int = 0) -> str:
        span = f"m.{self.start_measure}"
        if self.end_measure is not None:
            span += f"-{self.end_measure}"
        return ("  " * indent + f"[{self.label}] ({span})\n" +
                "".join(child.pretty_print(indent + 1) for child in self.children))

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)


@dataclass(frozen=True)
class TakingFormEvent:
    measure: int
    beat: str
    level: int
    label: str
    provenance: str = "explicit"
    source_measure: Optional[int] = None


@dataclass(frozen=True)
class TakingFormAudit:
    path: str
    last_measure: int
    explicit_events: int
    copied_events: int
    comparator_rows: int
    score_repeat_rows: int
    issues: tuple[str, ...] = ()


_COMPARATOR = re.compile(
    r"^\s*(?P<d0>\d+)(?:-(?P<d1>\d+))?\s*=\s*"
    r"(?P<s0>\d+)(?:-(?P<s1>\d+))?\s*$")
_PLAIN = re.compile(r"^\s*(\d+)(?:[a-z])?\s*$", re.IGNORECASE)
_REPEAT = re.compile(r"^\s*Repeat:\s*(\d+)(?:-(\d+))?\s*$", re.IGNORECASE)


def _parse_measure_field(value: str) -> Optional[int]:
    value = str(value).strip()
    match = _PLAIN.match(value) or _COMPARATOR.match(value)
    return int(match.group(1)) if match else None


def _parse_repeat_source(value: str) -> Optional[str]:
    match = _COMPARATOR.match(str(value).strip())
    if not match:
        return None
    end = match.group("s1")
    return f"m.{match.group('s0')}" + (f"-{end}" if end else "")


def _read_rows(csv_path: str | Path) -> list[list[str]]:
    frame = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"empty Taking Form file: {csv_path}")
    first = str(frame.iloc[0, 0]).strip()
    if not (_PLAIN.match(first) or _COMPARATOR.match(first) or _REPEAT.match(first)):
        frame = frame.iloc[1:].reset_index(drop=True)
    return [[str(value).strip() for value in row]
            for row in frame.itertuples(index=False, name=None)]


def read_taking_form_events(csv_path: str | Path
                            ) -> tuple[list[TakingFormEvent], TakingFormAudit]:
    """Expand comparator shorthand into a monotonic event table."""
    rows = _read_rows(csv_path)
    events: dict[tuple[int, int], TakingFormEvent] = {}
    comparator_rows = score_repeat_rows = copied_events = 0
    issues: list[str] = []
    last_measure = 0

    for row_number, row in enumerate(rows, 1):
        token = row[0]
        repeat = _REPEAT.match(token)
        if repeat:
            score_repeat_rows += 1
            last_measure = max(last_measure, int(repeat.group(2) or repeat.group(1)))
            continue

        comparator = _COMPARATOR.match(token)
        plain = _PLAIN.match(token)
        if not comparator and not plain:
            issues.append(f"row {row_number}: unsupported measure token {token!r}")
            continue

        if comparator:
            comparator_rows += 1
            destination_start = int(comparator.group("d0"))
            destination_end = int(comparator.group("d1") or destination_start)
            source_start = int(comparator.group("s0"))
            source_end = int(comparator.group("s1") or source_start)
            if destination_end - destination_start != source_end - source_start:
                issues.append(f"row {row_number}: comparator ranges have unequal lengths")
                continue
            if source_start >= destination_start:
                issues.append(f"row {row_number}: comparator source is not earlier")
                continue
            source = [event for event in events.values()
                      if source_start <= event.measure <= source_end]
            for event in sorted(source, key=lambda value: (value.measure, value.level)):
                destination = destination_start + event.measure - source_start
                key = (destination, event.level)
                if key not in events:
                    events[key] = TakingFormEvent(
                        destination, event.beat, event.level, event.label,
                        provenance="comparator_copy", source_measure=event.measure)
                    copied_events += 1
            measure = destination_start
            last_measure = max(last_measure, destination_end)
        else:
            measure = int(plain.group(1))
            last_measure = max(last_measure, measure)

        beat = row[1] if len(row) > 1 else ""
        for column, label in enumerate(row[2:], start=1):
            if not label or label.lower() == "nan":
                continue
            events[(measure, column)] = TakingFormEvent(
                measure, beat, column, label, provenance="explicit")

    ordered = sorted(events.values(), key=lambda value: (value.measure, value.level))
    explicit = sum(event.provenance == "explicit" for event in ordered)
    audit = TakingFormAudit(
        str(Path(csv_path)), last_measure, explicit, copied_events,
        comparator_rows, score_repeat_rows, tuple(issues))
    return ordered, audit


def form_tree_from_events(events: Sequence[TakingFormEvent], last_measure: int) -> FormNode:
    if last_measure < 1:
        raise ValueError("last_measure must be positive")
    root = FormNode("ROOT", 0, 1, last_measure)
    max_level = max((event.level for event in events), default=0)
    open_nodes: list[Optional[FormNode]] = [root] + [None] * max_level
    for event in events:
        depth = event.level
        for level in range(depth, len(open_nodes)):
            node = open_nodes[level]
            if node is not None and node.end_measure is None:
                node.end_measure = max(node.start_measure, event.measure - 1)
        for level in range(depth + 1, len(open_nodes)):
            open_nodes[level] = None
        parent = next((open_nodes[level] for level in range(depth - 1, -1, -1)
                       if open_nodes[level] is not None), root)
        node = FormNode(event.label, depth, event.measure)
        parent.add_child(node)
        open_nodes[depth] = node
    for node in open_nodes:
        if node is not None and node.end_measure is None:
            node.end_measure = last_measure
    return root


def load_taking_form_csv(csv_path: str | Path) -> FormNode:
    events, audit = read_taking_form_events(csv_path)
    if audit.issues:
        raise ValueError("; ".join(audit.issues))
    return form_tree_from_events(events, audit.last_measure)
