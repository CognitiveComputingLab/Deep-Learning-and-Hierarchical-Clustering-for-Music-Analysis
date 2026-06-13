"""
Loader for Taking Form CSV files.

Taking Form's tabular format:
- Column 0: measure number (or 'Repeat:' / range like '9-16=1-8')
- Column 1: beat number
- Columns 2+: formal labels from coarsest (left) to finest (right)

We convert this into a labelled tree where each node has:
- label (str): e.g., 'Exposition', 'Theme a'
- level (int): depth from root (0 = root = full piece)
- start_measure (int)
- end_measure (int)
- children (list)
"""
import os
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FormNode:
    """A node in a hierarchical form analysis tree."""
    label: str
    level: int
    start_measure: int
    end_measure: Optional[int] = None  # filled in after we know the next sibling/parent's start
    children: List['FormNode'] = field(default_factory=list)
    
    def add_child(self, child: 'FormNode'):
        self.children.append(child)
    
    def pretty_print(self, indent: int = 0) -> str:
        span = f"m.{self.start_measure}"
        if self.end_measure is not None:
            span += f"-{self.end_measure}"
        s = "  " * indent + f"[{self.label}] ({span})\n"
        for child in self.children:
            s += child.pretty_print(indent + 1)
        return s
    
    def size(self) -> int:
        """Count total nodes in subtree."""
        return 1 + sum(c.size() for c in self.children)
    

def _parse_measure_field(value: str) -> Optional[int]:
    """
    Parse Taking Form's measure column, which may contain:
      - Plain integer: '125'
      - Range notation: '125-132=1-8' (means "m.125-132 repeats m.1-8")
      - 'Repeat:' marker rows (in some files)
    
    Returns the first integer found, or None if no integer can be extracted.
    """
    value = str(value).strip()
    # Try the simple case first
    try:
        return int(value)
    except ValueError:
        pass
    # Try to extract the first integer using regex
    match = re.match(r'^\s*(\d+)', value)
    if match:
        return int(match.group(1))
    return None


def load_taking_form_csv(csv_path: str) -> FormNode:
    """
    Parse a Taking Form CSV file into a FormNode tree.
    
    Returns the root node (representing the whole piece).
    """
    df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
    
    # Detect if first row is a header
    try:
        int(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
    
    n_levels = df.shape[1] - 2
    
    # ── Find the last measure number in the file ─────────────────
    # We need this to close any still-open nodes at the end.
    last_measure = None
    for _, row in df.iterrows():
        m = _parse_measure_field(row[0])
        if m is not None:
            last_measure = m
    if last_measure is None:
        last_measure = 1  # defensive default
    
    # Root represents the whole piece.
    root = FormNode(label="ROOT", level=0, start_measure=1)
    
    open_nodes: List[Optional[FormNode]] = [root] + [None] * n_levels
    
    for _, row in df.iterrows():
        measure = _parse_measure_field(row[0])
        if measure is None:
            continue  # Skip rows where we can't extract a measure number
        
        for col_idx in range(2, df.shape[1]):
            cell = str(row[col_idx]).strip()
            if not cell or cell.lower() == 'nan':
                continue
            
            depth = col_idx - 1
            
            # Close previously-open node at this depth (and deeper)
            for d in range(depth, len(open_nodes)):
                if open_nodes[d] is not None and open_nodes[d].end_measure is None:
                    open_nodes[d].end_measure = measure - 1
            
            for d in range(depth + 1, len(open_nodes)):
                open_nodes[d] = None
            
            parent = open_nodes[depth - 1] if depth > 0 else root
            if parent is None:
                parent = root
            new_node = FormNode(label=cell, level=depth, start_measure=measure)
            parent.add_child(new_node)
            open_nodes[depth] = new_node
    
    # ── Close any remaining open nodes with last_measure ──────────
    # This fixes the bug where the last node of each level lacks end_measure.
    for d in range(len(open_nodes)):
        if open_nodes[d] is not None and open_nodes[d].end_measure is None:
            open_nodes[d].end_measure = last_measure
    
    # Also set root's end_measure
    if root.end_measure is None:
        root.end_measure = last_measure
    
    return root

if __name__ == "__main__":
    # Find one CSV file from the Beethoven corpus
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    BEETHOVEN_DIR = os.path.join(REPO_ROOT, "external/Taking-Form/corpus/Beethoven_Sonatas")
    
    csv_files = sorted(f for f in os.listdir(BEETHOVEN_DIR) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} CSV files in Beethoven corpus.")
    
    # Load the first one
    target = csv_files[0]
    print(f"\nLoading: {target}")
    tree = load_taking_form_csv(os.path.join(BEETHOVEN_DIR, target))
    
    print(f"\nTotal nodes: {tree.size()}")
    print(f"\nTree structure:")
    print(tree.pretty_print())